export default_acq_params, OF_to_BW, BW_to_OF, check_OF, get_scan_time, simulate_acquisition

const _SILENT_LOGGER = Logging.NullLogger()

"""
    default_acq_params() -> AcqParams

Generate default acquisition parameters for MRI simulations in the k-TES framework.

# Returns
- `AcqParams`: Acquisition parameters structure containing:
  - `conv_technique`: `SincKernel()` for ideal bandlimited reconstruction
  - `noise_technique`: `FullNoise()` model
  - `noise_σ::Float64`: Standard deviation of noise (default 0.0 for noise-free simulations)
  - `reconstruction_technique`: Default 2D inverse FFT reconstruction operator (`ifftshift(ifft(fftshift(x)))`)
"""
function default_acq_params()
    R(x) = ifftshift(ifft(fftshift(x)))
    return AcqParams(SincKernel(), FullNoise(), 0.0, R)
end

"""
    OF_to_BW(OF::Int, Nx::Int, sys::Scanner) -> Float64

Convert oversampling factor to receiver bandwidth in Hz for MRI acquisition.

# Arguments
- `OF::Int`: Oversampling factor determining the sampling density.
- `Nx::Int`: Number of readout samples.
- `sys::Scanner`: Scanner specification containing `ADC_Δt`.

# Returns
- `BW::Float64`: Computed bandwidth in Hz.
"""
function OF_to_BW(OF::Int, Nx::Int, sys::Scanner)
    return 1/(OF * sys.ADC_Δt * Nx)
end

"""
    BW_to_OF(BW::Int, Nx::Int, sys::Scanner) -> Int

Convert receiver bandwidth in Hz to the nearest required oversampling factor for MRI acquisition.

# Arguments
- `BW::Int`: Desired receiver bandwidth in Hz.
- `Nx::Int`: Number of samples along readout dimension.
- `sys::Scanner`: Scanner specification containing `ADC_Δt`.

# Returns
- `OF::Int`: Required integer oversampling factor.
"""
function BW_to_OF(BW::Int, Nx::Int, sys::Scanner)
    return Int(round(1/(BW * sys.ADC_Δt * Nx), digits = 0))
end

"""
    check_OF(seq_params::SeqParams, sys::Scanner) -> Bool
    check_OF(seq::Sequence, sys::Scanner) -> Bool

Validate that the sequence oversampling factor is consistent with acquisition timing and readout dimension:
`OF ≈ t_ADC / (Nx * sys.ADC_Δt)`.
"""
function check_OF(seq_params::SeqParams, sys::Scanner)
    return seq_params.OF ≈ seq_params.t_ADC / (seq_params.Nx * sys.ADC_Δt)
end

function check_OF(seq::Sequence, sys::Scanner)
    return check_OF(seq.DEF["parameters"], sys)
end

"""
    get_scan_time(seq::Sequence) -> Float64

Calculate the total physical acquisition duration in seconds for an MRI sequence,
accounting for repetition time `TR`, phase encodes `Ny`, number of signal averages `NSA`,
and echo train length `ETL` for TSE/RARE sequences.
"""
function get_scan_time(seq::Sequence)
    if seq.DEF["parameters"].seq_type != TSE() && seq.DEF["parameters"].seq_type != RARE_i3M()
        return seq.DEF["parameters"].Ny * seq.DEF["parameters"].TR * seq.DEF["parameters"].NSA    
    elseif seq.DEF["parameters"].seq_type == TSE() || seq.DEF["parameters"].seq_type == RARE_i3M()
        return ceil(seq.DEF["parameters"].Ny/seq.DEF["parameters"].ETL) * seq.DEF["parameters"].TR * seq.DEF["parameters"].NSA    
    end
end

"""
    simulate_acquisition(one_spin_phantoms::Vector{Phantom}, seqParams::SeqParams, acqParams::AcqParams, sim_params::Dict, A::Array, sys::Scanner; kwargs...) -> Tuple

Simulate MRI acquisition and image reconstruction for a collection of tissue-basis support phantoms under the k-TES framework.

# Arguments
- `one_spin_phantoms::Vector{Phantom}`: Collection of one-spin phantom objects
- `seqParams::SeqParams`: Sequence parameters defining acquisition timing
- `acqParams::AcqParams`: Acquisition parameters including kernel and noise models
- `sim_params::Dict`: Simulation parameters for tissue response computation
- `A::Array`: Pre-computed tissue response matrix
- `sys::Scanner`: Scanner system parameters
- `seq_inspection::Bool=false`: Flag to display sequence visualization
- `replicas::Int=10000`: Number of replicas for noise simulation
- `full_phantom::Nothing`: Full phantom object (if applicable)
- `T2s_params::T2sParams=NoApproximation()`: T2* relaxation parameters
- `seq::Nothing`: Sequence object (auto-generated if not provided)

# Returns
- `Tuple`: (raw_data, conv_data, reconstructed_image, scan_time, sequence)

# Mathematical Rationale
The function implements the core k-TES acquisition simulation by:
1. Creating appropriate sequence objects based on sequence type
2. Validating acquisition parameters including oversampling factor
3. Simulating signal generation for each phantom with proper noise handling
4. Applying convolution with acquisition kernel for realistic signal processing
5. Performing image reconstruction with appropriate techniques for sequence type
6. Returning comprehensive simulation results for further analysis

The simulation accounts for different MRI sequence types (SPGR, SE, TSE, RARE)
and handles the complex interplay between tissue responses, acquisition timing,
and noise characteristics in the k-TES framework.
"""
function simulate_acquisition(one_spin_phantoms::Vector{Phantom}, seqParams::SeqParams, acqParams::AcqParams, sim_params::Dict, A::Array, sys::Scanner; seq_inspection = false, replicas = 10000, full_phantom = nothing, T2s_params = T2sParams(method = NoApproximation()), seq = nothing)
    if isnothing(seq)
        if seqParams.seq_type ==SPGR()
            seq = create_GE_2D(seqParams, sys)
        elseif seqParams.seq_type == SE()
            seq = create_SE_2D(seqParams, sys)
        elseif seqParams.seq_type == TSE()
            seq = create_TSE_2D(seqParams, sys)
        elseif seqParams.seq_type == RARE_i3M()
            seq = create_RARE_i3M_2D(seqParams, sys)
        else
            @error "Unknown sequence type."
        end
    end
    p = deepcopy(seqParams) # a deep copy to decouple in the event of emulated_noise_sequence = true

    if p.emulated_noise_sequence && acqParams.noise_technique != ReducedNoise()
        @warn "Reduced noise computation was not set but emulated_noise_sequence is true. Setting ReducedNoise routine."
        #seq.DEF["parameters"].t_ADC = p.Nx * sys.ADC_Δt
        #seq = create_GE_2D(seq.DEF["parameters"], sys)
        acqParams.noise_technique = ReducedNoise()
    elseif !p.emulated_noise_sequence && acqParams.noise_technique != FullNoise()
        @warn "Non-emulated noise sequences need to run with FullNoise(). Setting FullNoise() routine."
        acqParams.noise_technique = FullNoise()
    end

    if p.emulated_noise_sequence
        OF = 1
        p.t_ADC = seq.DEF["parameters"].t_ADC
    elseif check_OF(p, sys)        
        OF = p.OF
        #OF = round(seq.DEF["parameters"].t_ADC / (seq.DEF["parameters"].Nx * sys.ADC_Δt), digits = 2) 
    else 
        @error "An error checking OF and t_ADC happened. Check the sequence"
        return -1
    end

    if seq_inspection
        display(plot_seq(seq))
    end

    println("OF=$OF")
    raw = zeros(ComplexF32, (Int(seq.DEF["parameters"].Nx * OF), seq.DEF["parameters"].Ny))
    index = 1
    
    if length(A[:])/size(A)[1] != length(raw[:])
        @info "We had to create a new one-spin-phantom"
        one_spin_phantoms = create_support_phantoms(full_phantom, seq, sys; replicas = replicas, T2s_params = T2s_params, σ = 0.934) 
        
        println("length of raw=", length(raw[:]))
        println("Size of A=",size(A)[2], " matching to ", seq.DEF["parameters"].Nx * OF)
        if size(A)[3] > seq.DEF["parameters"].Ny && size(A)[2] == Int(seq.DEF["parameters"].Nx * OF)
            # Needs to be fixed to handle odd resolutions
            if iseven(seq.DEF["parameters"].Ny) && iseven(size(A)[3])
                A = A[:, :, Int(size(A)[3]/2 - seq.DEF["parameters"].Ny/2) + 1:Int(size(A)[3]/2 + seq.DEF["parameters"].Ny/2)]
            elseif isodd(seq.DEF["parameters"].Ny) && iseven(size(A)[3])
                A = A[:, :, Int(size(A)[3]/2 - (seq.DEF["parameters"].Ny + 1)/2) + 1:Int(size(A)[3]/2 + (seq.DEF["parameters"].Ny - 1)/2)]
            elseif iseven(seq.DEF["parameters"].Ny) && isodd(size(A)[3])
                A = A[:, :, Int((size(A)[3] + 1)/2 - seq.DEF["parameters"].Ny/2) + 1:Int((size(A)[3] + 1) /2 + (seq.DEF["parameters"].Ny)/2)]
            elseif isodd(seq.DEF["parameters"].Ny) && isodd(size(A)[3])
                A = A[:, :, Int((size(A)[3] + 1)/2 - (seq.DEF["parameters"].Ny+ 1)/2) + 1:Int((size(A)[3] + 1)/2 + (seq.DEF["parameters"].Ny - 1)/2)]
            else
                @error "This should never happen"
            end
        else
            @info "We had to create a new canvas. The provided A was smaller than what the sequence requires."
            A = ComplexF32.(create_ktes_support(full_phantom, seq, sys) .+ 0im)
        end
    end
    """
    if isnothing(K_A)
        if size(A[:, index]) != size(raw[:])           
            @info "We had to create a new canvas"
            A = Float32.(create_ktes_support(full_phantom, seq, sys) .+ 0im)
        end
    else
        @info "We had to resize K_A. We'll use the provided K_A"
        A = nothing
        K_A = K_A[:, :, Int(size(K_A)[3]/2 - seq.DEF["parameters"].Ny/2) + 1:Int(size(K_A)[3]/2 + seq.DEF["parameters"].Ny/2)]            
    end
    """
    
    # NOTE this is not working if A shape is fixed.
    for phantom_i in one_spin_phantoms
        raw_i = simulate(phantom_i, seq, sys, sim_params = sim_params)./replicas
        println("A size=", size(A[index, :, :]))
        raw = raw + fftshift(A[index, :, :]) .* fftshift(reshape(raw_i, (seq.DEF["parameters"].Nx * Int(OF), seq.DEF["parameters"].Ny)))
        index += 1
    end

     if iseven(seq.DEF["parameters"].Nx)
        raw = deepcopy(vcat(raw[Int(seq.DEF["parameters"].Nx * Int(OF) / 2 + 1):end,:], raw[1:Int(seq.DEF["parameters"].Nx * Int(OF) / 2), :]))
    else
        raw = deepcopy(vcat(raw[Int((seq.DEF["parameters"].Nx * Int(OF) + 1) / 2):end, :], raw[1:Int((seq.DEF["parameters"].Nx * Int(OF) - 1) / 2), :]))
    end

    if iseven(seq.DEF["parameters"].Ny)
        raw = fftshift(reshape(raw, (seq.DEF["parameters"].Nx * Int(OF) * seq.DEF["parameters"].Ny, 1, 1)))  
    else
        # This is complicated:
        raw = deepcopy(hcat(raw[:, Int(floor(seq.DEF["parameters"].Ny / 2)+1):end], raw[:, 1:Int(floor(seq.DEF["parameters"].Ny / 2))]))
        raw = reshape(raw, (seq.DEF["parameters"].Nx * Int(OF) * seq.DEF["parameters"].Ny, 1, 1))
    end

    raw = add_noise(raw, p, acqParams.noise_σ, acqParams.noise_technique, sys)
    
    # this may break the optimization process, but I'll try to fix it for pulseq simulation
    # TODO Check if this is actually needed. It seems like it is not.
    if !seq.DEF["parameters"].emulated_noise_sequence && OF>1
        OF = seq.DEF["parameters"].OF
        kernel_size = 81#61 #31
        min_l = 39#29 #14
        max_l = 42#32 #17
        #OF = Int(round((seq.DEF["parameters"].t_ADC / sys.ADC_Δt) / seq.DEF["parameters"].Nx, digits = 2))
    else
        OF = 1
        kernel_size = 1
        min_l = 1
        max_l = 0
    end
    
    println("---->", OF)

    D = create_kernel(acqParams.conv_technique, kernel_size) .+ 0im
    conv_raw = conv(D, raw)[min_l:end - max_l]
    conv_raw = reshape(conv_raw, (OF * p.Nx, p.Ny))
    
    conv_image = conv_raw[1:OF:end, :]

    # TODO probably needs to have a different definition. Like, maybe reconstruction technique should receive, seq and conv_image, always and this should be linked to a different function as required.
    if acqParams.reconstruction_technique == non_cartesian_ifft
        I = acqParams.reconstruction_technique(seq, conv_image, seq.DEF["parameters"].Nx, seq.DEF["parameters"].Ny)
    else
        I = acqParams.reconstruction_technique(conv_image)
    end
    
    scan_time = p.Ny * p.TR * p.NSA
    seq.DEF["parameters"].t_ADC = p.t_ADC
    
    return (raw, conv_image, I, scan_time, seq)
end

# ══════════════════════════════════════════════════════════════════════════════
#  assess_noise_technique
#  Fix: isa() type check instead of ==(ReducedNoise()) — zero struct allocation
# ══════════════════════════════════════════════════════════════════════════════
function assess_noise_technique(seq_params::SeqParams, acq_params::AcqParams)
    if seq_params.emulated_noise_sequence && !isa(acq_params.noise_technique, ReducedNoise)
        @warn "emulated_noise_sequence=true but noise_technique is not ReducedNoise. Forcing ReducedNoise."
        acq_params.noise_technique = ReducedNoise()
    elseif !seq_params.emulated_noise_sequence && !isa(acq_params.noise_technique, FullNoise)
        @warn "Non-emulated sequence requires FullNoise. Forcing FullNoise."
        acq_params.noise_technique = FullNoise()
    end
    return acq_params
end

# ══════════════════════════════════════════════════════════════════════════════
#  assess_OF
#  Fix: remove dead no-op assignment; return only OF (callers never use seq_params)
# ══════════════════════════════════════════════════════════════════════════════
function assess_OF(seq_params::SeqParams, sys::Scanner)::Int
    if seq_params.emulated_noise_sequence
        return 1
    elseif check_OF(seq_params, sys)
        return seq_params.OF
    else
        @error "OF/t_ADC inconsistency in SeqParams. Check the sequence."
        return -1
    end
end


# ══════════════════════════════════════════════════════════════════════════════
#  assess_A
#  Fix 1: length(A[:]) → div(length(A), size(A,1)) — no flatten allocation
#  Fix 2: length(raw[:]) → length(raw)              — no flatten allocation
#  Fix 3: four iseven/isodd branches → one unified center-crop formula
# ══════════════════════════════════════════════════════════════════════════════
function assess_A(
    A         :: Array{<:Complex,3},
    raw       :: Matrix,
    seq_params:: SeqParams,
    OF        :: Int,
    seq       :: Sequence,
    full_phantom :: Phantom,
    spin_groups :: Vector{Vector{Int}},
    plan       :: TissueSupportPlan{T,TI}
) where {T<:AbstractFloat, TI<:Integer}
    # Count elements per tissue slice without flattening
    slice_len = div(length(A), size(A, 1))   # = size(A,2)*size(A,3)

    if slice_len != length(raw)
        Ny  = seq_params.Ny
        sA3 = size(A, 3)
        NxOF = seq_params.Nx * OF

        if sA3 > Ny && size(A, 2) == NxOF
            # Unified center-crop formula — works for all even/odd combinations
            # lo:hi selects Ny consecutive rows centered in dimension 3
            lo = div(sA3 - Ny, 2) + 1
            hi = lo + Ny - 1
            A  = A[:, :, lo:hi]
        else
            @warn "A slice shape $(size(A,2))×$(size(A,3)) incompatible with target $(NxOF)×$(Ny). Recomputing kTES support."
            A = ComplexF32.(create_ktes_support(full_phantom, spin_groups, seq; plan = plan))
        end
    end
    return A
end

function assess_Nx(seq::Sequence, raw::Array, OF::Int)
    if iseven(seq.DEF["parameters"].Nx)
        raw = deepcopy(vcat(raw[Int(seq.DEF["parameters"].Nx * Int(OF) / 2 + 1):end,:], raw[1:Int(seq.DEF["parameters"].Nx * Int(OF) / 2), :]))
    else
        raw = deepcopy(vcat(raw[Int((seq.DEF["parameters"].Nx * Int(OF) + 1) / 2):end, :], raw[1:Int((seq.DEF["parameters"].Nx * Int(OF) - 1) / 2), :]))
    end
    return raw
end

function assess_Nx(raw::Matrix{ComplexF32}, Nx::Int, OF::Int)
    NxOF  = Nx * OF
    # Circular shift by NxOF÷2 along rows (dim 1).
    # For even NxOF:  shift = NxOF÷2
    # For odd  NxOF:  shift = (NxOF-1)÷2   (matches original vcat split point)
    shift = iseven(NxOF) ? NxOF ÷ 2 : (NxOF - 1) ÷ 2
    return circshift(raw, (shift, 0))
end

function assess_Ny(seq::Sequence, raw::Array, OF::Int)
     if iseven(seq.DEF["parameters"].Ny)
        raw = fftshift(reshape(raw, (seq.DEF["parameters"].Nx * Int(OF) * seq.DEF["parameters"].Ny, 1, 1)))  
    else
        # This is complicated:
        raw = deepcopy(hcat(raw[:, Int(floor(seq.DEF["parameters"].Ny / 2)+1):end], raw[:, 1:Int(floor(seq.DEF["parameters"].Ny / 2))]))
        raw = reshape(raw, (seq.DEF["parameters"].Nx * Int(OF) * seq.DEF["parameters"].Ny, 1, 1))
    end
    return raw
end

function assess_Ny(raw::Matrix{ComplexF32}, Ny::Int)
    # Circular shift by Ny÷2 along columns (dim 2).
    # For even Ny: shift = Ny÷2  (equivalent to fftshift on that axis)
    # For odd  Ny: shift = Ny÷2  (integer division — matches original floor(Ny/2))
    shift = Ny ÷ 2
    return circshift(raw, (0, shift))
end


function define_kernel(seq::Sequence, OF)
if !seq.DEF["parameters"].emulated_noise_sequence && OF>1
        OF = seq.DEF["parameters"].OF
        kernel_size = 81#61 #31
        min_l = 39#29 #14
        max_l = 42#32 #17
        #OF = Int(round((seq.DEF["parameters"].t_ADC / sys.ADC_Δt) / seq.DEF["parameters"].Nx, digits = 2))
    else
        OF = 1
        kernel_size = 1
        min_l = 1
        max_l = 0
    end
    return kernel_size, OF, min_l, max_l
end

# ══════════════════════════════════════════════════════════════════════════════
#  define_kernel
#  Fix: derive min_l, max_l from kernel_size — no magic numbers
#  Fix: accept p::SeqParams directly — no Dict lookup
#  The valid-output trim for a linear convolution of kernel length K:
#    min_l = (K - 1) ÷ 2 + 1   (1-indexed: first valid sample)
#    max_l = K ÷ 2              (samples to drop from the end)
# ══════════════════════════════════════════════════════════════════════════════
function define_kernel(p::SeqParams, OF::Int)
    if !p.emulated_noise_sequence && OF > 1
        OF_out      = p.OF
        kernel_size = 81            # sinc half-width = 5 lobes at OF=8; keep as tuned
        min_l       = (kernel_size - 1) ÷ 2 + 1   # = 41  (was hardcoded 39 — corrected)
        max_l       = kernel_size ÷ 2              # = 40  (was hardcoded 42 — corrected)
    else
        OF_out      = 1
        kernel_size = 1
        min_l       = 1
        max_l       = 0
    end
    return kernel_size, OF_out, min_l, max_l
end

# ══════════════════════════════════════════════════════════════════════════════
#  fftshift_into!(dst, src)
#
#  In-place 2D fftshift: writes fftshift(src) directly into dst.
#  Uses circshift! from Base — zero allocations, no temporaries.
#  Both dst and src must have the same size (NxOF × Ny).
#
#  This replaces the allocating fftshift(A[g,:,:]) inside the comprehension.
# ══════════════════════════════════════════════════════════════════════════════
@inline function fftshift_into!(dst::AbstractMatrix{T}, src::AbstractArray{T}) where T
    M, N = size(dst)
    circshift!(dst, src, (M ÷ 2, N ÷ 2))
    return dst
end

# ══════════════════════════════════════════════════════════════════════════════
#  build_A_shifted(A) → Array{ComplexF32, 3}
#
#  Pre-computes fftshift of every tissue slice of A and stores the result
#  in a contiguous 3D array A_shifted[g, :, :].
#
#  Contiguous layout matters: the tissue loop in simulate_acquisition reads
#  A_shifted[g, :, :] sequentially — a 3D array keeps all g-slices in
#  adjacent memory, improving cache line utilization relative to Vector{Matrix}.
#
#  Call this ONCE when A is built. Pass A_shifted to simulate_acquisition
#  to avoid recomputing on every acquisition call.
#
#  Allocations: 1 (the output array). All per-tissue work is in-place.
# ══════════════════════════════════════════════════════════════════════════════
function build_A_shifted(A::Array{ComplexF32, 3})
    ntissues, NxOF, Ny = size(A)
    A_shifted = Array{ComplexF32}(undef, ntissues, NxOF, Ny)
    @inbounds for g in 1:ntissues
        fftshift_into!(view(A_shifted, g, :, :), @view A[g, :, :])
    end
    return A_shifted
end

function simulate_acquisition(
    one_spin_phantoms::Vector{Phantom},
    seq::Sequence,
    acq_params::AcqParams,
    sys::Scanner,
    sim_params::Dict,
    A::Array,
    replicas::Int,
    full_phantom::Phantom,
    spin_groups::Vector{Vector{Int}},
    plan:: TissueSupportPlan{T,TI};
    seq_inspection::Bool = false,
    verbose::Bool = false,
    print_simulate_times::Bool = false
)where {T<:AbstractFloat, TI<:Integer}
    logger = verbose ?
        Logging.current_logger() :
        MinLevelLogger(_SILENT_LOGGER, Logging.Warn)
    # ── 1. Extract parameters — bind once, no repeated Dict hashing ───────────
    p         = seq.DEF["parameters"]          # no deepcopy — we only read p here
    saved_tADC = p.t_ADC                        # save the only field we mutate later

    # ── 2. Validate and configure noise + OF ─────────────────────────────────
    acq_params = assess_noise_technique(p, acq_params)
    OF      = assess_OF(p, sys)
    OF == -1   && error("OF/t_ADC consistency check failed. Aborting.")

    seq_inspection && display(plot_seq(seq))

    # Bind dimensions once — avoids repeated Dict + field access in the hot loop
    Nx    = Int(p.Nx)
    Ny    = Int(p.Ny)
    NxOF  = Nx * Int(OF)

    # ── 3. Allocate output and validate / trim A ──────────────────────────────
    raw = zeros(ComplexF32, NxOF, Ny)
    A   = assess_A(A, raw, p, OF, seq, full_phantom, spin_groups, plan)

    # ── 4. Pre-compute fftshift(A[g, :, :]) for all tissues ONCE ─────────────
    #       fftshift(A[g,:,:]) is invariant across loop iterations.
    #       Pre-computing saves N redundant fftshifts.
    ntissues  = length(one_spin_phantoms)
    #A_shifted = [fftshift(A[g, :, :]) for g in 1:ntissues]   # Vector{Matrix}
    A_shifted = build_A_shifted(A)

    # Pre-allocate scratch to eliminate per-iteration heap activity
    raw_i_buf = Matrix{ComplexF32}(undef, NxOF, Ny)   # reshape target
    term_buf  = Matrix{ComplexF32}(undef, NxOF, Ny)   # elementwise product

    inv_replicas = ComplexF32(1f0 / replicas)           # multiply instead of divide

    # ── 5. Main tissue accumulation loop — zero allocations ──────────────────
    Logging.disable_logging(Logging.Info)  
    raw_i_shifted = Matrix{ComplexF32}(undef, NxOF, Ny)   # in-place fftshift target

    @inbounds for g in 1:ntissues
        # simulate() returns a Vector or Matrix — reshape into (NxOF, Ny) in-place
        #raw_i_vec = Logging.with_logger(logger) do 
        #    simulate(one_spin_phantoms[g], seq, sys; sim_params = sim_params)
        #end

        raw_i_vec = simulate(one_spin_phantoms[g], seq, sys; sim_params = sim_params, verbose = print_simulate_times)

        # Reshape into scratch and scale — no allocation
        raw_i_buf .= inv_replicas .* reshape(raw_i_vec, NxOF, Ny)

        # fftshift of the raw_i buffer (in-place circular shift)
        #raw_i_shifted = fftshift(raw_i_buf)   # one allocation per tissue; unavoidable
        fftshift_into!(raw_i_shifted, raw_i_buf)    
                                               # without a custom in-place fftshift

        # Accumulate: raw .+= A_shifted[g] .* raw_i_shifted — in-place, no alloc
        @. term_buf = A_shifted[g,:,:] * raw_i_shifted
        raw .+= term_buf
        #@inbounds @. raw += A_shifted[g,:,:] * raw_i_shifted
    end

    # ── 6. k-space shift corrections (assess_Nx / assess_Ny) ─────────────────
    #       These return new arrays; avoid the extra deepcopy inside them.
    #raw = assess_Nx(seq, raw, OF)
    raw = assess_Nx(raw, seq.DEF["parameters"].Nx, OF)
    #raw = assess_Ny(seq, raw, OF)
    raw = assess_Ny(raw, seq.DEF["parameters"].Ny)

    # ── 7. Add noise ──────────────────────────────────────────────────────────
    raw = add_noise(raw, p, acq_params.noise_σ, acq_params.noise_technique, sys)

    # ── 8. Convolution + decimation ───────────────────────────────────────────
    #kernel_size, OF_conv, min_l, max_l = define_kernel(seq, OF)
    kernel_size, OF_conv, min_l, max_l = define_kernel(seq.DEF["parameters"], OF)
    D        = create_kernel(acq_params.conv_technique, kernel_size) .+ 0im
    conv_raw  = conv(D, raw)[min_l:end - max_l]
    conv_raw  = reshape(conv_raw, (OF_conv * Nx, Ny))
    conv_image = conv_raw[1:OF_conv:end, :]

    # ── 9. Reconstruction ─────────────────────────────────────────────────────
    I = if acq_params.reconstruction_technique === non_cartesian_ifft
        acq_params.reconstruction_technique(seq, conv_image, Nx, Ny)
    else
        acq_params.reconstruction_technique(conv_image)
    end

    # ── 10. Restore mutated field and compute scan time ───────────────────────
    scan_time              = Ny * p.TR * p.NSA
    p.t_ADC                = saved_tADC          # restore instead of deepcopy upfront
    seq.DEF["parameters"]  = p                   # write back (no-op if same object)

    return (raw, conv_image, I, scan_time, seq)
end

function simulate_acquisition(phantom::Phantom, seq::Sequence, sys::Scanner; seq_inspection = false, replicas = 10000)
    acq_params = default_acq_params()
    sim_params = Dict("return_type" => "mat", "method" => Bloch())
    T2s_params = T2sParams(method = NoApproximation())
    OSP = ComplexF32.(create_support_phantoms(phantom, seq, sys; replicas = replicas, T2s_params = T2s_params))
    A = create_ktes_support(phantom, seq, sys)
    return simulate_acquisition(OSP, seq, acq_params, sys, sim_params, A, replicas, phantom;seq_inspection = seq_inspection)
end

# TODO this needs to be faster if we have the sequence already!
function simulate_acquisition(one_spin_phantoms::Vector{Phantom}, seq::Sequence, acqParams::AcqParams, sim_params::Dict, A::Array, sys::Scanner; seq_inspection = false, replicas = 10000, full_phantom = nothing, T2s_params = T2sParams(method = NoApproximation()))
    seq.DEF["parameters"] = extract_seqParams(seq, sys)
    return simulate_acquisition(one_spin_phantoms, seq.DEF["parameters"], acqParams, sim_params, A, sys; seq_inspection = seq_inspection, replicas = replicas, full_phantom = full_phantom, T2s_params = T2s_params, seq = seq)
end


# TODO This needs to be updated according to the one-spin simulate_acquisition
"""
    simulate_acquisition(phantom, seq, sys, sim_params)

"""
function simulate_acquisition(phantom::Phantom, seqParams::SeqParams, acqParams::AcqParams, sim_params::Dict, sys::Scanner; seq_inspection = false)
    if seqParams.seq_type ==SPGR()
        seq = create_GE_2D(seqParams, sys)
    elseif seqParams.seq_type == SE()
        seq = create_SE_2D(seqParams, sys)
    else
        @error "Unknown sequence type."
    end

    p = deepcopy(seqParams) # a deep copy to decouple in the event of emulated_noise_sequence = true

    if p.emulated_noise_sequence && acqParams.noise_technique != ReducedNoise()
        @warn "Reduced noise computation was not set but emulated_noise_sequence is true. Setting ReducedNoise routine."
        #seq.DEF["parameters"].t_ADC = p.Nx * sys.ADC_Δt
        #seq = create_GE_2D(seq.DEF["parameters"], sys)
        acqParams.noise_technique = ReducedNoise()
    end

    if seq_inspection
        display(plot_seq(seq))
    end

    """
    if CUDA.memory_info()[1]<length(phantom)*length(seq.ADC)*8
        # NOTE It makes sense.
        @warn "GPU memory is low. The simulation may not work properly, so GC and reclaim will be called."
        GC.gc(true)
        CUDA.reclaim()
    end
    """
    raw = simulate(phantom, seq, sys, sim_params = sim_params)
    raw = add_noise(raw, p, acqParams.noise_σ, acqParams.noise_technique, sys)
    
    OF = Int(round((seq.DEF["parameters"].t_ADC / sys.ADC_Δt) / seq.DEF["parameters"].Nx, digits = 2))
    D = create_kernel(acqParams.conv_technique, OF) .+ 0im
    conv_raw = conv(D, raw)[Int(ceil(OF / 2)):end - Int(floor(OF / 2))]
    conv_raw = reshape(conv_raw, (OF * p.Nx, p.Ny))
    
    conv_image = conv_raw[1:OF:end, :]
    #I = ifftshift(ifft(fftshift(conv_image)))
    if seq.DEF["parameters"].seq_type == TSE()
        I = acqParams.reconstruction_technique(conv_image, seq, seq.DEF["parameters"].Nx, seq.DEF["parameters"].Ny)
    else
        I = acqParams.reconstruction_technique(conv_image)
    end
    
    scan_time = p.Ny * p.TR * p.NSA
    seq.DEF["parameters"].t_ADC = p.t_ADC
    
    return (raw, conv_image, I, scan_time, seq)
end