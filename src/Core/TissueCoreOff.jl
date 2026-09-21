# TissueCoreOff.jl
# Off-resonance–aware k-TES support: computes M_n^f(t_m) for macroscopic B0 inhomogeneity.
#
# Mathematical model (consistent with KomaMRI Bloch):
#   M̂_n^f(t_m) = env(t_m) * Σ_j  w_j * exp(-i2π t_m f_j) * exp(-i2π k(t_m)·x_j)
#
# PHYSICAL DEFINITION OF t_m (matching BlochCPU.jl):
#   KomaMRI's Bloch simulation accumulates off-resonance phase within each simulation
#   block, resetting block_time=0 at the start of each block. For SPGR, the ADC
#   samples fall within a single precession block per TR. Thus t_m is measured from
#   the START OF THE ADC BLOCK (not from the RF pulse), and this value is IDENTICAL
#   for all Ny PE lines. t_readout = repeat(t_within_ADC_block, Ny) is correct.
#
# TIME-SEGMENTATION NOTE:
#   Because t_readout is periodic (same Nx-length pattern tiled Ny times), segments
#   must NOT cross TR boundaries. We segment within each PE line independently
#   (seg_mode = :per_line, the default), OR use n_segments <= Nx to ensure all bins
#   fall within one TR. Segmenting by flat index across TR boundaries causes t_mid
#   to be computed as the average of a large end-of-readout time and a small
#   start-of-readout time → wrong demodulation → signal erasure.

using Base.Threads

export TissueSupportPlanOff, build_tissue_support_plan_off,
       create_ktes_support_off,
       create_support_finufft_cpu_off, get_readout_times

# ─────────────────────────────────────────────────────────────────────────────
# Extended plan struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    TissueSupportPlanOff{T<:AbstractFloat, TI<:Integer}

Precomputed trajectory, CSR tissue mapping, and off-resonance field distribution \$\\Delta\\omega(\\mathbf{x})\$
used for off-resonance-aware k-space tissue support calculation \$M_n^f(t)\$.

# Fields
- `base::TissueSupportPlan{T, TI}`: Underlying spatial and trajectory plan.
- `tissue_f::Vector{T}`: Off-resonance frequency in Hz for each grouped isochromat.
- `t_readout::Vector{T}`: Intra-ADC-block readout time in seconds per sample (length = \$M = N_x \\times N_y\$).
"""
struct TissueSupportPlanOff{T<:AbstractFloat, TI<:Integer}
    base      ::TissueSupportPlan{T, TI}
    tissue_f  ::Vector{T}   # off-resonance [Hz] per phantom point, length = total
    t_readout ::Vector{T}   # intra-ADC-block time [s] per sample, length = M (= Nx*Ny),
                            # periodic: repeat(t_within_block, Ny)
end

@inline _plan_ntissues(p::TissueSupportPlanOff) = p.base.ntissues
@inline _plan_M(p::TissueSupportPlanOff)        = p.base.M
@inline _plan_Nx(p::TissueSupportPlanOff)       = p.base.Nx
@inline _plan_Ny(p::TissueSupportPlanOff)       = p.base.Ny

# ─────────────────────────────────────────────────────────────────────────────
# Readout time extraction
# ─────────────────────────────────────────────────────────────────────────────

"""
    get_readout_times(seq::Sequence; any_rf::Bool=true) -> Vector{Float64}

Returns t_m [s]: the time elapsed from the **start of the ADC block** to each ADC
sample, measured within one TR. This matches the `block_time` variable in KomaMRI's
BlochCPU.jl `run_spin_precession!`, which resets to zero at the start of each
simulation block.

For SPGR, every PE line has an identical ADC block starting at the same intra-TR
offset, so the full output is `repeat(t_within_block, Ny)` (length = M = Nx*Ny).

### Why ADC-block start, not RF pulse:
KomaMRI splits the sequence into excitation blocks (RF on) and precession blocks
(RF off, gradient/ADC). The off-resonance phase in a precession block accumulates
as `Δf * block_time` where `block_time` starts at 0 for that block. For SPGR the
entire ADC window is within one precession block, so t_m = time since that block
started = time since ADC block start.
"""
function get_readout_times(seq::Sequence; any_rf::Bool=true)
    params   = seq.DEF["parameters"]
    Nx       = Int(params.Nx)
    Ny       = Int(params.Ny)

    if isa(params.seq_type, SPGR)
        # ── Absolute ADC times for the entire sequence (from KomaMRI) ─────────
        # get_adc_sampling_times returns absolute times from sequence start (t=0).
        t_all_adc = get_adc_sampling_times(seq)   # length = Nx * Ny

        length(t_all_adc) >= Nx || error(
            "get_readout_times: fewer ADC samples ($(length(t_all_adc))) than Nx=$Nx.")

        # ── ADC times for the FIRST TR only ───────────────────────────────────
        t_adc_first = t_all_adc[1:Nx]   # absolute, from sequence start

        # ── Find the start time of the ADC block in the first TR ─────────────
        # We need the absolute start time of the block that contains the first ADC.
        # get_block_start_times gives T0[i] = start time of block i.
        # The ADC block is the first block index i where is_ADC_on(seq[i]) is true.
        T0 = get_block_start_times(seq)
        adc_block_idx = findfirst(i -> is_ADC_on(seq[i]), 1:length(seq))
        adc_block_idx === nothing && error(
            "get_readout_times: no ADC block found in sequence.")
        t_adc_block_start = T0[adc_block_idx]   # absolute time of ADC block start

        # ── t_m = time from start of ADC block to each sample ─────────────────
        # This is exactly the block_time variable in BlochCPU run_spin_precession!
        t_within_block = t_adc_first .- t_adc_block_start   # length Nx, all ≥ 0

        all(t_within_block .>= -eps(Float64)) || @warn(
            "get_readout_times: negative t_within_block values detected. " *
            "Check that ADC samples fall within the ADC block.")
        t_within_block = max.(t_within_block, 0.0)

        # ── Tile for all Ny PE lines ───────────────────────────────────────────
        # Each TR has the same intra-block timing (Bloch block_time resets each TR).
        t_readout = repeat(t_within_block, Ny)   # length Nx*Ny
        return t_readout

    else
        # ── Generic fallback: per-TR ADC block start times ────────────────────
        T0        = get_block_start_times(seq)
        t_blocks  = Float64[]
        for i in 1:length(seq)
            is_ADC_on(seq[i]) || continue
            t_adc_i = times(seq.ADC[i]) .+ T0[i]   # absolute times of this ADC event
            push!(t_blocks, (t_adc_i .- T0[i])...)  # relative to THIS block start
        end
        isempty(t_blocks) && error("get_readout_times: no ADC samples found.")
        length(t_blocks) == Nx * Ny || @warn(
            "get_readout_times: total ADC count $(length(t_blocks)) ≠ Nx*Ny=$(Nx*Ny).")
        return t_blocks
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Plan construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_tissue_support_plan_off(phantom::Phantom, spin_groups::Vector{Vector{Int}}, seq::Sequence; kwargs...) -> TissueSupportPlanOff

Precompute an off-resonance–aware plan [`TissueSupportPlanOff`](@ref) containing trajectory points,
tissue pointers, and local off-resonance frequencies \$\\Delta\\omega\$ converted to Hz.

# Arguments
- `phantom::Phantom`: Digital phantom object.
- `spin_groups::Vector{Vector{Int}}`: Tissue group isochromat indices.
- `seq::Sequence`: Sequence containing RF and gradient waveforms.

# Keyword Arguments
- `use_float32::Bool = true`: Store arrays in Float32 precision.
- `spin_σ::Real = 1e-30`: Spatial subunit Gaussian scale factor \$\\sigma\$.
- `any_rf::Bool = true`: Treat all RF pulses as TR boundaries.
"""
function build_tissue_support_plan_off(
    phantom    ::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq        ::Sequence;
    use_float32::Bool = true,
    spin_σ     ::Real = 1e-30,
    any_rf     ::Bool = true,
)
    T = use_float32 ? Float32 : Float64

    base = build_tissue_support_plan(
        phantom, spin_groups, seq;
        use_float32=use_float32, spin_σ=spin_σ, any_rf=any_rf,
    )

    # Off-resonance: KomaMRI Δw is in rad/s → convert to Hz
    total    = length(base.tissue_x)
    tissue_f = Vector{T}(undef, total)
    pΔw      = phantom.Δw :: Vector{Float64}

    @inbounds for g in 1:base.ntissues
        s  = _group_start(base.tissue_ptr, g)
        ng = _group_len(base.tissue_ptr, g)
        for q in 1:ng
            p_idx           = s + q - 1
            vox_idx         = Int(base.tissue_voxel_idx[p_idx])
            tissue_f[p_idx] = T(pΔw[vox_idx] / (2π))
        end
    end

    t_raw   = get_readout_times(seq; any_rf=any_rf)
    M_check = length(t_raw)
    M_check == base.M || throw(ArgumentError(
        "Readout time length ($M_check) ≠ plan M ($(base.M)). " *
        "Check sequence ADC configuration."))
    t_readout = Vector{T}(t_raw)

    return TissueSupportPlanOff{T, eltype(base.tissue_ptr)}(base, tissue_f, t_readout)
end

# ─────────────────────────────────────────────────────────────────────────────
# Off-resonance workspace
# ─────────────────────────────────────────────────────────────────────────────

mutable struct CPUSupportWorkspaceOff{T<:AbstractFloat}
    x   ::Vector{T}
    y   ::Vector{T}
    z   ::Vector{T}
    w   ::Vector{Complex{T}}
    f   ::Vector{T}
    woff::Vector{Complex{T}}
    F   ::Vector{Complex{T}}
end

function CPUSupportWorkspaceOff(::Type{T}, max_group::Int, M::Int) where {T<:AbstractFloat}
    return CPUSupportWorkspaceOff{T}(
        Vector{T}(undef, max_group),
        Vector{T}(undef, max_group),
        Vector{T}(undef, max_group),
        Vector{Complex{T}}(undef, max_group),
        Vector{T}(undef, max_group),
        Vector{Complex{T}}(undef, max_group),
        Vector{Complex{T}}(undef, M),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Fill workspace
# ─────────────────────────────────────────────────────────────────────────────

function _fill_cpu_workspace_off!(
    ws     ::CPUSupportWorkspaceOff{T},
    plan   ::TissueSupportPlanOff{T, TI},
    weights,
    g      ::Int,
) where {T<:AbstractFloat, TI<:Integer}
    base = plan.base
    s    = _group_start(base.tissue_ptr, g)
    ng   = _group_len(base.tissue_ptr, g)

    if isnothing(weights)
        one_c = complex(one(T), zero(T))
        @inbounds @simd for q in 1:ng
            p       = s + q - 1
            ws.x[q] = base.tissue_x[p]
            ws.y[q] = base.tissue_y[p]
            ws.z[q] = base.tissue_z[p]
            ws.w[q] = one_c
            ws.f[q] = plan.tissue_f[p]
        end
    else
        @inbounds for q in 1:ng
            p       = s + q - 1
            ws.x[q] = base.tissue_x[p]
            ws.y[q] = base.tissue_y[p]
            ws.z[q] = base.tissue_z[p]
            j       = Int(base.tissue_voxel_idx[p])
            ws.w[q] = _effective_weight(weights, j, T)
            ws.f[q] = plan.tissue_f[p]
        end
    end
    return ng
end

# ─────────────────────────────────────────────────────────────────────────────
# Direct sum with macroscopic off-resonance
# ─────────────────────────────────────────────────────────────────────────────

function _dirac_sum_off_direct_cpu!(
    F         ::Vector{Complex{T}},
    x         ::Vector{T}, y::Vector{T}, z::Vector{T},
    w         ::Vector{Complex{T}},
    f         ::Vector{T},
    kx        ::Vector{T}, ky::Vector{T}, kz::Vector{T},
    env       ::Vector{T},
    t_readout ::Vector{T},
    ng        ::Int,
) where {T<:AbstractFloat}
    M     = length(kx)
    two_π = T(2π)

    @inbounds for m in 1:M
        kxm     = kx[m]
        kym     = ky[m]
        kzm     = kz[m]
        two_π_t = two_π * t_readout[m]

        acc_re = zero(T)
        acc_im = zero(T)

        for q in 1:ng
            # exp(-i2π(k·x + t*f)) = exp(-i*φ), φ = 2π(k·x + t*f)
            φ = muladd(kxm, x[q], muladd(kym, y[q], kzm * z[q])) +
                two_π_t * f[q]
            sφ, cφ = sincos(φ)
            wr = real(w[q])
            wi = imag(w[q])
            # (wr + i*wi) * (cφ - i*sφ) = (wr*cφ + wi*sφ) + i*(wi*cφ - wr*sφ)
            acc_re = muladd(wr, cφ, muladd(wi,  sφ, acc_re))
            acc_im = muladd(wi, cφ, muladd(-wr, sφ, acc_im))
        end

        em   = env[m]
        F[m] = complex(acc_re * em, acc_im * em)
    end
    return F
end

# ─────────────────────────────────────────────────────────────────────────────
# Time-segmented path with FINUFFT — segments within each PE line
# ─────────────────────────────────────────────────────────────────────────────
#
# KEY DESIGN: because t_readout = repeat(t_within_block, Ny), the time vector
# is PERIODIC with period Nx. We MUST segment within individual PE lines to
# avoid computing t_mid across a TR boundary (which would give a nonsensical
# average of a large end-of-readout time and a small start-of-readout time,
# destroying the demodulation). This is why n_segments had no beneficial effect
# at high values — it was crossing TR boundaries with every segment.
#
# Strategy: for each PE line (ny = 0..Ny-1), subdivide the Nx samples into
# ceil(n_seg_per_line) segments, compute t_mid within that line, and
# accumulate the NUFFT result.

function _dirac_sum_off_segmented_cpu!(
    F         ::Vector{Complex{T}},
    x         ::Vector{T}, y::Vector{T}, z::Vector{T},
    w         ::Vector{Complex{T}},
    f         ::Vector{T},
    kx        ::Vector{T}, ky::Vector{T}, kz::Vector{T},
    env       ::Vector{T},
    t_readout ::Vector{T},
    ng        ::Int,
    Nx        ::Int,
    Ny        ::Int;
    n_segments::Int = 16,
    eps        ::T  = _default_eps(T),
) where {T<:AbstractFloat}
    M     = Nx * Ny
    two_π = T(2π)
    fill!(F, zero(Complex{T}))

    # n_seg_per_line: how many segments per PE line
    # Clamp to [1, Nx] so we never request more segments than readout samples
    n_seg_per_line = clamp(n_segments, 1, Nx)

    wseg = Vector{Complex{T}}(undef, ng)
    xv   = x[1:ng]; yv = y[1:ng]; zv = z[1:ng]

    # Segment edges within one PE line (indices 1..Nx)
    seg_edges = round.(Int, LinRange(0, Nx, n_seg_per_line + 1))

    @inbounds for ny in 0:(Ny - 1)
        line_offset = ny * Nx   # flat index offset for this PE line

        for ℓ in 1:n_seg_per_line
            nx_start = seg_edges[ℓ]     + 1   # 1-based within line
            nx_stop  = seg_edges[ℓ + 1]
            nx_start > nx_stop && continue

            # Global flat indices for this segment
            m_start = line_offset + nx_start
            m_stop  = line_offset + nx_stop

            # t_mid: midpoint of t_readout within this line segment.
            # Since all Ny lines share the same t_within_block, we can equivalently
            # use t_readout[m_start] directly (same pattern every TR).
            t_mid = T(0.5) * (t_readout[m_start] + t_readout[m_stop])

            # Demodulate source weights by off-resonance phase at t_mid
            @simd for q in 1:ng
                φ_off  = two_π * t_mid * f[q]
                sφ, cφ = sincos(φ_off)
                wr = real(w[q]); wi = imag(w[q])
                # Multiply by exp(-i * φ_off) = cφ - i*sφ
                wseg[q] = complex(
                    muladd(wr, cφ, wi * sφ),
                    muladd(wi, cφ, -wr * sφ),
                )
            end

            kx_seg = kx[m_start:m_stop]
            ky_seg = ky[m_start:m_stop]
            kz_seg = kz[m_start:m_stop]

            out = FINUFFT.nufft3d3(
                xv[1:ng], yv[1:ng], zv[1:ng], wseg[1:ng],
                -1, eps,
                kx_seg, ky_seg, kz_seg,
            )

            # Residual phase correction: exp(-i2π*(t_m - t_mid)*f) is handled by
            # FINUFFT via the modified weights wseg above (demodulated at t_mid).
            # We still need to apply env and accumulate.
            @inbounds for (i, m) in enumerate(m_start:m_stop)
                F[m] = out[i] * env[m]
            end
        end
    end
    return F
end

# ─────────────────────────────────────────────────────────────────────────────
# CPU entry point
# ─────────────────────────────────────────────────────────────────────────────

"""
    create_support_finufft_cpu_off(phantom::Phantom, spin_groups::Vector{Vector{Int}}, seq::Sequence; kwargs...) -> Matrix{Complex{T}}

Compute off-resonance–modulated k-space tissue support using time-segmented type-3 NUFFT on CPU.

# Arguments
- `phantom::Phantom`: Object geometry and \$\\Delta\\omega\$ frequency map.
- `spin_groups::Vector{Vector{Int}}`: Isochromat groupings.
- `seq::Sequence`: Excitation and gradient sequence.

# Keyword Arguments
- `plan::Union{Nothing, TissueSupportPlanOff} = nothing`: Precomputed off-resonance support plan.
- `use_float32::Bool = true`: Precision flag (Float32 vs Float64).
- `spin_σ::Real = 1e-30`: Spatial subunit scale parameter \$\\sigma\$.
- `weights = nothing`: Optional voxel weights.
- `threaded::Bool = true`: Multi-threaded CPU execution.
- `direct_threshold::Int = 64`: Direct summation threshold for small tissue classes.
- `n_segments::Int = 16`: Number of temporal time-segments per readout line.
"""
function create_support_finufft_cpu_off(
    phantom    ::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq        ::Sequence;
    plan              ::Union{Nothing, TissueSupportPlanOff} = nothing,
    use_float32       ::Bool = true,
    spin_σ            ::Real = 1e-30,
    weights                  = nothing,
    threaded          ::Bool = true,
    direct_threshold  ::Int  = 64,
    n_segments        ::Int  = 16,
)
    T = use_float32 ? Float32 : Float64
    return _create_support_finufft_cpu_off_typed(
        T, phantom, spin_groups, seq;
        plan=plan, spin_σ=spin_σ, weights=weights,
        threaded=threaded, direct_threshold=direct_threshold,
        n_segments=n_segments,
    )
end

function _create_support_finufft_cpu_off_typed(
    ::Type{T},
    phantom    ::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq        ::Sequence;
    plan       ::Union{Nothing, TissueSupportPlanOff},
    spin_σ     ::Real,
    weights,
    threaded   ::Bool,
    direct_threshold::Int,
    n_segments ::Int,
) where {T<:AbstractFloat}

    CT   = Complex{T}
    epsT = _default_eps(T)

    local_plan = if isnothing(plan)
        build_tissue_support_plan_off(phantom, spin_groups, seq;
            use_float32 = (T === Float32), spin_σ = spin_σ)
    else
        plan
    end

    base = local_plan.base
    eltype(base.kx) == T ||
        throw(ArgumentError("Plan precision mismatch: plan is $(eltype(base.kx)), expected $T"))

    Nx = base.Nx
    Ny = base.Ny

    max_group = 0
    @inbounds for g in 1:base.ntissues
        ng = _group_len(base.tissue_ptr, g)
        ng > max_group && (max_group = ng)
    end

    K_spaces   = Array{CT}(undef, base.ntissues, Nx, Ny)
    nw         = threaded ? nthreads() : 1
    workspaces = [CPUSupportWorkspaceOff(T, max_group, base.M) for _ in 1:nw]

    function _run(g::Int)
        ws = workspaces[threaded ? threadid() : 1]
        ng = _fill_cpu_workspace_off!(ws, local_plan, weights, g)

        if ng <= direct_threshold
            _dirac_sum_off_direct_cpu!(
                ws.F, ws.x, ws.y, ws.z, ws.w, ws.f,
                base.kx, base.ky, base.kz, base.env, local_plan.t_readout, ng,
            )
        else
            _dirac_sum_off_segmented_cpu!(
                ws.F, ws.x, ws.y, ws.z, ws.w, ws.f,
                base.kx, base.ky, base.kz, base.env, local_plan.t_readout, ng,
                Nx, Ny;
                n_segments=n_segments, eps=epsT,
            )
        end
        @views K_spaces[g, :, :] .= reshape(ws.F, Nx, Ny)
    end

    if threaded
        @threads for g in 1:base.ntissues
            _run(g)
        end
    else
        for g in 1:base.ntissues
            _run(g)
        end
    end

    return K_spaces
end

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

"""
    create_ktes_support_off(phantom::Phantom, spin_groups::Vector{Vector{Int}}, seq::Sequence; kwargs...) -> Array{Complex{T}, 3}

Compute the macroscopic off-resonance–modulated k-space tissue support tensor \$M_n^f(t)\$
incorporating spatial field inhomogeneities \$\\Delta\\omega(\\mathbf{x})\$.

# Mathematical Formulation
As derived in *the k-TES paper*, macroscopic off-resonance modifies the spatial tissue density support:
```math
M_n^f(t) = \\int m_n(\\mathbf{x}) e^{-i 2\\pi t f(\\mathbf{x})} e^{-i 2\\pi \\mathbf{k}(t) \\cdot \\mathbf{x}} d\\mathbf{x}
```
evaluated numerically via time-segmented NUFFT along the readout trajectory.

# Arguments
- `phantom::Phantom`: Digital phantom with inhomogeneous \$\\Delta\\omega\$.
- `spin_groups::Vector{Vector{Int}}`: Isochromat indices per tissue class.
- `seq::Sequence`: Excitation and encoding sequence.

# Keyword Arguments
- `plan::Union{Nothing, TissueSupportPlanOff} = nothing`: Precomputed off-resonance support plan.
- `use_float32::Bool = true`: Precision mode.
- `spin_σ::Real = 1e-30`: Spatial subunit scale factor.
- `weights = nothing`: Optional voxel weights.
- `threaded_cpu::Bool = true`: Multi-threaded CPU execution.
- `direct_threshold::Int = 64`: Threshold below which direct summation is used.
- `n_segments::Int = 16`: Number of temporal segments per readout line.

# Returns
- `Array{Complex{T}, 3}`: Off-resonance tissue support tensor with dimensions `(N_tissues, Nx, Ny)`.
"""
function create_ktes_support_off(
    phantom    ::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq        ::Sequence;
    plan              ::Union{Nothing, TissueSupportPlanOff} = nothing,
    use_float32       ::Bool = true,
    spin_σ            ::Real = 1e-30,
    weights                  = nothing,
    threaded_cpu      ::Bool = true,
    direct_threshold  ::Int  = 64,
    n_segments        ::Int  = 16,
)
    @info "Using CPU (off-resonance path) for k-TES support generation" maxlog=1

    if has_accelerator()
        @warn "GPU path is not yet implemented for off-resonance k-TES. Falling back to CPU."
    end

    T = use_float32 ? Float32 : Float64
    return _create_support_finufft_cpu_off_typed(
        T, phantom, spin_groups, seq;
        plan=plan, spin_σ=spin_σ, weights=weights,
        threaded=threaded_cpu, direct_threshold=direct_threshold,
        n_segments=n_segments,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Compatibility bridge
# ─────────────────────────────────────────────────────────────────────────────

function assess_A(
    A            ::Array{<:Complex,3},
    raw          ::Matrix,
    seq_params   ::SeqParams,
    OF           ::Int,
    seq          ::Sequence,
    full_phantom ::Phantom,
    spin_groups  ::Vector{Vector{Int}},
    plan         ::TissueSupportPlanOff{T, TI},
) where {T<:AbstractFloat, TI<:Integer}
    slice_len = div(length(A), size(A, 1))
    if slice_len != length(raw)
        Ny   = seq_params.Ny
        sA3  = size(A, 3)
        NxOF = seq_params.Nx * OF
        if sA3 > Ny && size(A, 2) == NxOF
            lo = div(sA3 - Ny, 2) + 1
            hi = lo + Ny - 1
            A  = A[:, :, lo:hi]
        else
            @warn "Off-resonance A canvas incompatible. Recomputing with create_ktes_support_off."
            A = ComplexF32.(create_ktes_support_off(full_phantom, spin_groups, seq; plan=plan))
        end
    end
    return A
end