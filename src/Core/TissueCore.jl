using Base.Threads

export tissue_detector, create_ktes_support, build_tissue_support_plan,
       create_support_accel_streaming, create_support_finufft_cpu,
       create_support_phantoms, phantom_descriptor, TissueSupportPlan

# ─────────────────────────────────────────────────────────────────────────────
# Tissue detection
# ─────────────────────────────────────────────────────────────────────────────

"""
    phantom_descriptor(phantom::Phantom) -> (unique_spins, counts, groups)

Analyze a digital `phantom` to identify unique tissue classes defined by identical
intrinsic physical parameters \$(T_1, T_2, T_2^*, \\rho, \\Delta\\omega)\$.

# Arguments
- `phantom::Phantom`: Digital phantom object.

# Returns
- `unique_spins::Matrix{Float64}`: Array of size `(N_tissues, 5)` containing \$(T_1, T_2, T_2^*, \\rho, \\Delta\\omega)\$ for each tissue.
- `counts::Vector{Int}`: Number of isochromats belonging to each tissue class.
- `groups::Vector{Vector{Int}}`: Isochromat indices for each tissue class.
"""
function phantom_descriptor(phantom::Phantom)
    idx_map = Dict{NTuple{5, Float64}, Int}()
    groups  = Vector{Vector{Int}}()
    counts  = Int[]

    pT1  = phantom.T1  :: Vector{Float64}
    pT2  = phantom.T2  :: Vector{Float64}
    pT2s = phantom.T2s :: Vector{Float64}
    pρ   = phantom.ρ   :: Vector{Float64}
    pΔw  = phantom.Δw  :: Vector{Float64}

    n = length(pT1)

    for i in 1:n
        key = (pT1[i], pT2[i], pT2s[i], pρ[i], pΔw[i])

        gi = get!(idx_map, key) do
            push!(groups, Int[])
            push!(counts, 0)
            length(groups)
        end

        counts[gi] += 1
        push!(groups[gi], i)
    end

    nunique = length(groups)
    unique_spins = Array{Float64}(undef, nunique, 5)

    for (key, gi) in idx_map
        unique_spins[gi, 1] = key[1]
        unique_spins[gi, 2] = key[2]
        unique_spins[gi, 3] = key[3]
        unique_spins[gi, 4] = key[4]
        unique_spins[gi, 5] = key[5]
    end

    return unique_spins, counts, groups
end

"""
    tissue_detector(phantom::Phantom) -> (unique_spins, counts, groups)

Alias for [`phantom_descriptor`](@ref) to partition `phantom` isochromats into discrete tissue classes.
"""
function tissue_detector(phantom::Phantom)
    # Returns (unique_matrix, counts, groups) for backward compat
    return phantom_descriptor(phantom)
end

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

@inline _default_eps(::Type{Float32}) = Float32(1f-5)
@inline _default_eps(::Type{Float64}) = Float64(1e-12)

@inline _group_start(ptr, g::Int) = Int(ptr[g])
@inline _group_stop(ptr,  g::Int) = Int(ptr[g + 1] - 1)
@inline _group_len(ptr,   g::Int) = Int(ptr[g + 1] - ptr[g])

@inline _to_complex(x::Real,    ::Type{T}) where {T<:AbstractFloat} = complex(T(x), zero(T))
@inline _to_complex(x::Complex, ::Type{T}) where {T<:AbstractFloat} = complex(T(real(x)), T(imag(x)))

# ── Weight mode classification ────────────────────────────────────────────────
# Encode as an Int so the branch inside the hot loop is a cheap integer test
# that the compiler can constant-fold when weights::Nothing.
#   0 = :none   (no weights)
#   1 = :voxel  (AbstractVector)
#   2 = :voxel_col (AbstractMatrix, Nx1)
const _WM_NONE      = 0
const _WM_VOXEL     = 1
const _WM_VOXEL_COL = 2

function _weight_mode(weights, nvox::Int)
    isnothing(weights) && return _WM_NONE
    if weights isa AbstractVector && length(weights) == nvox && !(eltype(weights) <: AbstractArray)
        return _WM_VOXEL
    elseif weights isa AbstractMatrix && size(weights, 1) == nvox && size(weights, 2) == 1
        return _WM_VOXEL_COL
    else
        throw(ArgumentError(
            "weights must be nothing, a length-nvox vector, or an nvox×1 matrix; " *
            "got $(typeof(weights)) incompatible with nvox=$nvox"))
    end
end

# Type-stable weight fetch — three fully specialised methods.
# The :none case is now a separate function that the compiler inlines
# to a constant, eliminating the branch entirely.
@inline function _effective_weight(::Nothing, ::Int, ::Type{T}) where {T<:AbstractFloat}
    return complex(one(T), zero(T))
end

@inline function _effective_weight(weights::AbstractVector, j::Int, ::Type{T}) where {T<:AbstractFloat}
    return _to_complex(weights[j], T)
end

@inline function _effective_weight(weights::AbstractMatrix, j::Int, ::Type{T}) where {T<:AbstractFloat}
    return _to_complex(weights[j, 1], T)
end

# ─────────────────────────────────────────────────────────────────────────────
# Plan struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    TissueSupportPlan{T<:AbstractFloat, TI<:Integer}

Precomputed spatial and k-space trajectory plan for accelerating tissue support evaluation.
Holds trajectory coordinates, compressed sparse row (CSR) indexing for phantom tissues,
and voxel envelopes.

# Fields
- `ntissues::Int`: Number of unique tissue classes.
- `nvox::Int`: Total number of spatial voxel points in the phantom.
- `M::Int`: Total number of k-space readout samples.
- `Nx::Int`, `Ny::Int`: In-plane acquisition matrix dimensions.
- `kx::Vector{T}`, `ky::Vector{T}`, `kz::Vector{T}`: k-space trajectory coordinates in \$\\text{m}^{-1}\$.
- `env::Vector{T}`: Gaussian subunit Fourier envelope weighting \$\\hat{g}_\\sigma(\\mathbf{k})\$.
- `tissue_ptr::Vector{TI}`: CSR column pointers for tissue isochromat groupings.
- `tissue_x::Vector{T}`, `tissue_y::Vector{T}`, `tissue_z::Vector{T}`: Spatial coordinates of each grouped isochromat in meters.
- `tissue_voxel_idx::Vector{TI}`: Mapping from grouped entry back to phantom voxel index.
"""
struct TissueSupportPlan{T<:AbstractFloat, TI<:Integer}
    ntissues::Int
    nvox::Int
    M::Int
    Nx::Int
    Ny::Int

    kx::Vector{T}
    ky::Vector{T}
    kz::Vector{T}
    env::Vector{T}

    tissue_ptr::Vector{TI}
    tissue_x::Vector{T}
    tissue_y::Vector{T}
    tissue_z::Vector{T}
    tissue_voxel_idx::Vector{TI}
end

# ─────────────────────────────────────────────────────────────────────────────
# k-space trajectory helpers
# ─────────────────────────────────────────────────────────────────────────────

function is_tr_start(seq::Sequence, i::Int; any_rf::Bool=false)
    rf = seq.RF[i]
    is_RF_on(rf) || return false
    any_rf        && return true
    return rf.use isa Excitation
end

function get_tr_ranges(seq::Sequence; any_rf::Bool=false)
    tr_starts = Int[]
    for i in 1:length(seq)
        is_tr_start(seq, i; any_rf=any_rf) && push!(tr_starts, i)
    end
    isempty(tr_starts) && error(
        "No TR starts found. Try get_tr_ranges(seq; any_rf=true) and inspect rf.use.")
    tr_ranges = UnitRange{Int}[]
    for j in 1:length(tr_starts)
        first_block = tr_starts[j]
        last_block  = j < length(tr_starts) ? tr_starts[j+1] - 1 : length(seq)
        push!(tr_ranges, first_block:last_block)
    end
    return tr_ranges
end

function get_kspace_reset(seq::Sequence; Δt=1e-5, any_rf::Bool=false)
    tr_ranges = get_tr_ranges(seq; any_rf=any_rf)
    k_blocks  = Matrix{Float64}[]
    for r in tr_ranges
        seq_tr = seq[r]
        is_ADC_on(seq_tr) || continue
        _, k_adc_tr = get_kspace(seq_tr; Δt=Δt)
        push!(k_blocks, k_adc_tr)
    end
    isempty(k_blocks) && return zeros(Float64, 0, 3), tr_ranges
    return vcat(k_blocks...), tr_ranges
end

# ─────────────────────────────────────────────────────────────────────────────
# Plan construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_tissue_support_plan(phantom::Phantom, spin_groups::Vector{Vector{Int}}, seq::Sequence; kwargs...) -> TissueSupportPlan

Precompute and cache spatial coordinates, trajectory samples, and CSR tissue groupings into a [`TissueSupportPlan`](@ref).

# Arguments
- `phantom::Phantom`: Object geometry and parameters.
- `spin_groups::Vector{Vector{Int}}`: Isochromat indices for each tissue class (from [`tissue_detector`](@ref)).
- `seq::Sequence`: Sequence containing RF and gradient timing.

# Keyword Arguments
- `use_float32::Bool = true`: Store arrays in single precision for memory efficiency and GPU transfer speed.
- `spin_σ::Real = 1e-30`: Gaussian subunit spatial scale parameter \$\\sigma\$.
- `any_rf::Bool = true`: Detect all RF pulses as TR boundaries.
"""
function build_tissue_support_plan(
    phantom::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq::Sequence;
    use_float32::Bool = true,
    spin_σ::Real      = 1e-30,
    any_rf::Bool      = true,
)
    T = use_float32 ? Float32 : Float64
    return _build_plan(T, phantom, spin_groups, seq, spin_σ, any_rf)
end

# Inner function parametric on T — all inference is local and concrete here.
function _build_plan(
    ::Type{T},
    phantom::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq::Sequence,
    spin_σ::Real,
    any_rf::Bool,
    ::Type{TI} = Int32,
) where {T<:AbstractFloat, TI<:Integer}

    ntissues = length(spin_groups)
    nvox     = length(phantom.x)

    # CSR pointer array
    tissue_ptr    = Vector{TI}(undef, ntissues + 1)
    tissue_ptr[1] = TI(1)
    total = 0
    @inbounds for g in 1:ntissues
        total += length(spin_groups[g])
        tissue_ptr[g+1] = TI(total + 1)
    end

    tissue_x         = Vector{T}(undef, total)
    tissue_y         = Vector{T}(undef, total)
    tissue_z         = Vector{T}(undef, total)
    tissue_voxel_idx = Vector{TI}(undef, total)

    # Annotate phantom fields concretely to kill Any dispatch
    px = phantom.x :: Vector{Float64}
    py = phantom.y :: Vector{Float64}
    pz = phantom.z :: Vector{Float64}

    @inbounds for g in 1:ntissues
        grp  = spin_groups[g]
        ng   = length(grp)
        base = Int(tissue_ptr[g]) - 1
        @simd for k in 1:ng
            idx                 = grp[k]
            q                   = base + k
            tissue_x[q]         = T(px[idx])
            tissue_y[q]         = T(py[idx])
            tissue_z[q]         = T(pz[idx])
            tissue_voxel_idx[q] = TI(idx)
        end
    end

    # k-space trajectory
    seq_kspace = if haskey(seq.DEF, "kk")
        get_kspace_reset(seq; any_rf=any_rf)[1] :: Matrix{Float64}
    else
        get_kspace(seq)[2] :: Matrix{Float64}
    end

    M, d = size(seq_kspace)
    d == 3 || throw(ArgumentError("Expected 3D k-space trajectory, got d=$d"))

    params = seq.DEF["parameters"]
    OF     = params.emulated_noise_sequence ? 1 : Int(params.OF)
    Nx     = Int(params.Nx) * OF
    Ny     = Int(params.Ny)
    Nx * Ny == M || throw(ArgumentError("M=$M ≠ Nx*Ny=$(Nx*Ny)"))

    fac = T(2π)
    σ   = T(spin_σ)
    σ > 0 || throw(ArgumentError("spin_σ must be > 0, got $σ"))
    σ2h = T(0.5) * σ * σ

    kx  = Vector{T}(undef, M)
    ky  = Vector{T}(undef, M)
    kz  = Vector{T}(undef, M)
    env = Vector{T}(undef, M)

    @inbounds @simd for m in 1:M
        kxm    = fac * T(seq_kspace[m, 1])
        kym    = fac * T(seq_kspace[m, 2])
        kzm    = fac * T(seq_kspace[m, 3])
        kx[m]  = kxm
        ky[m]  = kym
        kz[m]  = kzm
        k2     = muladd(kxm, kxm, muladd(kym, kym, kzm * kzm))
        env[m] = exp(-σ2h * k2)
    end

    return TissueSupportPlan{T,TI}(
        ntissues, nvox, M, Nx, Ny,
        kx, ky, kz, env,
        tissue_ptr, tissue_x, tissue_y, tissue_z, tissue_voxel_idx,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# CPU workspace
# ─────────────────────────────────────────────────────────────────────────────

mutable struct CPUSupportWorkspace{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    w::Vector{Complex{T}}
    F::Vector{Complex{T}}
    # FINUFFT requires concrete Array{T}, not SubArray.
    # Pre-allocate scratch here so _dirac_sum_finufft_cpu! never allocates.
    xf::Vector{T}
    yf::Vector{T}
    zf::Vector{T}
    wf::Vector{Complex{T}}
end

function CPUSupportWorkspace(::Type{T}, max_group::Int, M::Int) where {T<:AbstractFloat}
    return CPUSupportWorkspace{T}(
        Vector{T}(undef, max_group),
        Vector{T}(undef, max_group),
        Vector{T}(undef, max_group),
        Vector{Complex{T}}(undef, max_group),
        Vector{Complex{T}}(undef, M),
        # FINUFFT scratch — same size, separate allocation
        Vector{T}(undef, max_group),
        Vector{T}(undef, max_group),
        Vector{T}(undef, max_group),
        Vector{Complex{T}}(undef, max_group),
    )
end

# Two specialisations: weights=nothing (hot path) and weights present.
# This eliminates the Symbol-dispatch branch from the inner loop entirely.
function _fill_cpu_workspace!(
    ws::CPUSupportWorkspace{T},
    plan::TissueSupportPlan{T,TI},
    ::Nothing,                    # weights = nothing  → specialised
    g::Int,
) where {T<:AbstractFloat, TI<:Integer}
    s  = _group_start(plan.tissue_ptr, g)
    ng = _group_len(plan.tissue_ptr, g)
    one_c = complex(one(T), zero(T))

    @inbounds @simd for q in 1:ng
        p       = s + q - 1
        ws.x[q] = plan.tissue_x[p]
        ws.y[q] = plan.tissue_y[p]
        ws.z[q] = plan.tissue_z[p]
        ws.w[q] = one_c            # constant — no memory read from weights
    end
    return ng
end

function _fill_cpu_workspace!(
    ws::CPUSupportWorkspace{T},
    plan::TissueSupportPlan{T,TI},
    weights,                      # weights ≠ nothing
    g::Int,
) where {T<:AbstractFloat, TI<:Integer}
    s  = _group_start(plan.tissue_ptr, g)
    ng = _group_len(plan.tissue_ptr, g)

    @inbounds for q in 1:ng
        p       = s + q - 1
        ws.x[q] = plan.tissue_x[p]
        ws.y[q] = plan.tissue_y[p]
        ws.z[q] = plan.tissue_z[p]
        j       = Int(plan.tissue_voxel_idx[p])
        ws.w[q] = _effective_weight(weights, j, T)
    end
    return ng
end

# ─────────────────────────────────────────────────────────────────────────────
# Direct Dirac sum (CPU, ng ≤ direct_threshold)
# ─────────────────────────────────────────────────────────────────────────────

function _dirac_sum_direct_cpu!(
    F  ::Vector{Complex{T}},
    x  ::Vector{T}, y::Vector{T}, z::Vector{T},
    w  ::Vector{Complex{T}},
    kx ::Vector{T}, ky::Vector{T}, kz::Vector{T},
    env::Vector{T},
    ng ::Int,
) where {T<:AbstractFloat}
    M = length(kx)

    # Loop order: m-outer, q-inner.
    # kx/ky/kz[m] loaded once per m; x/y/z/w[q] accessed sequentially (fits L1).
    @inbounds for m in 1:M
        kxm = kx[m]
        kym = ky[m]
        kzm = kz[m]

        acc_re = zero(T)
        acc_im = zero(T)

        for q in 1:ng
            ϕ        = muladd(kxm, x[q], muladd(kym, y[q], kzm * z[q]))
            # sincos computes both in a single FPU instruction on x86/ARM
            sϕ, cϕ  = sincos(ϕ)
            wr       = real(w[q])
            wi       = imag(w[q])
            acc_re   = muladd(wr, cϕ, muladd(wi, sϕ, acc_re))  # muladd fuses mul+add
            acc_im   = muladd(wi, cϕ, muladd(-wr, sϕ, acc_im))
        end

        em   = env[m]
        F[m] = complex(acc_re * em, acc_im * em)   # type-stable Base inline
    end

    return F
end

# ─────────────────────────────────────────────────────────────────────────────
# FINUFFT path (CPU, ng > direct_threshold)
# ─────────────────────────────────────────────────────────────────────────────

function _dirac_sum_finufft_cpu!(
    F  ::Vector{Complex{T}},
    x  ::Vector{T}, y::Vector{T}, z::Vector{T},
    w  ::Vector{Complex{T}},
    xf ::Vector{T}, yf::Vector{T}, zf::Vector{T},   # ← FINUFFT scratch
    wf ::Vector{Complex{T}},
    kx ::Vector{T}, ky::Vector{T}, kz::Vector{T},
    env::Vector{T},
    ng ::Int;
    eps::T = _default_eps(T),
) where {T<:AbstractFloat}
    # Copy ng elements into concrete scratch — FINUFFT needs Array{T}, not SubArray
    @inbounds copyto!(xf, 1, x, 1, ng)
    @inbounds copyto!(yf, 1, y, 1, ng)
    @inbounds copyto!(zf, 1, z, 1, ng)
    @inbounds copyto!(wf, 1, w, 1, ng)

    # Slice to exact length — these are Array{T}, not views, so FINUFFT accepts them
    out = FINUFFT.nufft3d3(
        xf[1:ng], yf[1:ng], zf[1:ng], wf[1:ng],
        -1, eps,
        kx, ky, kz,
    )

    @inbounds @simd for m in eachindex(out)
        F[m] = out[m] * env[m]
    end
    return F
end
# ─────────────────────────────────────────────────────────────────────────────
# CPU FINUFFT path — public entry point
# ─────────────────────────────────────────────────────────────────────────────

"""
    create_support_finufft_cpu(phantom::Phantom, spin_groups::Vector{Vector{Int}}, seq::Sequence; kwargs...) -> Matrix{Complex{T}}

Compute the k-space tissue support matrix on the CPU using type-3 Non-Uniform Fast Fourier Transforms (FINUFFT)
for large tissue groups, falling back to direct summations for small groups below `direct_threshold`.

# Arguments
- `phantom::Phantom`: Digital phantom defining spin coordinates.
- `spin_groups::Vector{Vector{Int}}`: Isochromat indices for each tissue class.
- `seq::Sequence`: Excitation and encoding sequence.

# Keyword Arguments
- `plan::Union{Nothing,TissueSupportPlan} = nothing`: Optional precomputed support plan.
- `use_float32::Bool = true`: Compute in single precision (Float32).
- `spin_σ::Real = 1e-30`: Spatial subunit scale factor \$\\sigma\$.
- `weights = nothing`: Optional voxel weighting vector or matrix.
- `threaded::Bool = true`: Parallelize across available CPU threads.
- `direct_threshold::Int = 64`: Isochromat count below which direct Dirac summation is used instead of NUFFT.
"""
function create_support_finufft_cpu(
    phantom    ::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq        ::Sequence;
    plan              ::Union{Nothing,TissueSupportPlan} = nothing,
    use_float32       ::Bool    = true,
    spin_σ            ::Real    = 1e-30,
    weights                     = nothing,
    threaded          ::Bool    = true,
    direct_threshold  ::Int     = 64,
)
    T    = use_float32 ? Float32 : Float64
    return _create_support_finufft_cpu_typed(
        T, phantom, spin_groups, seq;
        plan=plan, spin_σ=spin_σ, weights=weights,
        threaded=threaded, direct_threshold=direct_threshold,
    )
end

# Inner function with T as a type parameter → fully concrete inference
function _create_support_finufft_cpu_typed(
    ::Type{T},
    phantom    ::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq        ::Sequence;
    plan              ::Union{Nothing,TissueSupportPlan},
    spin_σ            ::Real,
    weights,
    threaded          ::Bool,
    direct_threshold  ::Int,
) where {T<:AbstractFloat}

    CT   = Complex{T}
    epsT = _default_eps(T)

    local_plan = if isnothing(plan)
        build_tissue_support_plan(phantom, spin_groups, seq;
            use_float32 = (T === Float32), spin_σ = spin_σ)
    else
        plan
    end
    eltype(local_plan.kx) == T ||
        throw(ArgumentError("Plan precision mismatch: plan is $(eltype(local_plan.kx)), expected $T"))

    max_group = 0
    @inbounds for g in 1:local_plan.ntissues
        ng = _group_len(local_plan.tissue_ptr, g)
        ng > max_group && (max_group = ng)
    end

    K_spaces   = Array{CT}(undef, local_plan.ntissues, local_plan.Nx, local_plan.Ny)
    nw         = threaded ? nthreads() : 1
    workspaces = [CPUSupportWorkspace(T, max_group, local_plan.M) for _ in 1:nw]

    function _run(g::Int)
        ws = workspaces[threaded ? threadid() : 1]
        ng = _fill_cpu_workspace!(ws, local_plan, weights, g)

        if ng <= direct_threshold
            _dirac_sum_direct_cpu!(
                ws.F, ws.x, ws.y, ws.z, ws.w,
                local_plan.kx, local_plan.ky, local_plan.kz, local_plan.env, ng,
            )
        else
            _dirac_sum_finufft_cpu!(
                ws.F, ws.x, ws.y, ws.z, ws.w,
                ws.xf, ws.yf, ws.zf, ws.wf,          # ← pass scratch buffers
                local_plan.kx, local_plan.ky, local_plan.kz, local_plan.env, ng;
                eps = epsT,
            )
        end
        @views K_spaces[g, :, :] .= reshape(ws.F, local_plan.Nx, local_plan.Ny)
    end

    if threaded
        @threads for g in 1:local_plan.ntissues
            _run(g)
        end
    else
        for g in 1:local_plan.ntissues
            _run(g)
        end
    end

    return K_spaces
end

# ─────────────────────────────────────────────────────────────────────────────
# GPU kernel
# ─────────────────────────────────────────────────────────────────────────────

@kernel function accel_chunk_kernel!(
    partial_re, partial_im,
    xchunk, ychunk, zchunk, wchunk,
    kx, ky, kz,
    chunk_len::Int32,
)
    m = @index(Global, Linear)

    if m <= length(kx)
        kxm    = kx[m]
        kym    = ky[m]
        kzm    = kz[m]
        acc_re = zero(eltype(kx))
        acc_im = zero(eltype(kx))

        @inbounds for q in 1:chunk_len
            ϕ      = muladd(kxm, xchunk[q], muladd(kym, ychunk[q], kzm * zchunk[q]))
            sϕ, cϕ = sincos(ϕ)         # single instruction on PTX as well
            wq     = wchunk[q]
            wr     = real(wq)
            wi     = imag(wq)
            acc_re = muladd(wr, cϕ, muladd(wi, sϕ, acc_re))
            acc_im = muladd(wi, cϕ, muladd(-wr, sϕ, acc_im))
        end

        partial_re[m] = acc_re
        partial_im[m] = acc_im
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# GPU streaming path
# ─────────────────────────────────────────────────────────────────────────────

"""
    create_support_accel_streaming(phantom::Phantom, spin_groups::Vector{Vector{Int}}, seq::Sequence; kwargs...) -> Matrix{Complex{T}}

Compute the k-space tissue support matrix using hardware acceleration (CUDA, Metal, AMDGPU, oneAPI)
via chunked streaming kernels to fit within limited GPU VRAM.

# Arguments
- `phantom::Phantom`: Digital phantom object.
- `spin_groups::Vector{Vector{Int}}`: Isochromat indices for each tissue.
- `seq::Sequence`: Excitation and spatial encoding sequence.

# Keyword Arguments
- `plan::Union{Nothing,TissueSupportPlan} = nothing`: Optional precomputed support plan.
- `use_float32::Bool = true`: Precision flag (single precision is strongly recommended on GPU).
- `spin_σ::Real = 1e-30`: Spatial subunit scale factor \$\\sigma\$.
- `weights = nothing`: Optional voxel weighting vector or matrix.
- `voxel_chunk_size::Int = 131_072`: Maximum number of voxels processed per GPU streaming buffer.
- `workgroup_size::Int = 256`: KernelAbstractions workgroup/thread block size.
"""
function create_support_accel_streaming(
    phantom    ::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq        ::Sequence;
    plan              ::Union{Nothing,TissueSupportPlan} = nothing,
    use_float32       ::Bool = true,
    spin_σ            ::Real = 1e-30,
    weights                  = nothing,
    voxel_chunk_size  ::Int  = 131_072,
    workgroup_size    ::Int  = 256,
)
    T  = use_float32 ? Float32 : Float64
    return _create_support_accel_streaming_typed(
        T, phantom, spin_groups, seq;
        plan=plan, spin_σ=spin_σ, weights=weights,
        voxel_chunk_size=voxel_chunk_size, workgroup_size=workgroup_size,
    )
end

function _create_support_accel_streaming_typed(
    ::Type{T},
    phantom    ::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq        ::Sequence;
    plan              ::Union{Nothing,TissueSupportPlan},
    spin_σ            ::Real,
    weights,
    voxel_chunk_size  ::Int,
    workgroup_size    ::Int,
) where {T<:AbstractFloat}

    CT = Complex{T}

    local_plan = if isnothing(plan)
        build_tissue_support_plan(phantom, spin_groups, seq;
            use_float32 = (T === Float32), spin_σ = spin_σ)
    else
        plan
    end
    eltype(local_plan.kx) == T ||
        throw(ArgumentError("Plan precision mismatch: plan is $(eltype(local_plan.kx)), expected $T"))

    backend = get_accelerator_backend()
    M       = local_plan.M

    # Static device arrays — uploaded once, never reallocated
    dkx = _ka_array(backend, local_plan.kx)
    dky = _ka_array(backend, local_plan.ky)
    dkz = _ka_array(backend, local_plan.kz)

    kernel! = accel_chunk_kernel!(backend, workgroup_size)

    # ── Pre-allocate ALL buffers before any loop ───────────────────────────────
    host_x = Vector{T}(undef, voxel_chunk_size)
    host_y = Vector{T}(undef, voxel_chunk_size)
    host_z = Vector{T}(undef, voxel_chunk_size)
    host_w = Vector{CT}(undef, voxel_chunk_size)

    dx = _ka_array(backend, host_x)          # device pinned to voxel_chunk_size
    dy = _ka_array(backend, host_y)
    dz = _ka_array(backend, host_z)
    dw = _ka_array(backend, host_w)

    dpartial_re = _ka_zeros(T, backend, M)   # reused every chunk via fill!
    dpartial_im = _ka_zeros(T, backend, M)

    acc_re     = Vector{T}(undef, M)         # host accumulator (no alloc in loop)
    acc_im     = Vector{T}(undef, M)
    chunk_re   = Vector{T}(undef, M)         # download scratch (replaces _to_host)
    chunk_im   = Vector{T}(undef, M)
    tissue_vec = Vector{CT}(undef, M)

    K_spaces = Array{CT}(undef, local_plan.ntissues, local_plan.Nx, local_plan.Ny)

    for g in 1:local_plan.ntissues
        fill!(acc_re, zero(T))
        fill!(acc_im, zero(T))

        s  = _group_start(local_plan.tissue_ptr, g)
        ng = _group_len(local_plan.tissue_ptr,  g)

        offset = 0
        while offset < ng
            clen = min(voxel_chunk_size, ng - offset)

            # ── Stage host buffers (two specialised paths, no Symbol branch) ──
            if isnothing(weights)
                one_c = complex(one(T), zero(T))
                @inbounds @simd for q in 1:clen
                    p         = s + offset + q - 1
                    host_x[q] = local_plan.tissue_x[p]
                    host_y[q] = local_plan.tissue_y[p]
                    host_z[q] = local_plan.tissue_z[p]
                    host_w[q] = one_c
                end
            else
                @inbounds for q in 1:clen
                    p         = s + offset + q - 1
                    host_x[q] = local_plan.tissue_x[p]
                    host_y[q] = local_plan.tissue_y[p]
                    host_z[q] = local_plan.tissue_z[p]
                    j         = Int(local_plan.tissue_voxel_idx[p])
                    host_w[q] = _effective_weight(weights, j, T)
                end
            end

            # ── Upload slice — no allocation, just H2D copy ────────────────
            copyto!(dx, 1, host_x, 1, clen)
            copyto!(dy, 1, host_y, 1, clen)
            copyto!(dz, 1, host_z, 1, clen)
            copyto!(dw, 1, host_w, 1, clen)

            fill!(dpartial_re, zero(T))
            fill!(dpartial_im, zero(T))

            kernel!(
                dpartial_re, dpartial_im,
                dx, dy, dz, dw,
                dkx, dky, dkz,
                Int32(clen),
                ndrange = M,
            )
            KernelAbstractions.synchronize(backend)

            # ── Download into pre-allocated scratch, accumulate ────────────
            copyto!(chunk_re, dpartial_re)
            copyto!(chunk_im, dpartial_im)
            @inbounds @simd for m in 1:M
                acc_re[m] += chunk_re[m]
                acc_im[m] += chunk_im[m]
            end

            offset += clen
        end

        # ── Apply envelope — complex() instead of CT() constructor ─────────
        @inbounds @simd for m in 1:M
            em          = local_plan.env[m]
            tissue_vec[m] = complex(acc_re[m] * em, acc_im[m] * em)
        end
        @views K_spaces[g, :, :] .= reshape(tissue_vec, local_plan.Nx, local_plan.Ny)
    end

    return K_spaces
end

# ─────────────────────────────────────────────────────────────────────────────
# Backend description
# ─────────────────────────────────────────────────────────────────────────────

function describe_backend()
    a = active_accelerator()
    a === :cpu    && return "CPU"
    a === :cuda   && return "CUDA GPU"
    a === :amd    && return "AMDGPU/ROCm GPU"
    a === :metal  && return "Metal GPU"
    a === :oneapi && return "oneAPI GPU"
    return string(a)
end

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

"""
    create_ktes_support(phantom::Phantom, seq::Sequence, [sys::Scanner]; kwargs...) -> Array{Complex{T}, 3}
    create_ktes_support(phantom::Phantom, spin_groups::Vector{Vector{Int}}, seq::Sequence; kwargs...) -> Array{Complex{T}, 3}

Compute the k-space spatial density support tensor \$\\hat{M}_n(\\mathbf{k}(t))\$ for each tissue class
in `phantom` along the sequence trajectory `k(t)`.

# Mathematical Formulation
As introduced in *the k-TES paper*, given a phantom partitioned into \$N\$ tissue classes where each tissue
is represented by a spatial Dirac comb \$p_n(\\mathbf{x})\$, its discrete spatial density support is:
```math
\\hat{M}_n(t) = \\mathcal{F}\\{p_n(\\mathbf{x})\\}(\\mathbf{k}(t))
```
computed using type-3 NUFFT on CPU or accelerated streaming kernels on GPU.

# Arguments
- `phantom::Phantom`: Digital phantom object.
- `spin_groups::Vector{Vector{Int}}`: Isochromat indices for each tissue class (from [`tissue_detector`](@ref)).
- `seq::Sequence`: Sequence containing RF and spatial encoding gradients.
- `sys::Scanner`: Optional scanner hardware specification.

# Keyword Arguments
- `plan::Union{Nothing,TissueSupportPlan} = nothing`: Precomputed [`TissueSupportPlan`](@ref) for reusing trajectory calculations.
- `use_float32::Bool = true`: Compute in single precision (recommended for GPU memory and speed).
- `spin_σ::Real = 1e-30`: Spatial subunit Gaussian scale factor \$\\sigma\$.
- `weights = nothing`: Optional per-voxel weighting factor (e.g. coil sensitivities or partial volume).
- `voxel_chunk_size::Int = 131_072`: Voxel chunk size for GPU streaming.
- `workgroup_size::Int = 256`: GPU thread block dimension.
- `threaded_cpu::Bool = true`: Parallelize over threads on CPU.
- `direct_threshold::Int = 64`: Isochromat threshold below which direct Dirac summation is used on CPU.

# Returns
- `Array{Complex{T}, 3}`: Support tensor of dimensions `(N_tissues, Nx, Ny)`.
"""
function create_ktes_support(
    phantom    ::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq        ::Sequence;
    plan              ::Union{Nothing,TissueSupportPlan} = nothing,
    use_float32       ::Bool = true,
    spin_σ            ::Real = 1e-30,
    weights                  = nothing,
    voxel_chunk_size  ::Int  = 131_072,
    workgroup_size    ::Int  = 256,
    threaded_cpu      ::Bool = true,
    direct_threshold  ::Int  = 64,
)
    @info "Using $(describe_backend()) for k-TES support generation" maxlog=1

    # Dispatch to a typed inner function so T is concrete for all downstream calls.
    T = use_float32 ? Float32 : Float64
    return _create_ktes_support_typed(
        T, phantom, spin_groups, seq;
        plan=plan, spin_σ=spin_σ, weights=weights,
        voxel_chunk_size=voxel_chunk_size, workgroup_size=workgroup_size,
        threaded_cpu=threaded_cpu, direct_threshold=direct_threshold,
    )
end

function create_ktes_support(
    phantom::Phantom,
    seq    ::Sequence;
    kwargs...,
)
    spin_groups = tissue_detector(phantom)[3]
    return create_ktes_support(phantom, spin_groups, seq; kwargs...)
end

function create_ktes_support(
    phantom::Phantom,
    seq    ::Sequence,
    sys    ::Scanner;
    kwargs...,
)
    return create_ktes_support(phantom, seq; kwargs...)
end

function _create_ktes_support_typed(
    ::Type{T},
    phantom    ::Phantom,
    spin_groups::Vector{Vector{Int}},
    seq        ::Sequence;
    plan              ::Union{Nothing,TissueSupportPlan},
    spin_σ            ::Real,
    weights,
    voxel_chunk_size  ::Int,
    workgroup_size    ::Int,
    threaded_cpu      ::Bool,
    direct_threshold  ::Int,
) where {T<:AbstractFloat}

    if has_accelerator()
        try
            return _create_support_accel_streaming_typed(
                T, phantom, spin_groups, seq;
                plan=plan, spin_σ=spin_σ, weights=weights,
                voxel_chunk_size=voxel_chunk_size, workgroup_size=workgroup_size,
            )
        catch e
            # Only fall back on hardware-level failures, not logic bugs.
            e isa OutOfMemoryError || e isa KernelAbstractions.BackendException ||
                rethrow()
            @warn "GPU path failed ($e) — falling back to CPU FINUFFT path"
        end
    end

    return _create_support_finufft_cpu_typed(
        T, phantom, spin_groups, seq;
        plan=plan, spin_σ=spin_σ, weights=weights,
        threaded=threaded_cpu, direct_threshold=direct_threshold,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Support phantom construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    create_support_phantoms(phantom::Phantom, seq::Sequence, sys::Scanner; kwargs...) -> Vector{Phantom}
    create_support_phantoms(phantom::Phantom, seq::Sequence, d::Distribution, sys::Scanner; kwargs...) -> Vector{Phantom}

Generate a vector of single-tissue representative support phantoms, one per tissue class detected in `phantom`.
Each support phantom contains `replicas` Monte Carlo isochromats sampled according to the truncated
Gaussian subunit distribution \$g_\\sigma(\\xi)\$ within a localized discrete tissue support \$V_d\$.

# Mathematical Formulation
As established in *the k-TES paper*, the discrete tissue basis signal is approximated by Monte Carlo sampling:
```math
\\hat{S}_{0,n}^J(t) = \\frac{1}{J} \\sum_{j=1}^J s_{\\xi^{(j)}}(t), \\quad \\xi^{(j)} \\sim g_\\sigma(\\xi)
```
where \$J\$ (`replicas`) is typically \$\\ge 1000\$, and \$\\sigma = 1/20\$ represents ~6.4% of the voxel volume.

# Arguments
- `phantom::Phantom`: Macro-phantom containing all tissue classes.
- `seq::Sequence`: Excitation and encoding sequence.
- `sys::Scanner`: Hardware limits and ADC sampling parameters.
- `d::Distribution`: Isochromat spatial distribution (default: `Normal()`).

# Keyword Arguments
- `σ::AbstractFloat = 1/20.0`: Spatial subunit extent scale factor \$\\sigma\$.
- `replicas::Int = 10000`: Number of Monte Carlo isochromats \$J\$ per tissue basis phantom.
- `T2s_params::T2sParams = T2sParams()`: Configuration for intravoxel microscopic dephasing.
- `seeds = [2025, 2024, 2023]`: Deterministic RNG seeds for reproducible sampling.

# Returns
- `Vector{Phantom}`: Array of \$N\$ single-tissue phantoms, ready for Bloch simulation.
"""
function create_support_phantoms(
    phantom::Phantom, seq::Sequence, sys::Scanner;
    d=Normal(), σ::AbstractFloat=1/20.0, replicas::Int=10000,
    T2s_params::T2sParams=T2sParams(), seeds=[2025, 2024, 2023],
)
    return create_support_phantoms(phantom, seq, d, sys; σ=σ, replicas=replicas, T2s_params=T2s_params, seeds=seeds)
end

function create_support_phantoms(
    phantom::Phantom, seq::Sequence, d::Distributions.Distribution, sys::Scanner;
    σ::AbstractFloat=1/20.0, replicas::Int=10000,
    T2s_params::T2sParams=T2sParams(), seeds=[2025, 2024, 2023],
)
    params = seq.DEF["parameters"]
    OF = if params.emulated_noise_sequence
        1
    elseif check_OF(params, sys)
        params.OF
    else
        @warn "Anomaly in t_ADC/Nx — using OF as declared in sequence."
        params.OF
    end
    return create_support_phantoms(
        phantom,
        params.FOVx, params.FOVy, params.slice_thickness,
        params.Nx, params.Ny, d;
        OF=OF, σ=σ, replicas=replicas, T2s_params=T2s_params, seeds=seeds,
    )
end

function create_support_phantoms(
    phantom::Phantom,
    FOVx::T, FOVy::T, slice_thickness::T,
    Nx::Int, Ny::Int,
    d::Normal;
    OF::Int=1, σ::T=T(1/20), replicas::Int=10000,
    T2s_params::T2sParams=T2sParams(), seeds=[2025, 2024, 2023],
) where {T<:AbstractFloat}

    _, _, spin_groups = tissue_detector(phantom)
    nspins   = length(spin_groups)
    phantoms = Vector{Phantom}(undef, nspins)

    σ_x = σ * FOVx / (Nx * OF)
    σ_y = σ * FOVy / Ny
    σ_z = σ * slice_thickness

    x_rand = Xoshiro(seeds[1])
    y_rand = Xoshiro(seeds[2])
    z_rand = Xoshiro(seeds[3])

    # Pre-allocate single buffer per field; reuse across tissues
    x_buf = Vector{Float64}(undef, replicas)
    y_buf = Vector{Float64}(undef, replicas)
    z_buf = Vector{Float64}(undef, replicas)

    nd = Normal()   # unit normal, scaling applied below
    
    a = 4

    ndx = truncated(Normal(0, σ_x), -a * σ_x, a * σ_x)
    ndy = truncated(Normal(0, σ_y), -a * σ_y, a * σ_y)
    ndz = truncated(Normal(0, σ_z), -a * σ_z, a * σ_z)

    for i in 1:nspins
        # Constant fields: use fill! into a pre-allocated buffer — no repeat()
        idx = spin_groups[i][1]
        T1_val  = phantom.T1[idx]
        T2_val  = phantom.T2[idx]
        T2s_val = phantom.T2s[idx]
        ρ_val   = phantom.ρ[idx]
        Δw_val  = phantom.Δw[idx]

        T1  = fill(T1_val,  replicas)
        T2  = fill(T2_val,  replicas)
        T2s = fill(T2s_val, replicas)
        ρ   = fill(ρ_val,   replicas)
        Δw  = fill(Δw_val,  replicas)

        if replicas == 1
            x = [0.0]; y = [0.0]; z = [0.0]
        else
            rand!(x_rand, ndx, x_buf); #@. x_buf = σ_x * x_buf
            rand!(y_rand, ndy, y_buf); #@. y_buf = σ_y * y_buf
            rand!(z_rand, ndz, z_buf); #@. z_buf = σ_z * z_buf
            x = copy(x_buf); y = copy(y_buf); z = copy(z_buf)
        end

        phantoms[i] = Phantom(x=x, y=y, z=z, T1=T1, T2=T2, T2s=T2s, ρ=ρ, Δw=Δw)
    end

    return T2s_Δw(phantoms, T2s_params.method; T2s_params=T2s_params)
end

function create_support_phantoms(
    phantom::Phantom,
    FOVx::T, FOVy::T, slice_thickness::T,
    Nx::Int, Ny::Int,
    d::Uniform;
    OF::Int=1, σ::T=T(1/20), replicas::Int=10000,
    T2s_params::T2sParams=T2sParams(), seeds=[2025, 2024, 2023],
) where {T<:AbstractFloat}

    _, _, spin_groups = tissue_detector(phantom)
    nspins   = length(spin_groups)
    phantoms = Vector{Phantom}(undef, nspins)

    L_x = σ * FOVx / (Nx * OF)
    L_y = σ * FOVy / Ny
    L_z = σ * slice_thickness

    x_rand = Xoshiro(seeds[1])
    y_rand = Xoshiro(seeds[2])
    z_rand = Xoshiro(seeds[3])

    x_buf = Vector{Float64}(undef, replicas)
    y_buf = Vector{Float64}(undef, replicas)
    z_buf = Vector{Float64}(undef, replicas)

    ud = Uniform(-1.0, 1.0)
    # Δw for Uniform is a deterministic linspace — compute once and copy
    Δw_base = collect(2π .* (0:replicas-1) ./ replicas)

    for i in 1:nspins
        idx = spin_groups[i][1]
        T1_val  = phantom.T1[idx]
        T2_val  = phantom.T2[idx]
        T2s_val = phantom.T2s[idx]
        ρ_val   = phantom.ρ[idx]

        T1  = fill(T1_val,  replicas)
        T2  = fill(T2_val,  replicas)
        T2s = fill(T2s_val, replicas)
        ρ   = fill(ρ_val,   replicas)
        Δw  = copy(Δw_base)

        rand!(x_rand, ud, x_buf); @. x_buf = L_x * x_buf
        rand!(y_rand, ud, y_buf); @. y_buf = L_y * y_buf
        rand!(z_rand, ud, z_buf); @. z_buf = L_z * z_buf
        x = copy(x_buf); y = copy(y_buf); z = copy(z_buf)

        phantoms[i] = Phantom(x=x, y=y, z=z, T1=T1, T2=T2, T2s=T2s, ρ=ρ, Δw=Δw)
    end

    return T2s_Δw(phantoms, T2s_params.method; T2s_params=T2s_params)
end

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

function compare_tissues(spin::Vector{Float64}, spins::Matrix{Float64})
    nspins = size(spins, 1)
    v = zeros(nspins)
    @inbounds for i in 1:nspins
        spin == view(spins, i, :) && (v[i] = 1.0)
    end
    return v
end