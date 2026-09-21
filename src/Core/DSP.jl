export create_kernel, kspace_sum_gaussians_nufft, k_resize, non_cartesian_ifft, _dirac_sum_nufft2

"""
    create_kernel(kernel::UniformKernel, OF::Integer) -> Vector{Float64}

Generate a discrete uniform convolution kernel of length `OF` normalized to unit sum (`1/OF`).
"""
function create_kernel(kernel::UniformKernel, OF)
    D = 1/OF.*ones(Int(OF))
    return D
end

"""
    create_kernel(kernel::TriangularKernel, OF::Integer) -> Vector{Float64}

Generate a discrete triangular (Bartlett) convolution kernel of length `OF` normalized to unit sum.
"""
function create_kernel(kernel::TriangularKernel, OF)
    Δ = (2 / (OF + 1))
    c = zeros(Int(OF))
    for i=1:OF
        c[Int(i)] = -1 + i * Δ
    end
    c = collect(c)

    ∧(x) = max.(0, 1 .-abs.(x))
    D = ∧(c)
    D = D / sum(D)
    return D
end

"""
    create_kernel(kernel::SincKernel, OF::Integer) -> Vector{Float64}

Generate a discrete sinc convolution kernel of length `OF` normalized to unit sum for bandlimited interpolation.
"""
function create_kernel(kernel::SincKernel, OF)
    if iseven(OF)
        OF = OF + 1
    end
    c = collect(-(OF-2):2:OF-2)
    append!(c, 0)
    sort!(c)
    c = 1/2.0*c
    if iseven(OF)
        deleteat!(c, Int((OF+1)/2))
    end
    D = sinc.(c)
    D = D/sum(D)
    return D
end


"""
    kspace_sum_gaussians_nufft(points::AbstractMatrix, kpts::AbstractMatrix; sigma, kwargs...) -> Vector{ComplexF64}

Compute weighted k-space sum of Gaussian splats using NUFFT.

This function evaluates the discrete Fourier transform of a weighted sum of Gaussian
functions centered at `points` in k-space.

# Arguments
- `points::AbstractMatrix`: N×d matrix of Gaussian centers in d-dimensional space
- `kpts::AbstractMatrix`: M×d matrix of k-space sample locations
- `weights::Nothing` or `AbstractVector`: Optional complex weights for each Gaussian (default: all ones)
- `sigma`: Gaussian width parameter, supporting multiple types:
  - `Real`: Scalar σ for all Gaussians (isotropic)
  - `AbstractMatrix`: N×d matrix of per-Gaussian anisotropic widths
  - `AbstractVector`: N-vector of per-Gaussian isotropic widths
- `eps::Real=1e-12`: Numerical tolerance for NUFFT computation
- `k_units::Symbol=:cycles`: Unit convention for k-space (`:cycles` or `:angular`)
- `norm::Symbol=:unit_peak`: Normalization type (`:unit_peak` or `:none`)
- `nbins::Int=12`: Number of bins for σ variation (when σ varies per point)

# Returns
- `F::Vector{ComplexF64}`: M-dimensional k-space signal vector representing the weighted sum of Gaussian splats
"""
function kspace_sum_gaussians_nufft(points::AbstractMatrix, kpts::AbstractMatrix;
                                    weights=nothing, sigma, eps::Real=1e-12,
                                    k_units::Symbol=:cycles, norm::Symbol=:unit_peak,
                                    nbins::Int=12)
    N, d  = size(points)
    M, d2 = size(kpts)
    d == d2 || throw(ArgumentError("Dim mismatch: points are N×$d, kpts are M×$d2"))

    # weights
    w = weights === nothing ? ones(ComplexF64, N) : ComplexF64.(weights)
    length(w) == N || throw(ArgumentError("weights must have length N=$N"))

    # Precompute norms of k for envelopes (depending on k_units)
    # If k_units = :cycles → multiplier in exponent is 2π^2
    # If k_units = :angular → multiplier in exponent is 1/2
    k2     = @views vec(sum(kpts.^2; dims=2))   # length M
    k2cols = (d == 1) ? nothing : @views kpts.^2
    env_mul = k_units === :cycles ? (2π^2) : (0.5)

    # Prefactor base for unit-peak
    FT_VOL_BASE = (norm === :unit_peak) ? (2π)^(d/2) : 1.0

    # ---------- Case A: scalar σ (exact) ----------
    if sigma isa Real
        σ = float(sigma); σ > 0 || throw(ArgumentError("sigma must be > 0"))
        F0 = _dirac_sum_nufft(points, kpts, w; eps=eps, k_units=k_units)
        env = @. exp(-env_mul * (σ^2) * k2)
        pref = FT_VOL_BASE * (norm === :unit_peak ? σ^d : 1.0)
        return pref .* env .* F0
    end

    # ---------- Case B: identical-row anisotropic σ (exact) ----------
    if sigma isa AbstractMatrix
        size(sigma) == (N, d) || throw(ArgumentError("sigma matrix must be N×d = $N×$d"))
        any(<=(0), sigma) && throw(ArgumentError("all σ_{i,μ} must be > 0"))
        # check identical rows
        firstrow = vec(sigma[1, :])
        if all(@views sigma[i, :] == firstrow for i in 2:N)
            σrow  = Float64.(firstrow)
            σ2row = σrow.^2
            F0 = _dirac_sum_nufft(points, kpts, w; eps=eps, k_units=k_units)
            t = if d == 1
                @. σ2row[1] * k2
            else
                tmp = similar(k2)
                mul!(tmp, k2cols, σ2row)  # tmp[j] = Σ_μ k_{j,μ}^2 σ_μ^2
                tmp
            end
            env = @. exp(-env_mul * t)
            pref = FT_VOL_BASE * (norm === :unit_peak ? prod(σrow) : 1.0)
            return pref .* env .* F0
        end
        # fall through → treat like isotropic per-point with σ_eff
        σ = sqrt.(vec(sum(abs2, eachrow(sigma)))/d)
    elseif sigma isa AbstractVector
        length(sigma) == N || throw(ArgumentError("sigma vector must have length N=$N"))
        any(<=(0), sigma) && throw(ArgumentError("all σ_i must be > 0"))
        σ = Float64.(sigma)
    else
        throw(ArgumentError("Unsupported sigma type $(typeof(sigma))"))
    end

    # ---------- Case C: isotropic, σ_i varying (binned) ----------
    nbins = max(1, nbins)
    σmin, σmax = minimum(σ), maximum(σ)
    if σmin == σmax
        # actually constant
        σconst = σmin
        F0 = _dirac_sum_nufft(points, kpts, w; eps=eps, k_units=k_units)
        env = @. exp(-env_mul * (σconst^2) * k2)
        pref = FT_VOL_BASE * (norm === :unit_peak ? σconst^d : 1.0)
        return pref .* env .* F0
    end

    edges = range(σmin, σmax; length=nbins+1)
    F = zeros(ComplexF64, M)
    for b in 1:nbins
        lo = edges[b]; hi = edges[b+1]
        mask = (σ .>= lo) .& (σ .< hi)
        if b == nbins
            mask .|= (σ .== hi)
        end
        idx = findall(mask)
        isempty(idx) && continue

        σb = mean(@view σ[idx])
        Fb0 = _dirac_sum_nufft(points[idx, :], kpts, w[idx]; eps=eps, k_units=k_units)
        envb = @. exp(-env_mul * (σb^2) * k2)
        prefb = FT_VOL_BASE * (norm === :unit_peak ? σb^d : 1.0)
        @. F += (prefb * envb) * Fb0
    end
    return F
end

function _dirac_sum_nufft(
    points::AbstractMatrix,
    kpts::AbstractMatrix,
    w;
    eps::Real = 1e-6,
    k_units::Symbol = :cycles
)
    N, d = size(points)
    M, d2 = size(kpts)
    d == d2 || throw(ArgumentError("Dim mismatch: points are N×$d, kpts are M×$d2"))

    T = promote_type(eltype(points), eltype(kpts))
    T <: AbstractFloat || (T = Float64)

    x = Matrix{T}(points)

    s = if k_units === :cycles
        2T(pi) .* Matrix{T}(kpts)
    elseif k_units === :angular
        Matrix{T}(kpts)
    else
        throw(ArgumentError("k_units must be :cycles or :angular"))
    end

    c = if w isa AbstractVector
        length(w) == N || throw(ArgumentError("weights must have length N=$N"))
        Vector{Complex{T}}(w)
    elseif w isa AbstractMatrix
        size(w, 1) == N || throw(ArgumentError("weight matrix must be N×K, got $(size(w))"))
        Matrix{Complex{T}}(w)
    else
        throw(ArgumentError("w must be a vector or matrix of weights"))
    end

    isign = -1
    use_gpu = (@isdefined(CUDA)) && CUDA.functional()

    if use_gpu
        try
            if d == 1
                xg = CUDA.cu(view(x, :, 1))
                sg = CUDA.cu(view(s, :, 1))
                cg = CUDA.cu(c)
                fkg = ndims(cg) == 1 ? CUDA.CuArray{Complex{T}}(undef, M) : CUDA.CuArray{Complex{T}}(undef, M, size(cg, 2))
                FINUFFT.nufft1d3!(xg, cg, isign, T(eps), sg, fkg)
                return Array(fkg)
            elseif d == 2
                xg = CUDA.cu(view(x, :, 1)); yg = CUDA.cu(view(x, :, 2))
                sg = CUDA.cu(view(s, :, 1)); tg = CUDA.cu(view(s, :, 2))
                cg = CUDA.cu(c)
                fkg = ndims(cg) == 1 ? CUDA.CuArray{Complex{T}}(undef, M) : CUDA.CuArray{Complex{T}}(undef, M, size(cg, 2))
                FINUFFT.nufft2d3!(xg, yg, cg, isign, T(eps), sg, tg, fkg)
                return Array(fkg)
            elseif d == 3
                xg = CUDA.cu(view(x, :, 1)); yg = CUDA.cu(view(x, :, 2)); zg = CUDA.cu(view(x, :, 3))
                sg = CUDA.cu(view(s, :, 1)); tg = CUDA.cu(view(s, :, 2)); ug = CUDA.cu(view(s, :, 3))
                cg = CUDA.cu(c)
                fkg = ndims(cg) == 1 ? CUDA.CuArray{Complex{T}}(undef, M) : CUDA.CuArray{Complex{T}}(undef, M, size(cg, 2))
                FINUFFT.nufft3d3!(xg, yg, zg, cg, isign, T(eps), sg, tg, ug, fkg)
                return Array(fkg)
            else
                throw(ArgumentError("Only d ∈ {1,2,3} supported by FINUFFT type-3"))
            end
        catch err
            @warn "GPU NUFFT failed, falling back to CPU" exception=(err, catch_backtrace())
        end
    end

    if d == 1
        return FINUFFT.nufft1d3(x[:,1], c, isign, T(eps), s[:,1])
    elseif d == 2
        return FINUFFT.nufft2d3(x[:,1], x[:,2], c, isign, T(eps), s[:,1], s[:,2])
    elseif d == 3
        return FINUFFT.nufft3d3(x[:,1], x[:,2], x[:,3], c, isign, T(eps), s[:,1], s[:,2], s[:,3])
    else
        throw(ArgumentError("Only d ∈ {1,2,3} supported by FINUFFT type-3"))
    end
end


"""
Compute discrete Fourier transform of weighted dirac sum using NUFFT.

This internal helper function evaluates the non-uniform discrete Fourier transform
of a weighted sum of dirac delta functions located at `points` in d-dimensional space.
It serves as the computational backbone for the k-TES framework's signal synthesis,
utilizing the FINUFFT library for efficient non-uniform FFT computation. The function
supports both CPU and GPU execution paths with automatic fallback and handles
multiple weight configurations for multi-channel or multi-ensemble simulations.

# Arguments
- `points::AbstractMatrix`: N×d matrix of dirac locations in d-dimensional space
- `kpts::AbstractMatrix`: M×d matrix of k-space sample locations
- `w`: Complex weight vector or matrix of size N×K for K channels/ensembles
- `eps::Real=1e-6`: Relative tolerance for NUFFT computation
- `k_units::Symbol=:cycles`: Unit convention for k-space (`:cycles` or `:angular`)

# Returns
- `F::Vector{ComplexF64}` or `Matrix{ComplexF64}`: M×K matrix of k-space samples
  representing the weighted sum of dirac functions, where K is the number of weight channels
"""
function _dirac_sum_nufft1(points::AbstractMatrix, kpts::AbstractMatrix, w;
                          eps::Real=1e-6, k_units::Symbol=:cycles)
    N, d  = size(points)
    M, d2 = size(kpts)
    d == d2 || throw(ArgumentError("Dim mismatch: points are N×$d, kpts are M×$d2"))

    # Choose working float type (Float32 or Float64) from inputs
    Tpoints = eltype(points)
    Tkpts   = eltype(kpts)
    T = promote_type(Tpoints, Tkpts)
    T <: AbstractFloat || (T = Float64)  # fallback

    # Accept vector or matrix of weights and promote to Complex{T}
    if w isa AbstractVector
        length(w) == N || throw(ArgumentError("weights must have length N=$N"))
        if w isa AbstractVector{Complex{T}}
            c = w
        else
            c = Complex{T}.(w)
        end
        ntrans = 1
    elseif w isa AbstractMatrix
        size(w, 1) == N || throw(ArgumentError("weight matrix must be N×K, got $(size(w))"))
        if w isa AbstractMatrix{Complex{T}}
            c = w
        else
            c = Complex{T}.(w)
        end
        ntrans = size(c, 2)
    else
        throw(ArgumentError("w must be a vector or matrix of weights"))
    end

    # Coordinates in T, avoid copies if possible
    x = points isa Matrix{T} ? points : Array{T}(points)

    # k-sampling points, possibly rescaled
    if k_units === :cycles
        s = kpts isa Matrix{T} ? copy(kpts) : Array{T}(kpts)
        @. s = (T(2) * T(π)) * s
    elseif k_units === :angular
        s = kpts isa Matrix{T} ? kpts : Array{T}(kpts)
    else
        throw(ArgumentError("k_units must be :cycles or :angular"))
    end

    isign = -1
    use_gpu = (@isdefined(CUDA)) && CUDA.functional()

    # ---------------- GPU path ----------------
    if use_gpu
        try
            if d == 1
                xg  = CUDA.cu(x[:, 1])
                sg  = CUDA.cu(s[:, 1])
                cg  = CUDA.cu(c)
                fkg = ntrans == 1 ? CUDA.CuArray{Complex{T}}(undef, M) :
                                    CUDA.CuArray{Complex{T}}(undef, M, ntrans)
                FINUFFT.nufft1d3!(xg, cg, isign, T(eps), sg, fkg)
                return Array(fkg)

            elseif d == 2
                xg  = CUDA.cu(x[:, 1]); yg = CUDA.cu(x[:, 2])
                sg  = CUDA.cu(s[:, 1]); tg = CUDA.cu(s[:, 2])
                cg  = CUDA.cu(c)
                fkg = ntrans == 1 ? CUDA.CuArray{Complex{T}}(undef, M) :
                                    CUDA.CuArray{Complex{T}}(undef, M, ntrans)
                FINUFFT.nufft2d3!(xg, yg, cg, isign, T(eps), sg, tg, fkg)
                return Array(fkg)

            elseif d == 3
                xg  = CUDA.cu(x[:, 1]); yg = CUDA.cu(x[:, 2]); zg = CUDA.cu(x[:, 3])
                sg  = CUDA.cu(s[:, 1]); tg = CUDA.cu(s[:, 2]); ug = CUDA.cu(s[:, 3])
                cg  = CUDA.cu(c)
                fkg = ntrans == 1 ? CUDA.CuArray{Complex{T}}(undef, M) :
                                    CUDA.CuArray{Complex{T}}(undef, M, ntrans)
                FINUFFT.nufft3d3!(xg, yg, zg, cg, isign, T(eps), sg, tg, ug, fkg)
                return Array(fkg)

            else
                throw(ArgumentError("Only d ∈ {1,2,3} supported by FINUFFT type-3"))
            end
        catch
            # If GPU path fails, fall back to CPU
        end
    end

    # ---------------- CPU path ----------------
    if d == 1
        return FINUFFT.nufft1d3(x[:,1], c, isign, T(eps), s[:,1])
    elseif d == 2
        return FINUFFT.nufft2d3(x[:,1], x[:,2], c, isign, T(eps), s[:,1], s[:,2])
    elseif d == 3
        return FINUFFT.nufft3d3(x[:,1], x[:,2], x[:,3],
                                c, isign, T(eps),
                                s[:,1], s[:,2], s[:,3])
    else
        throw(ArgumentError("Only d ∈ {1,2,3} supported by FINUFFT type-3"))
    end
end

# --- Type-3 NUFFT dirac sum helper (d=1..3), with k-units control ---
# FINUFFT simple API: exp(i*isign*s⋅x), no 2π inside.
# If k_units == :cycles  → s = 2π * k
# If k_units == :angular → s = k          (you already supplied rad/unit)
function _dirac_sum_nufft2(points::AbstractMatrix, kpts::AbstractMatrix, w;
                          eps::Real=1e-6, k_units::Symbol=:cycles)
    N, d  = size(points)
    M, d2 = size(kpts)
    d == d2 || throw(ArgumentError("Dim mismatch: points are N×$d, kpts are M×$d2"))

    # Choose working float type (Float32 or Float64) from inputs
    Tpoints = eltype(points)
    Tkpts   = eltype(kpts)
    T = promote_type(Tpoints, Tkpts)
    T <: AbstractFloat || (T = Float64)  # fallback

    # Accept vector or matrix of weights and promote to Complex{T}
    if w isa AbstractVector
        length(w) == N || throw(ArgumentError("weights must have length N=$N"))
        if w isa AbstractVector{Complex{T}}
            c = w
        else
            c = Complex{T}.(w)
        end
        ntrans = 1
    elseif w isa AbstractMatrix
        size(w, 1) == N || throw(ArgumentError("weight matrix must be N×K, got $(size(w))"))
        if w isa AbstractMatrix{Complex{T}}
            c = w
        else
            c = Complex{T}.(w)
        end
        ntrans = size(c, 2)
    else
        throw(ArgumentError("w must be a vector or matrix of weights"))
    end

    # Coordinates in T, avoid copies if possible
    x = points isa Matrix{T} ? points : Array{T}(points)

    # k-sampling points, possibly rescaled
    if k_units === :cycles
        s = kpts isa Matrix{T} ? copy(kpts) : Array{T}(kpts)
        @. s = (T(2) * T(π)) * s
    elseif k_units === :angular
        s = kpts isa Matrix{T} ? kpts : Array{T}(kpts)
    else
        throw(ArgumentError("k_units must be :cycles or :angular"))
    end

    isign = -1
    use_gpu = (@isdefined(CUDA)) && CUDA.functional()

    # ---------------- GPU path ----------------
    if use_gpu
        try
            if d == 1
                xg  = CUDA.cu(x[:, 1])
                sg  = CUDA.cu(s[:, 1])
                cg  = CUDA.cu(c)
                fkg = ntrans == 1 ? CUDA.CuArray{Complex{T}}(undef, M) :
                                    CUDA.CuArray{Complex{T}}(undef, M, ntrans)
                FINUFFT.nufft1d3!(xg, cg, isign, T(eps), sg, fkg)
                return Array(fkg)

            elseif d == 2
                xg  = CUDA.cu(x[:, 1]); yg = CUDA.cu(x[:, 2])
                sg  = CUDA.cu(s[:, 1]); tg = CUDA.cu(s[:, 2])
                cg  = CUDA.cu(c)
                fkg = ntrans == 1 ? CUDA.CuArray{Complex{T}}(undef, M) :
                                    CUDA.CuArray{Complex{T}}(undef, M, ntrans)
                FINUFFT.nufft2d3!(xg, yg, cg, isign, T(eps), sg, tg, fkg)
                return Array(fkg)

            elseif d == 3
                xg  = CUDA.cu(x[:, 1]); yg = CUDA.cu(x[:, 2]); zg = CUDA.cu(x[:, 3])
                sg  = CUDA.cu(s[:, 1]); tg = CUDA.cu(s[:, 2]); ug = CUDA.cu(s[:, 3])
                cg  = CUDA.cu(c)
                fkg = ntrans == 1 ? CUDA.CuArray{Complex{T}}(undef, M) :
                                    CUDA.CuArray{Complex{T}}(undef, M, ntrans)
                FINUFFT.nufft3d3!(xg, yg, zg, cg, isign, T(eps), sg, tg, ug, fkg)
                return Array(fkg)

            else
                throw(ArgumentError("Only d ∈ {1,2,3} supported by FINUFFT type-3"))
            end
        catch
            # If GPU path fails, fall back to CPU
        end
    end

    # ---------------- CPU path ----------------
    if d == 1
        return FINUFFT.nufft1d3(x[:,1], c, isign, T(eps), s[:,1])
    elseif d == 2
        return FINUFFT.nufft2d3(x[:,1], x[:,2], c, isign, T(eps), s[:,1], s[:,2])
    elseif d == 3
        return FINUFFT.nufft3d3(x[:,1], x[:,2], x[:,3],
                                c, isign, T(eps),
                                s[:,1], s[:,2], s[:,3])
    else
        throw(ArgumentError("Only d ∈ {1,2,3} supported by FINUFFT type-3"))
    end
end

"""
    k_resize(base_image, Nx::Integer, Ny::Integer; smoothing=true, w_func=cosine) -> Array

Resize a 2D k-space image to target dimensions `Nx` × `Ny`, with optional edge smoothing windowing.
"""
function k_resize(base_image, Nx, Ny; smoothing = true, w_func = cosine)    
    # This could be optimized

    Nx_LF = size(base_image, 1)
    Ny_LF = size(base_image, 2)
    if Nx_LF != Nx || Ny_LF != Ny
        if Ny_LF < Ny
            diff = Ny - Ny_LF
            top_filling = Int(ceil((diff) / 2))
            bot_filling = Int(floor((diff) / 2))
            base_image = hcat(base_image, zeros(size(base_image)[1], top_filling))
            base_image = hcat(zeros(size(base_image)[1], bot_filling), base_image)
            if smoothing
                w_hann = w_func(Ny)
                w_matrix = (w_hann .* ones(1, Ny))'
                base_image = base_image .* w_matrix                
            end
        else
            diff = Ny_LF - Ny
            top_removal = Int(ceil(diff / 2)) 
            bot_removal = Int(floor(diff / 2))
            base_image = base_image[:, (top_removal+1):(end-bot_removal)]
            
            if smoothing
                w_hann = w_func(Ny)
                w_matrix = (w_hann .* ones(1, Ny))'
                base_image = base_image .* w_matrix                
            end
        end
        if Nx_LF != Nx        
            @error "This needs to get fixed!"
            return -1
        end
    end
    return base_image
end

"""
    k_resize(base_image, target_image; smoothing=true, w_func=hanning) -> Array

Resize `base_image` so its dimensions match `target_image`.
"""
function k_resize(base_image, target_image; smoothing = true, w_func = hanning)    
    if size(base_image)!=size(target_image)
        Nx = size(target_image, 1)
        Ny = size(target_image, 2)
        return k_resize(base_image, Nx, Ny; smoothing = smoothing, w_func = w_func)
    end
    return base_image
end

"""
    non_cartesian_ifft(seq::Sequence, kdata, Nx::Int, Ny::Int; ϵ=1e-6) -> Matrix{ComplexF64}

Perform 2D type-1 Non-Uniform Fast Fourier Transform (NUFFT) reconstruction on non-Cartesian k-space data.
"""
function non_cartesian_ifft(seq::Sequence, kdata, Nx::Int, Ny::Int; ϵ = 1e-6)
    _, k = get_kspace(seq)
    kx = k[:,1]
    ky = k[:,2]
    kx = vec(kx)
    ky = vec(ky)
    c  = ComplexF64.(vec(kdata)) 
    kmax = maximum(abs, vcat(kx, ky))
    if kmax > π + ϵ
        kx = (π/kmax) .* kx
        ky = (π/kmax) .* ky
    end

    F = nufft2d1(kx, ky, c, +1, 1e-12, Nx, Ny)
    return F[:,:,1]./length(kdata)
end