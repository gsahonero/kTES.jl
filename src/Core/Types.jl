export UniformKernel, TriangularKernel, SincKernel, ϵ_Truncated, fixed_Truncated, NoApproximation, T2sParams, AcqParams, FullNoise, ReducedNoise, GaussianB0, LinearB0, QuadraticB0, Kernel, T2sMethod, NoiseAddition, B0Map, ClusteringTechnique, NormalKMeans, FastKMeans, LloydKMeans, SubsampledKMeans, ParallelKMeansKMeans

"""
    Kernel

Abstract supertype for spatial convolution kernels used in k-space density interpolation and gridding operations.
"""
abstract type Kernel end

"""
    UniformKernel <: Kernel

Boxcar / rectangular spatial interpolation kernel with uniform weighting over the support window.
"""
struct UniformKernel <: Kernel end

"""
    TriangularKernel <: Kernel

Triangular (Bartlett) spatial interpolation kernel with linearly decaying weighting from center to boundary.
"""
struct TriangularKernel <: Kernel end

"""
    SincKernel <: Kernel

Ideal bandlimited spatial interpolation kernel based on the sinc function.
"""
struct SincKernel <: Kernel end

"""
    T2sMethod

Abstract supertype for \$T_2^*\$ intravoxel dephasing evaluation strategies.
"""
abstract type T2sMethod end

"""
    NoApproximation <: T2sMethod

Exact computation of \$T_2^*\$ decay via Monte Carlo isochromat sampling over the local tissue unit support,
reproducing continuous Lorentzian dephasing as described in *the k-TES paper*.
"""
struct NoApproximation <: T2sMethod end

"""
    ϵ_Truncated <: T2sMethod

Approximation method that truncates the dephasing spectrum when the tail amplitude falls below threshold `ϵ`.
"""
struct ϵ_Truncated <: T2sMethod end

"""
    fixed_Truncated <: T2sMethod

Approximation method that restricts the dephasing spectrum to a fixed half-width frequency window `L`.
"""
struct fixed_Truncated <: T2sMethod end

"""
    T2sParams(; method=NoApproximation(), ϵ=0.0, L=0.0)

Parameters configuring how intravoxel \$T_2^*\$ dephasing is treated during k-TES simulation.

# Fields
- `method::T2sMethod`: Evaluation technique (default: `NoApproximation()`).
- `ϵ::Float64`: Threshold tolerance for `ϵ_Truncated`.
- `L::Float64`: Fixed frequency truncation cutoff in Hz for `fixed_Truncated`.
"""
@kwdef mutable struct T2sParams
    method::T2sMethod = NoApproximation()
    ϵ::Float64 = 0.00 # for ϵ_Truncated
    L::Float64 = 0.0 # for fixed_Truncated
end

"""
    NoiseAddition

Abstract supertype defining thermal / complex Gaussian noise addition models.
"""
abstract type NoiseAddition end

"""
    FullNoise <: NoiseAddition

Injects complex Gaussian white noise directly onto the full composite k-space signal.
"""
struct FullNoise <: NoiseAddition end

"""
    ReducedNoise <: NoiseAddition

Noise injection model scaled according to the reduced number of acquisition echoes or tissue components.
"""
struct ReducedNoise <: NoiseAddition end

"""
    AcqParams(; conv_technique, noise_technique, noise_σ, reconstruction_technique)

Configuration container for high-level acquisition simulation in [`simulate_acquisition`](@ref).

# Fields
- `conv_technique::Kernel`: Gridding/interpolation kernel.
- `noise_technique::NoiseAddition`: Noise model (`FullNoise()` or `ReducedNoise()`).
- `noise_σ::Float64`: Standard deviation of complex Gaussian noise (0.0 for noiseless).
- `reconstruction_technique::Function`: Reconstruction callback function (e.g., standard 2D IFFT).
"""
@kwdef mutable struct AcqParams
    conv_technique::Kernel
    noise_technique::NoiseAddition
    noise_σ::Float64    
    reconstruction_technique::Function
end

"""
    B0Map

Abstract supertype for synthetic background \$B_0\$ magnetic field inhomogeneity distributions.
"""
abstract type B0Map end

"""
    GaussianB0 <: B0Map

Synthesizes a spatial \$B_0\$ field inhomogeneity characterized by a smooth Gaussian spatial envelope.
"""
struct GaussianB0 <: B0Map end

"""
    LinearB0 <: B0Map

Synthesizes a linear spatial gradient in the background \$B_0\$ field across the field of view.
"""
struct LinearB0 <: B0Map end

"""
    QuadraticB0 <: B0Map

Synthesizes a quadratic (second-order) spatial field deviation across the field of view.
"""
struct QuadraticB0 <: B0Map end

"""
    ClusteringTechnique

Abstract supertype for clustering algorithms used to group continuous phantom spins into discrete representative tissue bases.
"""
abstract type ClusteringTechnique end

"""
    NormalKMeans <: ClusteringTechnique

Standard \$k\$-means clustering implementation on tissue parameter vectors \$\\theta = [T_1, T_2, T_2^*, \\rho, \\Delta\\omega]\$.
"""
struct NormalKMeans <: ClusteringTechnique end

"""
    FastKMeans <: ClusteringTechnique

Accelerated \$k\$-means variant using triangle inequality acceleration to reduce distance evaluations.
"""
struct FastKMeans <: ClusteringTechnique end

"""
    LloydKMeans(; init_centers=nothing, maxiter=300)

Classic Lloyd algorithm with optional custom initial cluster centroids.

# Fields
- `init_centers::Union{Nothing, Matrix{Float64}}`: Explicit initial centroid matrix (default: `nothing`).
- `maxiter::Int`: Maximum number of iterations (default: 300).
"""
@kwdef struct LloydKMeans <: ClusteringTechnique
    init_centers::Union{Nothing, Matrix{Float64}} = nothing
    maxiter::Int = 300
end

"""
    SubsampledKMeans(; nsub=50_000, refine_full=true, maxiter_refine=5)

Clustering technique that fits \$k\$-means centroids on a random subset of phantom
points (`nsub`) and then assigns all \$N\$ points to the nearest centroid using a
KDTree (`NearestNeighbors.jl`). Optionally runs a short Lloyd refinement pass on
the full data after fitting.

# Fields
- `nsub::Int`: Number of points used for centroid fitting (default: 50,000).
- `refine_full::Bool`: Run a short Lloyd pass on full data after subsampling (default: `true`).
- `maxiter_refine::Int`: Maximum iterations for the refinement pass (default: 5).
"""
@kwdef struct SubsampledKMeans <: ClusteringTechnique
    nsub::Int         = 50_000
    refine_full::Bool = true
    maxiter_refine::Int = 5
end

"""
    ParallelKMeansKMeans(; algo=Hamerly(), nthreads=Threads.nthreads(), maxiter=300, tol=1e-6)

Multi-threaded \$k\$-means using `ParallelKMeans.jl`. Supports multiple algorithm
variants (`Lloyd`, `Hamerly`, `Yinyang`, `Coreset`). Uses all available Julia threads
by default.

# Fields
- `algo`: Algorithm variant — `Lloyd()`, `Hamerly()`, `Yinyang()`, `Coreset()` (default: `Hamerly()`).
- `nthreads::Int`: Number of worker threads (default: `Threads.nthreads()`).
- `maxiter::Int`: Maximum iterations (default: 300).
- `tol::Float64`: Convergence tolerance (default: 1e-6).
"""
@kwdef struct ParallelKMeansKMeans <: ClusteringTechnique
    algo     = Hamerly()
    nthreads::Int     = Threads.nthreads()
    maxiter::Int      = 300
    tol::Float64      = 1e-6
end