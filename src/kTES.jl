module kTES

using KomaMRI
using FFTW
using Random
using Plots
using PlotlyJS
using PlotlyBase
using DSP
using Distributions
using StaticArrays
using LinearAlgebra
using FINUFFT
using KernelAbstractions
# -- check
using NearestNeighbors
using Clustering
using Statistics
using ParallelKMeans: Lloyd, Hamerly, Yinyang, Coreset
# -- check
using Reexport
using Base.Threads
using Logging, LoggingExtras
using Interpolations

@reexport using KomaMRI
@reexport using FFTW
@reexport using Distributions
@reexport using PlotlyJS
@reexport using ParallelKMeans

const ACCELERATOR_REF = Ref{Symbol}(:cpu)
const ACCELERATOR_BACKEND_REF = Ref{KernelAbstractions.Backend}(KernelAbstractions.CPU())

"""
    active_accelerator() -> Symbol

Return the active hardware acceleration backend symbol (e.g. `:cpu`, `:cuda`, `:metal`, `:rocm`).
"""
active_accelerator() = ACCELERATOR_REF[]

"""
    has_accelerator() -> Bool

Check if non-CPU hardware acceleration (GPU) is currently loaded and active.
"""
has_accelerator() = ACCELERATOR_REF[] !== :cpu

"""
    get_accelerator_backend() -> KernelAbstractions.Backend

Return the current KernelAbstractions backend instance used for GPU kernel dispatch.
"""
get_accelerator_backend() = ACCELERATOR_BACKEND_REF[]

export active_accelerator, has_accelerator, get_accelerator_backend

function _ka_array(::KernelAbstractions.CPU, A::AbstractArray)
    return Array(A)
end

function _ka_zeros(::Type{T}, ::KernelAbstractions.CPU, dims...) where {T}
    return zeros(T, dims...)
end

_to_host(A) = Array(A)

include("Core/Types.jl")
include("Sequences/Types.jl")
include("Sequences/GE.jl")
include("Sequences/SE.jl")
include("Sequences/TSE.jl")
include("Sequences/Preparation.jl")
include("Sequences/RF_pulses.jl")
include("Sequences/Tools.jl")

include("Core/TissueCore.jl")
include("Core/Acquisition.jl")
include("Core/Clustering.jl")
include("Core/DSP.jl")
include("Core/Noise.jl")
include("Core/OffResonance.jl")
include("Core/TissueCoreOff.jl")
include("Phantoms/Types.jl")
include("Phantoms/Brain.jl")
include("Phantoms/Calibration.jl")
include("Phantoms/Geometries.jl")
include("Phantoms/Transformations.jl")
include("Utils/SimulationAddons.jl")
include("Utils/Plots.jl")

export KomaMRI

end # module kTES
