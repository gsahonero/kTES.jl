module kTESextCUDA

using kTES
using CUDA
using KernelAbstractions

function __init__()
    if CUDA.functional()
        kTES.ACCELERATOR_REF[] = :cuda
        kTES.ACCELERATOR_BACKEND_REF[] = CUDA.CUDABackend()
    end
end

function kTES._ka_array(::CUDA.CUDABackend, A::AbstractArray)
    return CUDA.CuArray(A)
end

function kTES._ka_zeros(::Type{T}, ::CUDA.CUDABackend, dims...) where {T}
    return CUDA.zeros(T, dims...)
end

kTES._to_host(A::CUDA.CuArray) = Array(A)

end # module
