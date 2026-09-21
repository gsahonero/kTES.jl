module kTESextAMDGPU

using kTES
using AMDGPU
using KernelAbstractions

function __init__()
    if AMDGPU.functional()
        kTES.ACCELERATOR_REF[] = :amd
        kTES.ACCELERATOR_BACKEND_REF[] = AMDGPU.ROCBackend()
    end
end

function kTES._ka_array(::AMDGPU.ROCBackend, A::AbstractArray)
    return AMDGPU.ROCArray(A)
end

function kTES._ka_zeros(::Type{T}, ::AMDGPU.ROCBackend, dims...) where {T}
    return AMDGPU.zeros(T, dims...)
end

kTES._to_host(A::AMDGPU.ROCArray) = Array(A)

end # module
