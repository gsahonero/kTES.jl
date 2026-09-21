module kTESextMetal

using kTES
using Metal
using KernelAbstractions

function __init__()
    if Metal.functional()
        kTES.ACCELERATOR_REF[] = :metal
        kTES.ACCELERATOR_BACKEND_REF[] = Metal.MetalBackend()
    end
end

function kTES._ka_array(::Metal.MetalBackend, A::AbstractArray)
    return Metal.MtlArray(A)
end

function kTES._ka_zeros(::Type{T}, ::Metal.MetalBackend, dims...) where {T}
    return Metal.MtlArray(zeros(T, dims...))
end

kTES._to_host(A::Metal.MtlArray) = Array(A)

end # module
