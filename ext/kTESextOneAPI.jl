module kTESextOneAPI

using kTES
using oneAPI
using KernelAbstractions

function __init__()
    if oneAPI.functional()
        kTES.ACCELERATOR_REF[] = :oneapi
        kTES.ACCELERATOR_BACKEND_REF[] = oneAPI.oneAPIBackend()
    end
end

function kTES._ka_array(::oneAPI.oneAPIBackend, A::AbstractArray)
    return oneAPI.oneArray(A)
end

function kTES._ka_zeros(::Type{T}, ::oneAPI.oneAPIBackend, dims...) where {T}
    return oneAPI.oneArray(zeros(T, dims...))
end

kTES._to_host(A::oneAPI.oneArray) = Array(A)

end # module
