using CUDA

try
    CUDA.functional() ||
        error("CUDA.jl reports that CUDA is not functional.")

    devices = collect(CUDA.devices())

    isempty(devices) &&
        error("No CUDA-compatible device was detected.")

    device = first(devices)
    CUDA.device!(device)

    test_array = CUDA.zeros(Float32, 1024)
    test_array .= 1f0
    CUDA.synchronize()

    println("GPU_AVAILABLE")
    println("GPU_NAME=", CUDA.name(device))

catch error
    println("GPU_UNAVAILABLE")
    showerror(stdout, error)
    println()

    exit(1)
end