export add_noise, default_noise_params

"""
    add_noise(raw::Array{ComplexF32, 3}, p::SeqParams, σ::Float64, technique::FullNoise, sys::Scanner)

Adds complex-valued Gaussian noise to the raw k-space data based on the specified noise standard deviation and acquisition parameters.

This function implements a noise model where the noise is sampled from a complex normal distribution with zero mean and standard deviation `σ / √(p.NSA)`, where `p.NSA` is the number of signal averages. The noise is added to the input `raw` k-space data to simulate realistic acquisition conditions. The output dimensions are preserved, and the noise is applied to the entire k-space volume.

# Arguments
- `raw::Array{ComplexF32, 3}`: Input raw k-space data of dimensions `(Nx × Ny × Nz, 1)` 
- `p::SeqParams`: Sequence parameters containing acquisition details such as `t_ADC`, `Nx`, `Ny`, and `NSA`
- `σ::Float64`: Standard deviation of the noise distribution
- `technique::FullNoise`: Noise model type (placeholder for future extensions)
- `sys::Scanner`: Scanner system parameters including `ADC_Δt` for sampling time conversion

# Returns
- `Array{ComplexF32, 3}`: Noisy k-space data with the same dimensions as input `raw`
"""
function add_noise(raw::Array{ComplexF32, 3}, p::SeqParams, σ::Float64, technique::FullNoise, sys::Scanner)
    OF = round((p.t_ADC / p.Nx)/ sys.ADC_Δt, digits = 2)
    noise = rand(Normal(0, (σ / sqrt(p.NSA))), (Int(OF * p.Nx * p.Ny), 1)) + rand(Normal(0, (σ / sqrt(p.NSA))), (Int(OF * p.Nx * p.Ny), 1)) .* 1im
    return raw .+ noise
end

function add_noise(raw::Matrix{ComplexF32}, p::SeqParams, σ::Float64, technique::FullNoise, sys::Scanner)
    OF = round((p.t_ADC / p.Nx)/ sys.ADC_Δt, digits = 2)
    noise = rand(Normal(0, (σ / sqrt(p.NSA))), (Int(OF * p.Nx), Int(p.Ny), 1)) + rand(Normal(0, (σ / sqrt(p.NSA))), (Int(OF * p.Nx), Int(p.Ny), 1)) .* 1im

    #noise = rand(Normal(0, (σ / sqrt(p.NSA)) / sqrt(OF)), (Int(p.Nx), Int(p.Ny), 1)) + rand(Normal(0, (σ / sqrt(p.NSA)) / sqrt(OF)), (Int(p.Nx), Int(p.Ny), 1)) .* 1im
    return raw .+ noise
end

"""
    add_noise(raw::Array{ComplexF32, 3}, p::SeqParams, σ::Float64, technique::ReducedNoise, sys::Scanner)

Adds complex-valued Gaussian noise to the raw k-space data based on the specified noise standard deviation and acquisition parameters.

This function implements a noise model where the noise is sampled from a complex normal distribution with zero mean and standard deviation `σ / √(p.NSA)`, where `p.NSA` is the number of signal averages. The noise is added to the input `raw` k-space data to simulate realistic acquisition conditions. The output dimensions are preserved, and the noise is applied to the entire k-space volume.

# Arguments
- `raw::Array{ComplexF32, 3}`: Input raw k-space data of dimensions `(Nx × Ny × Nz, 1)` 
- `p::SeqParams`: Sequence parameters containing acquisition details such as `t_ADC`, `Nx`, `Ny`, and `NSA`
- `σ::Float64`: Standard deviation of the noise distribution
- `technique::ReducedNoise`: Noise model type (placeholder for future extensions)
- `sys::Scanner`: Scanner system parameters including `ADC_Δt` for sampling time conversion

# Returns
- `Array{ComplexF32, 3}`: Noisy k-space data with the same dimensions as input `raw`
"""
function add_noise(raw::Array{ComplexF32, 3}, p::SeqParams, σ::Float64, technique::ReducedNoise, sys::Scanner)
    OF = p.OF
    noise = rand(Normal(0, (σ / sqrt(p.NSA)) / sqrt(OF)), (Int(p.Nx * p.Ny), 1)) + rand(Normal(0, (σ / sqrt(p.NSA)) / sqrt(OF)), (Int(p.Nx * p.Ny), 1)) .* 1im
    return raw .+ noise
end

function add_noise(raw::Matrix{ComplexF32}, p::SeqParams, σ::Float64, technique::ReducedNoise, sys::Scanner)
    OF = p.OF
    noise = rand(Normal(0, (σ / sqrt(p.NSA)) / sqrt(OF)), (Int(p.Nx), Int(p.Ny), 1)) + rand(Normal(0, (σ / sqrt(p.NSA)) / sqrt(OF)), (Int(p.Nx), Int(p.Ny), 1)) .* 1im
    return raw .+ noise
end

"""
    default_noise_params()

Returns a dictionary containing default noise parameters for MRI acquisition simulation.

This function provides a standard set of noise parameters that are commonly used in MRI simulations. The default standard deviation `σ = 14` is chosen to represent typical noise levels in clinical scanners, while the noise technique is set to `FullNoise()` to indicate full noise modeling including both real and imaginary components.

# Returns
- `Dict{String, Any}`: Dictionary containing default noise parameters with keys:
  - `"σ"`: Standard deviation of the noise distribution (Float64)
  - `"technique"`: Noise modeling technique (FullNoise)
"""
function default_noise_params()
    return Dict("σ" => 14, "technique" => FullNoise())
end