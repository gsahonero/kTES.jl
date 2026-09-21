export brain_phantom_3T, brain_phantom_HF, brain_phantom_LF

"""
    brain_phantom_3T(; ss=4, us=1) -> Phantom
    brain_phantom_HF(; ss=4, us=1) -> Phantom

Generate a 2D digital brain phantom with standard High-Field (HF, 1.5T - 3T clinical MRI) relaxation properties.

# Relaxation Parameters (3T)
- **CSF**: \$T_1 = 2.569\\text{ s}, T_2 = 329\\text{ ms}, T_2^* = 58\\text{ ms}, \\rho = 1.0\$
- **Gray Matter (GM)**: \$T_1 = 1.153\\text{ s}, T_2 = 83\\text{ ms}, T_2^* = 69\\text{ ms}, \\rho = 0.86\$
- **White Matter (WM)**: \$T_1 = 0.746\\text{ s}, T_2 = 70\\text{ ms}, T_2^* = 31\\text{ ms}, \\rho = 0.77\$

# Keyword Arguments
- `ss::Int = 4`: Subsampling factor for spatial resolution.
- `us::Int = 1`: Upsampling factor.
"""
function brain_phantom_3T(; ss = 4, us = 1)
    phantom_values_HF = 
        Dict(
            # ρ, T1, T2, T2*, Δw
            "CSF" => [1, 2.569, 0.329, 0.058, 0],
            "GM" => [0.86, 1.153, 0.083, 0.069, 0],
            "WM" => [0.77, 0.746, 0.070, 0.031, 0], #T2s = 0.061
            "FAT1" => [0, 0, 0, 0, 0],
            "MUSCLE" => [0, 0, 0, 0, 0],
            "SKIN/MUSCLE" => [0, 0, 0, 0, 0],
            "SKULL" => [0, 0, 0, 0, 0],
            "VESSELS" => [0, 0, 0, 0, 0],
            "FAT2" => [0, 0, 0, 0, 0],
            "DURA" => [0, 0, 0, 0, 0],
            "MARROW" => [0, 0, 0, 0, 0])
    phantom = brain_phantom2D(;ss = ss, us = us, tissue_properties = phantom_values_HF)
    return phantom
end

"""
    brain_phantom_HF(; ss=4, us=1) -> Phantom

Alias for [`brain_phantom_3T`](@ref). Generates a 2D digital brain phantom with standard High-Field (HF, 3T) MRI relaxation parameters.
"""
function brain_phantom_HF(; ss = 4, us = 1)
    return brain_phantom_3T(; ss = ss, us = us)
end

"""
    brain_phantom_LF(; ss=4, us=1) -> Phantom

Generate a 2D digital brain phantom calibrated for Low-Field (LF, 0.55T) MRI physics.

# Low-Field Contrast Characteristics
At 0.55T, \$T_1\$ relaxation is markedly shortened due to reduced spectral density at lower Larmor frequencies,
while \$T_2\$ values are moderately increased. Proton density is scaled by \$0.55/1.5\$ to account for lower thermal equilibrium polarization.

- **CSF**: \$T_1 = 2.625\\text{ s}, T_2 = 558\\text{ ms}, T_2^* = 58\\text{ ms}\$
- **Gray Matter (GM)**: \$T_1 = 0.635\\text{ s}, T_2 = 119\\text{ ms}, T_2^* = 95\\text{ ms}\$
- **White Matter (WM)**: \$T_1 = 0.526\\text{ s}, T_2 = 98\\text{ ms}, T_2^* = 84\\text{ ms}\$

# Keyword Arguments
- `ss::Int = 4`: Subsampling factor.
- `us::Int = 1`: Upsampling factor.
"""
function brain_phantom_LF(; ss = 4, us = 1)
    field_scaler = 0.55/1.5
    phantom_values_LF = 
        Dict(
        # ρ, T1, T2, T2*, Δw
        "CSF" => [field_scaler*1, 2.625, 0.558, 0.058, 0],
        "GM" => [field_scaler*0.86, 0.635, 0.119, 0.095, 0],
        "WM" => [field_scaler*0.77, 0.526, 0.098, 0.084, 0],
        "FAT1" => [0, 0, 0, 0, 0],
        "MUSCLE" => [0, 0, 0, 0, 0],
        "SKIN/MUSCLE" => [0, 0, 0, 0, 0],
        "SKULL" => [0, 0, 0, 0, 0],
        "VESSELS" => [0, 0, 0, 0, 0],
        "FAT2" => [0, 0, 0, 0, 0],
        "DURA" => [0, 0, 0, 0, 0],
        "MARROW" => [0, 0, 0, 0, 0])
    phantom = brain_phantom2D(;ss = ss, us = us, tissue_properties = phantom_values_LF)
    return phantom
end