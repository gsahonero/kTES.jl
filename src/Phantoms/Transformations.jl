export jiggle_phantom, rotate_phantom, phantom_diff, create_grid_space, oversample_phantom, create_2d_grid_space, extract_z0_slice, phantom_from_image_maps

using Random: Xoshiro, rand

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

@inline function _rotation_entries(θx::Real, θy::Real, θz::Real)
    cx, sx = cos(θx), sin(θx)
    cy, sy = cos(θy), sin(θy)
    cz, sz = cos(θz), sin(θz)

    r11 = cz*cy
    r12 = cz*sy*sx - sz*cx
    r13 = cz*sy*cx + sz*sx

    r21 = sz*cy
    r22 = sz*sy*sx + cz*cx
    r23 = sz*sy*cx - cz*sx

    r31 = -sy
    r32 = cy*sx
    r33 = cy*cx

    return (r11, r12, r13, r21, r22, r23, r31, r32, r33)
end

@inline function _same_length_phantom_fields(phantom::Phantom)
    n = length(phantom.x)
    @assert length(phantom.y)   == n
    @assert length(phantom.z)   == n
    @assert length(phantom.T1)  == n
    @assert length(phantom.T2)  == n
    @assert length(phantom.T2s) == n
    @assert length(phantom.ρ)   == n
    @assert length(phantom.Δw)  == n
    return n
end

# ------------------------------------------------------------
# Jiggle
# ------------------------------------------------------------

"""
    jiggle_phantom(phantom::Phantom, seeds::Vector{Int64}, d, off::Vector{Float64}, σ::Vector{Float64}) -> Phantom

Apply stochastic jitter (jiggle) to phantom isochromat positions to model random subvoxel displacements or motion.
"""
function jiggle_phantom(phantom::Phantom, seeds::Vector{Int64}, d, off::Vector{Float64}, σ::Vector{Float64})
    n = _same_length_phantom_fields(phantom)

    @assert length(seeds) >= 3
    @assert length(off) >= 3
    @assert length(σ) >= 3

    rngx = Xoshiro(seeds[1])
    rngy = Xoshiro(seeds[2])
    rngz = Xoshiro(seeds[3])

    ox = Float32(off[1]); oy = Float32(off[2]); oz = Float32(off[3])
    σx = Float32(σ[1]);   σy = Float32(σ[2]);   σz = Float32(σ[3])

    x = phantom.x
    y = phantom.y
    z = phantom.z

    @inbounds for i in 1:n
        x[i] = Float32(x[i]) + σx * (Float32(rand(rngx, d)) - ox)
        y[i] = Float32(y[i]) + σy * (Float32(rand(rngy, d)) - oy)
        z[i] = Float32(z[i]) + σz * (Float32(rand(rngz, d)) - oz)
    end

    return phantom
end

# ------------------------------------------------------------
# Rotate
# ------------------------------------------------------------

"""
    rotate_phantom(phantom::Phantom, θx::Real, θy::Real, θz::Real) -> Phantom

Rotate all spatial coordinates of `phantom` around the coordinate origin by Euler angles `(θx, θy, θz)` in radians.
"""
function rotate_phantom(phantom::Phantom, θx, θy, θz)
    n = _same_length_phantom_fields(phantom)

    R = _rotation_entries(θx, θy, θz)
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = R

    x = phantom.x
    y = phantom.y
    z = phantom.z

    @inbounds for i in 1:n
        xi = x[i]
        yi = y[i]
        zi = z[i]

        x[i] = r11*xi + r12*yi + r13*zi
        y[i] = r21*xi + r22*yi + r23*zi
        z[i] = r31*xi + r32*yi + r33*zi
    end

    return phantom
end

# ------------------------------------------------------------
# Diff
# ------------------------------------------------------------

"""
    phantom_diff(phantom_1::Phantom, phantom_2::Phantom) -> Phantom

Perform a set difference (boolean subtraction) of `phantom_2` from `phantom_1` by filtering out shared isochromat positions.
"""
function phantom_diff(phantom_1::Phantom, phantom_2::Phantom)
    n1 = _same_length_phantom_fields(phantom_1)
    n2 = _same_length_phantom_fields(phantom_2)

    x1, y1, z1 = phantom_1.x, phantom_1.y, phantom_1.z
    x2, y2, z2 = phantom_2.x, phantom_2.y, phantom_2.z

    p2_set = Set{NTuple{3,Float32}}()
    sizehint!(p2_set, n2)
    @inbounds for i in 1:n2
        push!(p2_set, (Float32(x2[i]), Float32(y2[i]), Float32(z2[i])))
    end

    xs = Float32[]
    ys = Float32[]
    zs = Float32[]
    sizehint!(xs, n1)
    sizehint!(ys, n1)
    sizehint!(zs, n1)

    @inbounds for i in 1:n1
        p = (Float32(x1[i]), Float32(y1[i]), Float32(z1[i]))
        if !(p in p2_set)
            push!(xs, p[1])
            push!(ys, p[2])
            push!(zs, p[3])
        end
    end

    n = length(xs)

    T1v  = n == 0 ? Float32[] : fill(Float32(phantom_1.T1[1]),  n)
    T2v  = n == 0 ? Float32[] : fill(Float32(phantom_1.T2[1]),  n)
    T2sv = n == 0 ? Float32[] : fill(Float32(phantom_1.T2s[1]), n)
    ρv   = n == 0 ? Float32[] : fill(Float32(phantom_1.ρ[1]),   n)
    Δwv  = n == 0 ? Float32[] : fill(Float32(phantom_1.Δw[1]),  n)

    return Phantom(
        x   = xs,
        y   = ys,
        z   = zs,
        T1  = T1v,
        T2  = T2v,
        T2s = T2sv,
        ρ   = ρv,
        Δw  = Δwv,
    )
end

# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

"""
    create_grid_space(FOV::Float64, resolution::Float64)
    create_grid_space(FOV::Float64, resolution::Float64, slice_thickness::Float64)

Construct 3D Cartesian coordinates `(gx, gy, gz)` sampled across `FOV` with spacing `resolution`.
"""
function create_grid_space(FOV::Float64, resolution::Float64)
    g = collect(-FOV/2:resolution:FOV/2)
    return g, copy(g), copy(g)
end

function create_grid_space(FOV::Float64, resolution::Float64, slice_thickness::Float64)
    g = collect(-FOV/2:resolution:FOV/2)
    gz = collect(-slice_thickness/2:resolution:slice_thickness/2)
    return g, copy(g), gz
end

"""
    create_2d_grid_space(FOV::Float64, resolution::Float64)
    create_2d_grid_space(FOV::Float64, points::Int)

Construct 2D Cartesian coordinates `(gx, gy, gz)` with `gz` centered at zero.
"""
function create_2d_grid_space(FOV::Float64, resolution::Float64)
    g = collect(-FOV/2:resolution:FOV/2)
    gz = zeros(length(g))
    return g, copy(g), gz
end

function create_2d_grid_space(FOV::Float64, points::Int)
    g = collect(LinRange(-FOV/2, FOV/2, points))
    gz = zeros(length(g))
    return g, copy(g), gz
end
# ------------------------------------------------------------
# Oversample
# ------------------------------------------------------------

"""
    oversample_phantom(phantom::Phantom, factor::Integer; σ = 0.5f-4)

Oversample a discrete `phantom` by duplicating each spin `factor` times with uniform jitter bounded by `σ` [m].
Preserves physical relaxation properties while reducing artificial discretization artifacts in numerical simulations.
"""
function oversample_phantom(phantom::Phantom, factor::Integer; σ = 0.5f-4)
    N = _same_length_phantom_fields(phantom)
    @assert factor >= 1

    M = N * factor

    xs   = Vector{Float32}(undef, M)
    ys   = Vector{Float32}(undef, M)
    zs   = Vector{Float32}(undef, M)
    T1s  = Vector{Float32}(undef, M)
    T2s  = Vector{Float32}(undef, M)
    T2ss = Vector{Float32}(undef, M)
    ρs   = Vector{Float32}(undef, M)
    Δws  = Vector{Float32}(undef, M)

    px   = phantom.x
    py   = phantom.y
    pz   = phantom.z
    pT1  = phantom.T1
    pT2  = phantom.T2
    pT2s = phantom.T2s
    pρ   = phantom.ρ
    pΔw  = phantom.Δw

    σ32 = Float32(σ)

    @inbounds for i in 1:N
        base = (i - 1) * factor

        x0   = Float32(px[i])
        y0   = Float32(py[i])
        z0   = Float32(pz[i])
        T1_0 = Float32(pT1[i])
        T2_0 = Float32(pT2[i])
        T2s0 = Float32(pT2s[i])
        ρ0   = Float32(pρ[i])
        Δw0  = Float32(pΔw[i])

        for k in 1:factor
            idx = base + k

            δx = (rand(Float32) - 0.5f0) * σ32
            δy = (rand(Float32) - 0.5f0) * σ32
            δz = (rand(Float32) - 0.5f0) * σ32

            xs[idx] = x0 + δx
            ys[idx] = y0 + δy
            zs[idx] = z0 + δz

            T1s[idx]  = T1_0
            T2s[idx]  = T2_0
            T2ss[idx] = T2s0
            ρs[idx]   = ρ0
            Δws[idx]  = Δw0
        end
    end

    return Phantom(
        x   = xs,
        y   = ys,
        z   = zs,
        T1  = T1s,
        T2  = T2s,
        T2s = T2ss,
        ρ   = ρs,
        Δw  = Δws,
    )
end

"""
    extract_z0_slice(phantom::Phantom; eps::Float64=1e-3)

Extract unique `(x, y)` spins located on the central slice `|z| <= eps`.
Useful for converting 3D volumetric phantoms into 2D simulation slices.
"""
function extract_z0_slice(phantom::Phantom; eps::Float64=1e-3)
    
    # Step 1: Identify points near z=0
    # We find all indices i where |z[i]| <= eps
    indices = i -> abs(phantom.z[i]) <= eps
    
    # Step 2: Filter indices (using a Set for deduplication of (x, y) pairs)
    unique_xy_pairs = Set{Tuple}()
    filtered_indices = Int[]
    
    # We must iterate through the original indices to ensure all required data (T1, T2, etc.) is collected
    for i in eachindex(phantom.z)
        if abs(phantom.z[i]) <= eps
            pair = (phantom.x[i], phantom.y[i])
            if !(pair in unique_xy_pairs)
                push!(unique_xy_pairs, pair)
                push!(filtered_indices, i)
            end
        end
    end
    
    # If no points are found near z=0, return an empty phantom
    if isempty(filtered_indices)
        return Phantom(Float32[], Float32[], Float32[], Float32[], Float32[], Float32[], Float32[], Float32[])
    end
    
    # Step 3: Extract data only for the unique indices
    filtered_x = [phantom.x[i] for i in filtered_indices]
    filtered_y = [phantom.y[i] for i in filtered_indices]
    filtered_z = [phantom.z[i] for i in filtered_indices]
    filtered_T1 = [phantom.T1[i] for i in filtered_indices]
    filtered_T2 = [phantom.T2[i] for i in filtered_indices]
    filtered_T2s = [phantom.T2s[i] for i in filtered_indices]
    filtered_ρ = [phantom.ρ[i] for i in filtered_indices]
    filtered_Δw = [phantom.Δw[i] for i in filtered_indices]
    
    # Return the new Phantom object
    return Phantom(
        x   = filtered_x,
        y   = filtered_y,
        z   = filtered_z,
        T1  = filtered_T1,
        T2  = filtered_T2,
        T2s = filtered_T2s,
        ρ   = filtered_ρ,
        Δw  = filtered_Δw,
    )
end

using Interpolations

"""
    oversample_map_linear(A::AbstractMatrix, oversampling_factor::Real)

Bilinearly resample a 2D parameter map matrix `A` by `oversampling_factor`.
"""
function oversample_map_linear(A::AbstractMatrix, oversampling_factor::Real)
    @assert oversampling_factor > 0

    Nx, Ny = size(A)
    Nox = round(Int, Nx * oversampling_factor)
    Noy = round(Int, Ny * oversampling_factor)

    if Nox == Nx && Noy == Ny
        return copy(A)
    end

    xs = 1:Nx
    ys = 1:Ny

    itp = linear_interpolation((xs, ys), A)

    xq = range(1, Nx, length=Nox)
    yq = range(1, Ny, length=Noy)

    B = Array{Float64}(undef, Nox, Noy)
    for j in 1:Noy, i in 1:Nox
        B[i, j] = itp(xq[i], yq[j])
    end

    return B
end

"""
    flattened_pixel_coordinates(Nx::Int, Ny::Int, FOVx, FOVy; oversampling_factor::Real=1, order::Symbol=:xfast)

Compute flattened `(x, y)` Cartesian grid coordinates centered at `(0, 0)` spanning `FOVx × FOVy`.
"""
function flattened_pixel_coordinates(Nx::Int, Ny::Int, FOVx, FOVy;
                                     oversampling_factor::Real=1,
                                     order::Symbol=:xfast)

    @assert Nx > 0 && Ny > 0
    @assert oversampling_factor > 0

    Nox = round(Int, Nx * oversampling_factor)
    Noy = round(Int, Ny * oversampling_factor)

    Δx = FOVx / Nox
    Δy = FOVy / Noy

    x_axis = collect(range(-FOVx/2 + Δx/2, step=Δx, length=Nox))
    y_axis = collect(range(-FOVy/2 + Δy/2, step=Δy, length=Noy))

    if order == :xfast
        x = repeat(x_axis, Noy)
        y = repeat(y_axis, inner=Nox)
    elseif order == :yfast
        x = repeat(x_axis, inner=Noy)
        y = repeat(y_axis, Nox)
    else
        error("order must be :xfast or :yfast")
    end

    return x, y
end

"""
    oversample_map_repeat(A::AbstractMatrix, oversampling_factor::Integer)

Nearest-neighbor block repetition of matrix `A`.
"""
function oversample_map_repeat(A::AbstractMatrix, oversampling_factor::Integer)
    @assert oversampling_factor >= 1
    repeat(A, inner=(oversampling_factor, oversampling_factor))
end

"""
    phantom_from_image_maps(T1::AbstractMatrix, T2::AbstractMatrix, PD::AbstractMatrix, FOVx::AbstractFloat, FOVy::AbstractFloat; oversampling_factor::Real=1, order::Symbol=:xfast, drop_zeros::Bool=true)

Create a discrete `Phantom` from quantitative image maps (`T1`, `T2`, `PD`).

# Arguments
- `T1, T2, PD`: 2D matrices containing quantitative values in seconds (T1, T2) and proton density.
- `FOVx, FOVy`: Field of view along x and y [m].

# Keywords
- `oversampling_factor = 1`: Sub-pixel grid oversampling factor.
- `order = :xfast`: Coordinate flattening order (`:xfast` or `:yfast`).
- `drop_zeros = true`: When true, omit points with zero proton density to save memory and compute.
"""
function phantom_from_image_maps(T1::AbstractMatrix,
                                 T2::AbstractMatrix,
                                 PD::AbstractMatrix,
                                 FOVx::AbstractFloat,
                                 FOVy::AbstractFloat;
                                 oversampling_factor::Real=1,
                                 order::Symbol=:xfast,
                                 drop_zeros::Bool=true)

    @assert size(T1) == size(T2) == size(PD)

    T1os = oversampling_factor == 1 ? Float64.(T1) : oversample_map_linear(T1, oversampling_factor)
    T2os = oversampling_factor == 1 ? Float64.(T2) : oversample_map_linear(T2, oversampling_factor)
    PDos = oversampling_factor == 1 ? Float64.(PD) : oversample_map_linear(PD, oversampling_factor)

    Nox, Noy = size(T1os)

    x, y = flattened_pixel_coordinates(size(T1,1), size(T1,2), FOVx, FOVy;
                                       oversampling_factor=oversampling_factor,
                                       order=order)

    T = Float64

    t1  = vec(T1os)
    t2  = vec(T2os)
    ρ   = vec(PDos)
    z   = zeros(T, length(x))
    t2s = zeros(T, length(x))
    Δw  = zeros(T, length(x))

    @assert length(x) == length(y) == length(t1) == length(t2) == length(ρ) == Nox * Noy

    if drop_zeros
        mask = ρ .> 0
        return Phantom{T}(
            x = T.(x[mask]),
            y = T.(y[mask]),
            z = z[mask],
            T1 = t1[mask],
            T2 = t2[mask],
            T2s = t2s[mask],
            ρ = ρ[mask],
            Δw = Δw[mask],
        )
    else
        return Phantom{T}(
            x = T.(x),
            y = T.(y),
            z = z,
            T1 = t1,
            T2 = t2,
            T2s = t2s,
            ρ = ρ,
            Δw = Δw,
        )
    end
end