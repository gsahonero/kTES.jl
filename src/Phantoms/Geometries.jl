export vials_phantom,
       circle_phantom,
       square_phantom,
       rectangle_phantom,
       sphere_phantom,
       cube_phantom,
       box_phantom,
       right_triangle_prism_phantom,
       triangle_prism_phantom,
       trapezoid_prism_phantom,
       ring_phantom,
       solid_disc_phantom,
       cylinder_phantom


using LinearAlgebra


const PROP_KEYS = ("T1", "T2", "T2s", "ρ", "Δw")


# ---------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------


@inline _floattype(::Type{T}) where {T<:AbstractFloat} = T
@inline _floattype(::Type{T}) where {T<:Real} = float(T)


@inline function _promote_float_type(xs...)
    T = promote_type(map(x -> x isa Type ? x : typeof(x), xs)...)
    return _floattype(T)
end


@inline function _phantom_type_from_inputs(args...)
    T = Float32
    @inbounds for a in args
        if a isa AbstractArray
            T = promote_type(T, _floattype(eltype(a)))
        elseif a isa Tuple || a isa AbstractVector
            for v in a
                T = promote_type(T, _floattype(typeof(v)))
            end
        elseif a isa Dict
            for key in PROP_KEYS
                T = promote_type(T, _floattype(typeof(a[key])))
            end
        else
            T = promote_type(T, _floattype(typeof(a)))
        end
    end
    return T
end


# ---------------------------------------------------------------------
# Grids and properties
# ---------------------------------------------------------------------


@inline function _grid(FOV::Real, resolution::Real)
    T = _promote_float_type(FOV, resolution)
    fov = T(FOV)
    res = T(resolution)
    return collect(range(-fov/2, stop=fov/2, step=res))
end


@inline function _props(properties::Dict{String,<:Real}, ::Type{T}) where {T<:AbstractFloat}
    return (
        T1  = T(properties["T1"]),
        T2  = T(properties["T2"]),
        T2s = T(properties["T2s"]),
        ρ   = T(properties["ρ"]),
        Δw  = T(properties["Δw"]),
    )
end


# ---------------------------------------------------------------------
# Coordinate collector with exact tuple deduplication
# ---------------------------------------------------------------------


mutable struct _CoordCollector{T<:AbstractFloat}
    seen::Set{NTuple{3,T}}
    xs::Vector{T}
    ys::Vector{T}
    zs::Vector{T}
end


@inline function _CoordCollector(::Type{T}, n_hint::Integer=0) where {T<:AbstractFloat}
    xs = Vector{T}()
    ys = Vector{T}()
    zs = Vector{T}()
    if n_hint > 0
        sizehint!(xs, n_hint)
        sizehint!(ys, n_hint)
        sizehint!(zs, n_hint)
    end
    return _CoordCollector{T}(Set{NTuple{3,T}}(), xs, ys, zs)
end


@inline function _push_unique!(C::_CoordCollector{T}, x::T, y::T, z::T) where {T}
    xyz = (x, y, z)
    if !(xyz in C.seen)
        push!(C.seen, xyz)
        push!(C.xs, x)
        push!(C.ys, y)
        push!(C.zs, z)
    end
    return nothing
end


# ---------------------------------------------------------------------
# Geometry helpers - Dimension-aware rotation and translation
# ---------------------------------------------------------------------


@inline function _rotation_entries(θx::T, θy::T, θz::T) where {T<:AbstractFloat}
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


@inline function _rotate_translate_2d(dx::T, dy::T, p::NTuple{2,T}, R) where {T<:AbstractFloat}
    # 2D rotation: only use the 2x2 upper-left block
    r11, r12, _, r21, r22, _, _, _, _ = R
    x = r11*dx + r12*dy + p[1]
    y = r21*dx + r22*dy + p[2]
    return x, y
end


@inline function _rotate_translate_3d(dx::T, dy::T, dz::T, p::NTuple{3,T}, R) where {T<:AbstractFloat}
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = R
    x = r11*dx + r12*dy + r13*dz + p[1]
    y = r21*dx + r22*dy + r23*dz + p[2]
    z = r31*dx + r32*dy + r33*dz + p[3]
    return x, y, z
end


@inline function _append_rotated_2d!(C::_CoordCollector{T}, dx::T, dy::T, p::NTuple{2,T}, R) where {T}
    x, y = _rotate_translate_2d(dx, dy, p, R)
    _push_unique!(C, x, y, zero(T))
    return nothing
end


@inline function _append_rotated_3d!(C::_CoordCollector{T}, dx::T, dy::T, dz::T, p::NTuple{3,T}, R) where {T}
    x, y, z = _rotate_translate_3d(dx, dy, dz, p, R)
    _push_unique!(C, x, y, z)
    return nothing
end


@inline function _build_phantom(C::_CoordCollector{T}, pr) where {T<:AbstractFloat}
    n = length(C.xs)
    return Phantom(
        x   = C.xs,
        y   = C.ys,
        z   = C.zs,
        T1  = fill(pr.T1, n),
        T2  = fill(pr.T2, n),
        T2s = fill(pr.T2s, n),
        ρ   = fill(pr.ρ, n),
        Δw  = fill(pr.Δw, n),
    )
end


@inline function _find_range(grid::AbstractVector, lo::Real, hi::Real)
    i1 = searchsortedfirst(grid, lo)
    i2 = searchsortedlast(grid, hi)
    if i1 > i2 || i1 > lastindex(grid) || i2 < firstindex(grid)
        return (firstindex(grid)-1):(firstindex(grid)-2)  # empty range
    end
    i1 = max(i1, firstindex(grid))
    i2 = min(i2, lastindex(grid))
    return i1:i2
end


@inline function _xy_ranges(grid_x, grid_y, cx, cy, rx, ry)
    ix = _find_range(grid_x, cx - rx, cx + rx)
    iy = _find_range(grid_y, cy - ry, cy + ry)
    return ix, iy
end


@inline function _xyz_ranges(grid_x, grid_y, grid_z, cx, cy, cz, rx, ry, rz)
    ix = _find_range(grid_x, cx - rx, cx + rx)
    iy = _find_range(grid_y, cy - ry, cy + ry)
    iz = _find_range(grid_z, cz - rz, cz + rz)
    return ix, iy, iz
end


@inline function _typed_point(p, ::Type{T}, n::Int) where {T<:AbstractFloat}
    return ntuple(i -> T(p[i]), n)
end


@inline function _rot_and_point(::Type{T}, p, θx, θy, θz, ndim::Int) where {T<:AbstractFloat}
    pp = _typed_point(p, T, ndim)
    R = _rotation_entries(T(θx), T(θy), T(θz))
    return pp, R
end


# ---------------------------------------------------------------------
# 2D vial phantom
# ---------------------------------------------------------------------

@doc raw"""
    vials_phantom(n::Integer, N::Int, M::Int, r::Real, properties::Vector{<:Dict}; a=0.01, resolution=1e-3, FOV=20e-2) -> Phantom

Construct a 2D multi-vial calibration phantom comprising an $N \times M$ grid of circular vials with radius `r`,
vial wall separation `a`, and individualized relaxation dictionaries in `properties`.
"""
function vials_phantom(n::Integer, N::Int, M::Int, r::Real,
                       properties::Vector{<:Dict{String,<:Real}};
                       a = 0.01, resolution = 1e-3, FOV = 20e-2)

    T = _phantom_type_from_inputs(r, a, resolution, FOV, properties)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))

    c  = T(2) * T(r + a)
    rr = T(r)
    aa = T(a)
    Lx = T(M) * (aa + rr) * T(2)
    Ly = T(N) * (aa + rr) * T(2)

    xs  = T[]
    ys  = T[]
    zs  = T[]
    T1  = T[]
    T2  = T[]
    T2s = T[]
    ρ   = T[]
    Δw  = T[]

    vials = 0
    dx_base = -Lx / T(2) + rr + aa
    dy_base = -Ly / T(2) + rr + aa

    for i in 1:M
        cx = dx_base + T(i - 1) * c
        for j in 1:N
            cy = dy_base + T(j - 1) * c
            vials += 1
            if vials > n
                break
            end

            prop = _props(properties[vials], T)
            ix, iy = _xy_ranges(grid_x, grid_y, cx, cy, rr, rr)
            r2 = rr * rr

            for ii in ix
                x = grid_x[ii]
                dx = x - cx
                dx2 = dx * dx
                for jj in iy
                    y = grid_y[jj]
                    dy = y - cy
                    if dx2 + dy * dy <= r2
                        push!(xs, x)
                        push!(ys, y)
                        push!(zs, zero(T))
                        push!(T1, prop.T1)
                        push!(T2, prop.T2)
                        push!(T2s, prop.T2s)
                        push!(ρ, prop.ρ)
                        push!(Δw, prop.Δw)
                    end
                end
            end
        end
    end

    return Phantom(x = xs, y = ys, z = zs, T1 = T1, T2 = T2, T2s = T2s, ρ = ρ, Δw = Δw)
end


# ---------------------------------------------------------------------
# Circle (2D)
# ---------------------------------------------------------------------

"""
    circle_phantom(p, r, properties::Dict{String,<:Real}; kwargs...) -> Phantom
    circle_phantom(grid_x, grid_y, p, r, properties::Dict{String,<:Real}; kwargs...) -> Phantom

Construct a 2D circular digital phantom centered at `p = (x0, y0)` with radius `r`.
"""
function circle_phantom(p, r, properties::Dict{String,<:Real};
                        resolution = 1e-3, FOV = 20e-2,
                        θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, r, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    return circle_phantom(grid_x, grid_y, p, r, properties; θx=θx, θy=θy, θz=θz)
end


function circle_phantom(grid_x::AbstractVector, grid_y::AbstractVector, p, r,
                        properties::Dict{String,<:Real};
                        θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, p, r, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 2)
    C = _CoordCollector(T)

    rr = T(r)
    ix, iy = _xy_ranges(grid_x, grid_y, pp[1], pp[2], rr, rr)
    r2 = rr * rr

    @inbounds for j in ix
        x = T(grid_x[j])
        dx = x - pp[1]
        dx2 = dx * dx
        for k in iy
            y = T(grid_y[k])
            dy = y - pp[2]
            if dx2 + dy * dy <= r2
                _append_rotated_2d!(C, dx, dy, pp, R)
            end
        end
    end

    return _build_phantom(C, pr)
end


# ---------------------------------------------------------------------
# Square (2D)
# ---------------------------------------------------------------------

"""
    square_phantom(p, l, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    square_phantom(grid_x::AbstractVector, grid_y::AbstractVector, p, l, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 2D square phantom centered at `p = (x, y)` with side length `l` (in meters).

# Arguments
- `p`: 2D center position `(x, y)` [m].
- `l`: Side length [m].
- `properties::Dict{String,<:Real}`: Tissue properties dictionary containing keys `"T1"`, `"T2"`, `"T2s"`, `"ρ"`, and `"Δw"`.

# Keywords
- `resolution = 1e-3`: Spatial sampling grid spacing [m].
- `FOV = 20e-2`: Field of view [m] for automatic grid generation.
- `θx, θy, θz = 0.0`: In-plane / out-of-plane rotation angles [rad].

# Returns
- `Phantom`: A `KomaMRIBase.Phantom` populated with spins bounded inside the square.
"""
function square_phantom(p, l, properties::Dict{String,<:Real};
                        resolution = 1e-3, FOV = 20e-2,
                        θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, l, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    return square_phantom(grid_x, grid_y, p, l, properties; θx=θx, θy=θy, θz=θz)
end


function square_phantom(grid_x::AbstractVector, grid_y::AbstractVector, p, l,
                        properties::Dict{String,<:Real};
                        θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, p, l, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 2)
    C = _CoordCollector(T)

    h = T(l) / T(2)
    ix, iy = _xy_ranges(grid_x, grid_y, pp[1], pp[2], h, h)

    @inbounds for j in ix
        x = T(grid_x[j])
        dx = x - pp[1]
        for k in iy
            y = T(grid_y[k])
            dy = y - pp[2]
            if abs(dx) <= h && abs(dy) <= h
                _append_rotated_2d!(C, dx, dy, pp, R)
            end
        end
    end

    return _build_phantom(C, pr)
end


# ---------------------------------------------------------------------
# Rectangle (2D)
# ---------------------------------------------------------------------

"""
    rectangle_phantom(p, lx, ly, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    rectangle_phantom(grid_x::AbstractVector, grid_y::AbstractVector, p, lx, ly, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 2D rectangular phantom centered at `p = (x, y)` with side lengths `lx` and `ly` [m].

# Arguments
- `p`: 2D center position `(x, y)` [m].
- `lx, ly`: Dimensions along x and y [m].
- `properties::Dict{String,<:Real}`: Physical tissue properties dictionary.

# Keywords
- `resolution = 1e-3`: Spatial grid resolution [m].
- `FOV = 20e-2`: Field of view [m].
- `θx, θy, θz = 0.0`: Spatial rotation angles [rad].

# Returns
- `Phantom`: Sampled spin phantom.
"""
function rectangle_phantom(p, lx, ly, properties::Dict{String,<:Real};
                           resolution = 1e-3, FOV = 20e-2,
                           θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, lx, ly, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    return rectangle_phantom(grid_x, grid_y, p, lx, ly, properties; θx=θx, θy=θy, θz=θz)
end


function rectangle_phantom(grid_x::AbstractVector, grid_y::AbstractVector, p, lx, ly,
                           properties::Dict{String,<:Real};
                           θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, p, lx, ly, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 2)
    C = _CoordCollector(T)

    hx, hy = T(lx)/T(2), T(ly)/T(2)
    ix, iy = _xy_ranges(grid_x, grid_y, pp[1], pp[2], hx, hy)

    @inbounds for j in ix
        x = T(grid_x[j])
        dx = x - pp[1]
        for k in iy
            y = T(grid_y[k])
            dy = y - pp[2]
            if abs(dx) <= hx && abs(dy) <= hy
                _append_rotated_2d!(C, dx, dy, pp, R)
            end
        end
    end

    return _build_phantom(C, pr)
end


# ---------------------------------------------------------------------
# Ring (2D)
# ---------------------------------------------------------------------

"""
    ring_phantom(p, r_inner, r_outer, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    ring_phantom(grid_x::AbstractVector, grid_y::AbstractVector, p, r_inner, r_outer, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 2D annulus / ring phantom centered at `p = (x, y)` with radii `r_inner` and `r_outer` [m].

# Arguments
- `p`: 2D center position `(x, y)` in meters.
- `r_inner`: Inner radius in meters.
- `r_outer`: Outer radius in meters with `r_outer >= r_inner`.
- `properties::Dict{String,<:Real}`: Physical tissue properties dictionary.

# Keywords
- `resolution = 1e-3`: Spatial grid resolution [m].
- `FOV = 20e-2`: Field of view [m].
- `θx, θy, θz = 0.0`: Rotation angles [rad].

# Returns
- `Phantom`: Ring-shaped spin phantom.
"""
function ring_phantom(p, r_inner, r_outer, properties::Dict{String,<:Real};
                      resolution = 1e-3, FOV = 20e-2,
                      θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, r_inner, r_outer, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    return ring_phantom(grid_x, grid_y, p, r_inner, r_outer, properties; θx=θx, θy=θy, θz=θz)
end


function ring_phantom(grid_x::AbstractVector, grid_y::AbstractVector, p, r_inner, r_outer,
                      properties::Dict{String,<:Real};
                      θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, p, r_inner, r_outer, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 2)
    C = _CoordCollector(T)

    ri = T(r_inner)
    ro = T(r_outer)
    ix, iy = _xy_ranges(grid_x, grid_y, pp[1], pp[2], ro, ro)
    r2i = ri * ri
    r2o = ro * ro

    @inbounds for j in ix
        x = T(grid_x[j])
        dx = x - pp[1]
        dx2 = dx * dx
        for k in iy
            y = T(grid_y[k])
            dy = y - pp[2]
            d2 = dx2 + dy * dy
            if r2i <= d2 <= r2o
                _append_rotated_2d!(C, dx, dy, pp, R)
            end
        end
    end

    return _build_phantom(C, pr)
end


# ---------------------------------------------------------------------
# Sphere (3D)
# ---------------------------------------------------------------------

"""
    sphere_phantom(p, r, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    sphere_phantom(grid_x, grid_y, grid_z, p, r, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 3D spherical phantom centered at `p = (x, y, z)` with radius `r` [m].

# Arguments
- `p`: 3D center position `(x, y, z)` [m].
- `r`: Sphere radius [m].
- `properties::Dict{String,<:Real}`: Physical tissue properties dictionary (`"T1"`, `"T2"`, `"T2s"`, `"ρ"`, `"Δw"`).

# Keywords
- `resolution = 1e-3`: 3D grid resolution [m].
- `FOV = 20e-2`: Field of view [m].
- `θx, θy, θz = 0.0`: Euler rotation angles [rad].

# Returns
- `Phantom`: 3D spherical spin distribution.
"""
function sphere_phantom(p, r, properties::Dict{String,<:Real};
                        resolution = 1e-3, FOV = 20e-2,
                        θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, r, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    grid_z = _grid(T(FOV), T(resolution))
    return sphere_phantom(grid_x, grid_y, grid_z, p, r, properties; θx=θx, θy=θy, θz=θz)
end


function sphere_phantom(grid_x::AbstractVector, grid_y::AbstractVector, grid_z::AbstractVector,
                        p, r, properties::Dict{String,<:Real};
                        θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, grid_z, p, r, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 3)
    C = _CoordCollector(T)

    rr = T(r)
    ix, iy, iz = _xyz_ranges(grid_x, grid_y, grid_z, pp[1], pp[2], pp[3], rr, rr, rr)
    r2 = rr * rr

    @inbounds for i in ix
        x = T(grid_x[i])
        dx = x - pp[1]
        dx2 = dx * dx
        for j in iy
            y = T(grid_y[j])
            dy = y - pp[2]
            dxy2 = dx2 + dy * dy
            if dxy2 <= r2
                for k in iz
                    z = T(grid_z[k])
                    dz = z - pp[3]
                    if dxy2 + dz * dz <= r2
                        _append_rotated_3d!(C, dx, dy, dz, pp, R)
                    end
                end
            end
        end
    end

    return _build_phantom(C, pr)
end


# ---------------------------------------------------------------------
# Cube / Box (3D)
# ---------------------------------------------------------------------

"""
    cube_phantom(p, l, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    cube_phantom(grid_x, grid_y, grid_z, p, l, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 3D cube phantom centered at `p = (x, y, z)` with side length `l` [m].
"""
function cube_phantom(p, l, properties::Dict{String,<:Real};
                      resolution = 1e-3, FOV = 20e-2,
                      θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, l, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    grid_z = _grid(T(FOV), T(resolution))
    return cube_phantom(grid_x, grid_y, grid_z, p, l, properties; θx=θx, θy=θy, θz=θz)
end


function cube_phantom(grid_x::AbstractVector, grid_y::AbstractVector, grid_z::AbstractVector,
                      p, l, properties::Dict{String,<:Real};
                      θx = 0.0, θy = 0.0, θz = 0.0)
    return box_phantom(grid_x, grid_y, grid_z, p, l, l, l, properties; θx=θx, θy=θy, θz=θz)
end


"""
    box_phantom(p, lx, ly, lz, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    box_phantom(grid_x, grid_y, grid_z, p, lx, ly, lz, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 3D rectangular cuboid / box phantom centered at `p = (x, y, z)` with dimensions `lx, ly, lz` [m].
"""
function box_phantom(p, lx, ly, lz, properties::Dict{String,<:Real};
                     resolution = 1e-3, FOV = 20e-2,
                     θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, lx, ly, lz, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    grid_z = _grid(T(FOV), T(resolution))
    return box_phantom(grid_x, grid_y, grid_z, p, lx, ly, lz, properties; θx=θx, θy=θy, θz=θz)
end


function box_phantom(grid_x::AbstractVector, grid_y::AbstractVector, grid_z::AbstractVector,
                     p, lx, ly, lz, properties::Dict{String,<:Real};
                     θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, grid_z, p, lx, ly, lz, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 3)
    C = _CoordCollector(T)

    hx, hy, hz = T(lx)/T(2), T(ly)/T(2), T(lz)/T(2)
    ix, iy, iz = _xyz_ranges(grid_x, grid_y, grid_z, pp[1], pp[2], pp[3], hx, hy, hz)

    @inbounds for i in ix
        x = T(grid_x[i])
        dx = x - pp[1]
        for j in iy
            y = T(grid_y[j])
            dy = y - pp[2]
            for k in iz
                z = T(grid_z[k])
                dz = z - pp[3]
                if abs(dx) <= hx && abs(dy) <= hy && abs(dz) <= hz
                    _append_rotated_3d!(C, dx, dy, dz, pp, R)
                end
            end
        end
    end

    return _build_phantom(C, pr)
end


# ---------------------------------------------------------------------
# Solid disc / Cylinder (3D)
# ---------------------------------------------------------------------

"""
    solid_disc_phantom(p, r, thickness, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    solid_disc_phantom(grid_x, grid_y, grid_z, p, r, thickness, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 3D solid disc phantom centered at `p = (x, y, z)` with radius `r` and `thickness` [m].
"""
function solid_disc_phantom(p, r, thickness, properties::Dict{String,<:Real};
                            resolution = 1e-3, FOV = 20e-2,
                            θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, r, thickness, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    grid_z = _grid(T(FOV), T(resolution))
    return solid_disc_phantom(grid_x, grid_y, grid_z, p, r, thickness, properties; θx=θx, θy=θy, θz=θz)
end


function solid_disc_phantom(grid_x::AbstractVector, grid_y::AbstractVector, grid_z::AbstractVector,
                            p, r, thickness, properties::Dict{String,<:Real};
                            θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, grid_z, p, r, thickness, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 3)
    C = _CoordCollector(T)

    rr = T(r)
    hz = T(thickness) / T(2)
    ix, iy, iz = _xyz_ranges(grid_x, grid_y, grid_z, pp[1], pp[2], pp[3], rr, rr, hz)
    r2 = rr * rr

    @inbounds for i in ix
        x = T(grid_x[i])
        dx = x - pp[1]
        dx2 = dx * dx
        for j in iy
            y = T(grid_y[j])
            dy = y - pp[2]
            if dx2 + dy * dy <= r2
                for k in iz
                    z = T(grid_z[k])
                    dz = z - pp[3]
                    if abs(dz) <= hz
                        _append_rotated_3d!(C, dx, dy, dz, pp, R)
                    end
                end
            end
        end
    end

    return _build_phantom(C, pr)
end


"""
    cylinder_phantom(p, r, h, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    cylinder_phantom(grid_x, grid_y, grid_z, p, r, h, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 3D cylindrical phantom centered at `p = (x, y, z)` with radius `r` and height `h` [m].
"""
function cylinder_phantom(p, r, h, properties::Dict{String,<:Real};
                          resolution = 1e-3, FOV = 20e-2,
                          θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, r, h, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    grid_z = _grid(T(FOV), T(resolution))
    return cylinder_phantom(grid_x, grid_y, grid_z, p, r, h, properties; θx=θx, θy=θy, θz=θz)
end


function cylinder_phantom(grid_x::AbstractVector, grid_y::AbstractVector, grid_z::AbstractVector,
                          p, r, h, properties::Dict{String,<:Real};
                          θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, grid_z, p, r, h, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 3)
    C = _CoordCollector(T)

    rr = T(r)
    hz = T(h) / T(2)
    ix, iy, iz = _xyz_ranges(grid_x, grid_y, grid_z, pp[1], pp[2], pp[3], rr, rr, hz)
    r2 = rr * rr

    @inbounds for i in ix
        x = T(grid_x[i])
        dx = x - pp[1]
        dx2 = dx * dx
        for j in iy
            y = T(grid_y[j])
            dy = y - pp[2]
            if dx2 + dy * dy <= r2
                for k in iz
                    z = T(grid_z[k])
                    dz = z - pp[3]
                    if abs(dz) <= hz
                        _append_rotated_3d!(C, dx, dy, dz, pp, R)
                    end
                end
            end
        end
    end

    return _build_phantom(C, pr)
end


# ---------------------------------------------------------------------
# Right triangle prism (3D)
# ---------------------------------------------------------------------

"""
    right_triangle_prism_phantom(p, b, h, L, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    right_triangle_prism_phantom(grid_x, grid_y, grid_z, p, b, h, L, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 3D right-triangular prism phantom with base `b`, height `h`, and length `L` [m].
"""
function right_triangle_prism_phantom(p, b, h, L, properties::Dict{String,<:Real};
                                      resolution = 1e-3, FOV = 20e-2,
                                      θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, b, h, L, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    grid_z = _grid(T(FOV), T(resolution))
    return right_triangle_prism_phantom(grid_x, grid_y, grid_z, p, b, h, L, properties; θx=θx, θy=θy, θz=θz)
end


function right_triangle_prism_phantom(grid_x::AbstractVector, grid_y::AbstractVector, grid_z::AbstractVector,
                                      p, b, h, L, properties::Dict{String,<:Real};
                                      θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, grid_z, p, b, h, L, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 3)
    C = _CoordCollector(T)

    bb = T(b)
    hh = T(h)
    LL = T(L)
    ix, iy, iz = _xyz_ranges(grid_x, grid_y, grid_z, pp[1] + bb/T(2), pp[2] + hh/T(2), pp[3], bb/T(2), hh/T(2), LL/T(2))
    hz = LL / T(2)

    @inbounds for i in ix
        x = T(grid_x[i])
        xr = x - pp[1]
        if zero(T) <= xr <= bb
            for j in iy
                y = T(grid_y[j])
                yr = y - pp[2]
                if zero(T) <= yr <= hh && yr <= hh * (one(T) - xr / bb)
                    for k in iz
                        z = T(grid_z[k])
                        dz = z - pp[3]
                        if abs(dz) <= hz
                            _append_rotated_3d!(C, xr, yr, dz, pp, R)
                        end
                    end
                end
            end
        end
    end

    return _build_phantom(C, pr)
end


# ---------------------------------------------------------------------
# Isosceles triangle prism (3D)
# ---------------------------------------------------------------------

"""
    triangle_prism_phantom(p, b, h, L, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    triangle_prism_phantom(grid_x, grid_y, grid_z, p, b, h, L, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 3D isosceles triangular prism phantom with base `b`, height `h`, and length `L` [m].
"""
function triangle_prism_phantom(p, b, h, L, properties::Dict{String,<:Real};
                                resolution = 1e-3, FOV = 20e-2,
                                θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, b, h, L, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    grid_z = _grid(T(FOV), T(resolution))
    return triangle_prism_phantom(grid_x, grid_y, grid_z, p, b, h, L, properties; θx=θx, θy=θy, θz=θz)
end


function triangle_prism_phantom(grid_x::AbstractVector, grid_y::AbstractVector, grid_z::AbstractVector,
                                p, b, h, L, properties::Dict{String,<:Real};
                                θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, grid_z, p, b, h, L, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 3)
    C = _CoordCollector(T)

    bb = T(b)
    hh = T(h)
    LL = T(L)
    ix, iy, iz = _xyz_ranges(grid_x, grid_y, grid_z, pp[1], pp[2], pp[3], bb/T(2), hh/T(2), LL/T(2))
    hz = LL / T(2)
    y0 = pp[2] - hh / T(2)

    @inbounds for i in ix
        x = T(grid_x[i])
        dx = x - pp[1]
        for j in iy
            y = T(grid_y[j])
            y_rel = y - y0
            if zero(T) <= y_rel <= hh
                t = y_rel / hh
                half_base = ((one(T) - t) * bb) / T(2)
                if abs(dx) <= half_base
                    dy = y - pp[2]
                    for k in iz
                        z = T(grid_z[k])
                        dz = z - pp[3]
                        if abs(dz) <= hz
                            _append_rotated_3d!(C, dx, dy, dz, pp, R)
                        end
                    end
                end
            end
        end
    end

    return _build_phantom(C, pr)
end


# ---------------------------------------------------------------------
# Trapezoid prism (3D)
# ---------------------------------------------------------------------

"""
    trapezoid_prism_phantom(p, l1, l2, h, properties::Dict{String,<:Real}; resolution=1e-3, FOV=20e-2, θx=0.0, θy=0.0, θz=0.0)
    trapezoid_prism_phantom(grid_x, grid_y, grid_z, p, l1, l2, h, properties::Dict{String,<:Real}; θx=0.0, θy=0.0, θz=0.0)

Construct a 3D trapezoidal prism phantom with parallel bases `l1` and `l2`, height `h`, and length spanned by `grid_z` [m].
"""
function trapezoid_prism_phantom(p, l1, l2, h, properties::Dict{String,<:Real};
                                 resolution = 1e-3, FOV = 20e-2,
                                 θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(p, l1, l2, h, properties, resolution, FOV, θx, θy, θz)
    grid_x = _grid(T(FOV), T(resolution))
    grid_y = _grid(T(FOV), T(resolution))
    grid_z = _grid(T(FOV), T(resolution))
    return trapezoid_prism_phantom(grid_x, grid_y, grid_z, p, l1, l2, h, properties; θx=θx, θy=θy, θz=θz)
end


function trapezoid_prism_phantom(grid_x::AbstractVector, grid_y::AbstractVector, grid_z::AbstractVector,
                                 p, l1, l2, h, properties::Dict{String,<:Real};
                                 θx = 0.0, θy = 0.0, θz = 0.0)
    T = _phantom_type_from_inputs(grid_x, grid_y, grid_z, p, l1, l2, h, properties, θx, θy, θz)
    pr = _props(properties, T)
    pp, R = _rot_and_point(T, p, θx, θy, θz, 3)
    C = _CoordCollector(T)

    ll1 = T(l1)
    ll2 = T(l2)
    hh  = T(h)

    Lz = T(maximum(grid_z) - minimum(grid_z))
    hz = Lz / T(2)
    halfmax = max(ll1, ll2) / T(2)
    ix, iy, iz = _xyz_ranges(grid_x, grid_y, grid_z, pp[1], pp[2], pp[3], halfmax, hh/T(2), hz)
    y0 = pp[2] - hh / T(2)

    @inbounds for i in ix
        x = T(grid_x[i])
        dx = x - pp[1]
        for j in iy
            y = T(grid_y[j])
            y_rel = y - y0
            if zero(T) <= y_rel <= hh
                t = y_rel / hh
                half_base = ((one(T) - t) * ll1 + t * ll2) / T(2)
                if abs(dx) <= half_base
                    dy = y - pp[2]
                    for k in iz
                        z = T(grid_z[k])
                        dz = z - pp[3]
                        if abs(dz) <= hz
                            _append_rotated_3d!(C, dx, dy, dz, pp, R)
                        end
                    end
                end
            end
        end
    end

    return _build_phantom(C, pr)
end