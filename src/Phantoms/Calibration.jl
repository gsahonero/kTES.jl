export T1MES_phantom,
       cross_field_phantom,
       default_T1MES_properties,
       default_cross_field_phantom_properties,
       ACR_phantom

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

@inline function _make_prop(T1, T2, T2s, ρ, Δw)
    return Dict{String,Real}(
        "T1"  => T1,
        "T2"  => T2,
        "T2s" => T2s,
        "ρ"   => ρ,
        "Δw"  => Δw,
    )
end

@inline function _append_cylinder_points!(
    xs::Vector{Float32},
    ys::Vector{Float32},
    zs::Vector{Float32},
    T1::Vector{Float32},
    T2::Vector{Float32},
    T2s::Vector{Float32},
    ρ::Vector{Float32},
    Δw::Vector{Float32},
    grid_x,
    grid_y,
    grid_z,
    p,
    r,
    h,
    prop::Dict{String,<:Real}
)
    pr = _props(prop)
    hz = h / 2
    ix, iy, iz = _xyz_ranges(grid_x, grid_y, grid_z, p[1], p[2], p[3], r, r, hz)
    r2 = r * r

    for i in ix
        x = grid_x[i]
        dx = x - p[1]
        dx2 = dx * dx
        for j in iy
            y = grid_y[j]
            dy = y - p[2]
            if dx2 + dy * dy <= r2
                for k in iz
                    z = grid_z[k]
                    if abs(z - p[3]) <= hz
                        push!(xs, Float32(x))
                        push!(ys, Float32(y))
                        push!(zs, Float32(z))
                        push!(T1, pr.T1)
                        push!(T2, pr.T2)
                        push!(T2s, pr.T2s)
                        push!(ρ, pr.ρ)
                        push!(Δw, pr.Δw)
                    end
                end
            end
        end
    end
    return nothing
end

@inline function _build_phantom_from_buffers(xs, ys, zs, T1, T2, T2s, ρ, Δw)
    return Phantom(
        x   = xs,
        y   = ys,
        z   = zs,
        T1  = T1,
        T2  = T2,
        T2s = T2s,
        ρ   = ρ,
        Δw  = Δw,
    )
end

# ------------------------------------------------------------
# T1MES
# ------------------------------------------------------------

"""
    default_T1MES_properties() -> Dict

Returns calibrated reference relaxation properties \$(T_1, T_2, T_2^*, \\rho)\$ for the 9 gel compartments (A–I) of the T1MES calibration phantom.
"""
function default_T1MES_properties()
    return Dict(
        "A" => _make_prop(255.0e-3, 155.0e-3, 155.0e-3, 0.99756, 0.0),
        "B" => _make_prop(1489.0e-3, 243.0e-3, 243.0e-3, 0.99627, 0.0),
        "C" => _make_prop(458.0e-3, 189.0e-3, 189.0e-3, 0.99675, 0.0),
        "D" => _make_prop(803.0e-3, 48.0e-3, 48.0e-3, 0.97786, 0.0),
        "E" => _make_prop(1333.0e-3, 50.0e-3, 50.0e-3, 0.97721, 0.0),
        "F" => _make_prop(1090.0e-3, 48.0e-3, 48.0e-3, 0.97744, 0.0),
        "G" => _make_prop(300.0e-3, 44.0e-3, 44.0e-3, 0.97826, 0.0),
        "H" => _make_prop(562.0e-3, 45.0e-3, 45.0e-3, 0.97623, 0.0),
        "I" => _make_prop(430.0e-3, 44.0e-3, 44.0e-3, 0.97694, 0.0),
    )
end

"""
    T1MES_phantom(; properties=nothing, resolution=1e-3, FOV=20e-2) -> Phantom
    T1MES_phantom(grid_x, grid_y, grid_z; properties=nothing, resolution=1e-3, FOV=20e-2) -> Phantom

Construct a 3D digital calibration phantom containing 9 cylindrical compartments corresponding to the T1MES standard.
"""
function T1MES_phantom(; properties = nothing, resolution = 1e-3, FOV = 20e-2)
    grid_x = _grid(FOV, resolution)
    grid_y = _grid(FOV, resolution)
    grid_z = _grid(FOV, resolution)
    return T1MES_phantom(grid_x, grid_y, grid_z; properties=properties, resolution=resolution, FOV=FOV)
end

function T1MES_phantom(grid_x, grid_y, grid_z; properties = nothing, resolution = 1e-3, FOV = 20e-2)
    if isnothing(properties)
        properties = default_T1MES_properties()
    end

    centers = Dict(
        "G" => [-36.5e-3,  34.1e-3, 0.0],
        "H" => [-36.5e-3,   0.0,    0.0],
        "I" => [-36.5e-3, -34.1e-3, 0.0],
        "D" => [  0.0,     34.1e-3, 0.0],
        "E" => [  0.0,      0.0,    0.0],
        "F" => [  0.0,    -34.1e-3, 0.0],
        "A" => [ 36.5e-3,  34.1e-3, 0.0],
        "B" => [ 36.5e-3,   0.0,    0.0],
        "C" => [ 36.5e-3, -34.1e-3, 0.0],
    )

    xs  = Float32[]
    ys  = Float32[]
    zs  = Float32[]
    T1  = Float32[]
    T2  = Float32[]
    T2s = Float32[]
    ρ   = Float32[]
    Δw  = Float32[]

    for key in ("G", "H", "I", "D", "E", "F", "A", "B", "C")
        _append_cylinder_points!(
            xs, ys, zs, T1, T2, T2s, ρ, Δw,
            grid_x, grid_y, grid_z,
            centers[key], 14e-3, 122e-3, properties[key]
        )
    end

    return _build_phantom_from_buffers(xs, ys, zs, T1, T2, T2s, ρ, Δw)
end

# ------------------------------------------------------------
# Cross-field
# ------------------------------------------------------------

"""
    default_cross_field_phantom_properties(field::Real) -> Dict

Calibrated cross-field relaxation properties for 4 test vials (`V1` to `V4`) at `field = 1.5` Tesla or `field = 0.55` Tesla.
"""
function default_cross_field_phantom_properties(field)
    if field == 1.5
        return Dict(
            "V1" => _make_prop(53.67e-3, 66.6e-3, 60e-3, 1.0, 0.0),
            "V2" => _make_prop(932.9e-3, 610.09e-3, 13e-3, 1.0, 0.0),
            "V3" => _make_prop(63.180e-3, 60e-3, 54e-3, 1.0, 0.0),
            "V4" => _make_prop(686.34e-3, 536.74e-3, 480e-3, 1.0, 0.0),
        )
    elseif field == 0.55
        return Dict(
            "V1" => _make_prop(59.84e-3, 52.79e-3, 52.6e-3, 1.0, 0.0),
            "V2" => _make_prop(600e-3, 680e-3, 550e-3, 1.0, 0.0),
            "V3" => _make_prop(65.78e-3, 60.05e-3, 57e-3, 1.0, 0.0),
            "V4" => _make_prop(550e-3, 530e-3, 515e-3, 1.0, 0.0),
        )
    else
        error("Unsupported field strength. Supported values are 1.5 and 0.55 Tesla.")
    end
end

"""
    cross_field_phantom(field::Real; kwargs...) -> Phantom
    cross_field_phantom(grid_x, grid_y, grid_z, field::Real; kwargs...) -> Phantom

Construct a 3D multi-field calibration phantom consisting of 4 cylindrical vials calibrated at 1.5T or 0.55T.
"""
function cross_field_phantom(field; properties = nothing, dx = 153.8e-3, dy = 68.2e-3, resolution = 1e-3, FOV = 20e-2)
    grid_x = _grid(FOV, resolution)
    grid_y = _grid(FOV, resolution)
    grid_z = _grid(FOV, resolution)
    return cross_field_phantom(grid_x, grid_y, grid_z, field; properties=properties, dx=dx, dy=dy, resolution=resolution, FOV=FOV)
end

function cross_field_phantom(grid_x, grid_y, grid_z, field; properties = nothing, dx = 153.8e-3, dy = 68.2e-3, resolution = 1e-3, FOV = 20e-2)
    if isnothing(properties)
        properties = default_cross_field_phantom_properties(field)
    end

    centers = Dict(
        "V1" => [-dx/2,  dy/2, 0.0],
        "V2" => [-dx/2, -dy/2, 0.0],
        "V3" => [ dx/2,  dy/2, 0.0],
        "V4" => [ dx/2, -dy/2, 0.0],
    )

    xs  = Float32[]
    ys  = Float32[]
    zs  = Float32[]
    T1  = Float32[]
    T2  = Float32[]
    T2s = Float32[]
    ρ   = Float32[]
    Δw  = Float32[]

    for key in ("V1", "V2", "V3", "V4")
        _append_cylinder_points!(
            xs, ys, zs, T1, T2, T2s, ρ, Δw,
            grid_x, grid_y, grid_z,
            centers[key], 13e-3, 40e-3, properties[key]
        )
    end

    return _build_phantom_from_buffers(xs, ys, zs, T1, T2, T2s, ρ, Δw)
end

# ------------------------------------------------------------
# ACR
# ------------------------------------------------------------

"""
    ACR_phantom(; resolution=0.5e-3, FOV=20e-2, z_offset=1e-2) -> Phantom

Construct a high-resolution 3D digital model of the American College of Radiology (ACR) MRI accreditation phantom.
"""
function ACR_phantom(; resolution = 0.5e-3, FOV = 20e-2, z_offset = 1e-2)
    @info "The phantom is designed to be used with a resolution of 0.5 mm. Using higher resolutions can lead to very large memory consumption and missing parts."

    S0 = _make_prop(800e-3, 0.01e-3, 0.01e-3, 0.0, 0.0)
    S1 = _make_prop(150e-3, 105e-3, 105e-3, 1.0, 0.0)
    S2 = _make_prop(90e-3, 50e-3, 50e-3, 1.0, 0.0)

    gx, gy, gz = create_grid_space(FOV, resolution)

    phantom_cylinder = cylinder_phantom(gx, gy, gz, [0.0, 0.0, 3e-2/2], 5e-2, 100e-3, S1)

    outer_cylinder = cylinder_phantom(gx, gy, gz, [0.0, -2.5e-2, -18e-3], 1.9e-2, 35e-3, S0)
    inner_cylinder = cylinder_phantom(gx, gy, gz, [0.0, -2.5e-2, -18e-3], 1.3e-2, 35e-3, S2)
    cylinder_1 = phantom_diff(outer_cylinder, inner_cylinder)
    cylinder_1 = cylinder_1 + inner_cylinder

    box_1 = box_phantom(gx, gy, gz, [0.0, 0.0, -3.6e-2/2], 90e-3, 11e-3, 3.6e-2, S0)
    box_2 = box_phantom(gx, gy, gz, [0.0, 2.75e-3, -1.5e-2], 90e-3, 5.5e-3, 3e-3, S1; θx=0, θy=5/180*π, θz=0)
    box_3 = box_phantom(gx, gy, gz, [0.0, -2.75e-3, -1.5e-2], 90e-3, 5.5e-3, 3e-3, S1; θx=0, θy=-5/180*π, θz=0)
    small_cylinder = cylinder_phantom(gx, gy, gz, [0.0, 0.0, 0.0], 7e-3/2, 90e-3, S1; θy=π/2)

    small_cylinder2 = phantom_diff(small_cylinder, box_1)
    small_cylinder = phantom_diff(small_cylinder, small_cylinder2)
    box_1 = phantom_diff(box_1, small_cylinder)
    box_1 = phantom_diff(box_1, box_2)
    box_1 = phantom_diff(box_1, box_3)
    box_1 = box_1 + box_2 + box_3 + small_cylinder

    t1 = right_triangle_prism_phantom(
        gx, gy, gz, [0.0, 3.5e-2 + 2.2e-2/2, -2.2e-2],
        2.2e-2, 2.2e-2, 0.5e-2, S0; θx=π, θy=-π/2, θz=0
    )
    t2 = right_triangle_prism_phantom(
        gx, gy, gz, [0.5e-2, 3.5e-2 + 2.2e-2/2, 0.0],
        3.5e-2, 2.2e-2, 0.5e-2, S0; θx=π, θy=π/2, θz=0
    )
    b1 = box_phantom(gx, gy, gz, [0.0, 3.5e-2, -2.9e-2], 2.2e-2, 0.5e-2, 1.4e-2, S0; θz=π/2)

    triangles = t1 + t2 + b1

    box_4 = box_phantom(gx, gy, gz, [0.0, 12.5e-3, -0.7e-2/2 - 0.7e-2], 75e-3, 20e-3, 0.7e-2, S0)

    holes = Phantom()
    d_x = (0.9e-3, 0.8e-3, 0.7e-3)
    x_base = (-25e-3, -5.0e-3, 15e-3)
    y_base = -22e-3

    for (idx, d) in enumerate(d_x)
        xb = x_base[idx]

        for j in 1:4
            yj = y_base + 2*j*d
            xjoff = j * 2.5e-4
            for i in 1:4
                if i == 4 && j == 4
                    continue
                end
                c = cylinder_phantom(
                    gx, gy, gz,
                    [xb + 2*d*i + xjoff, yj, +0.7e-2/2 + 0.7e-2],
                    d/2, 0.7e-2, S1
                )
                holes = holes + c
            end
        end

        x_b = xb + 2*3*d + 3*2.5e-4
        y_b = y_base + 2*3*d
        for j in 1:4
            yj = 2*d*j + y_b
            for i in 1:4
                c = cylinder_phantom(
                    gx, gy, gz,
                    [x_b + 2*d*i, yj + i*2.5e-4, +0.7e-2/2 + 0.7e-2],
                    d/2, 0.7e-2, S1
                )
                holes = holes + c
            end
        end
    end

    holes = rotate_phantom(holes, π, 0, 0)
    box_4 = phantom_diff(box_4, holes)
    box_4 = box_4 + holes

    phantom = box_1 + cylinder_1 + box_4 + triangles
    phantom_cylinder = phantom_diff(phantom_cylinder, phantom)
    phantom = phantom + phantom_cylinder

    phantom.z .+= Float32(z_offset)

    return phantom
end