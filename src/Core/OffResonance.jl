export T2s_Δw, gaussian_b0, linear_b0, quadratic_b0, effective_time, get_ppm, get_ΔB, plot_B0, grid_mls_linear, extract_B0_plane, simulate_off_resonance, fast_simulate_off_resonance, spherical_harmonic_B0, spherical_harmonic_Z2_ppm_roi, apply_Δw_map, interp_Δw_map

"""
    T2s_Δw(phantoms::Vector{Phantom}, T2s_method::ϵ_Truncated; T2s_params::T2sParams = T2sParams(), seed = 2025)

Computes and adds off-resonance frequency shifts to phantom tissue components based on the truncated Cauchy distribution method.

This function models the T2* relaxation effects by computing the off-resonance frequency shift `Δw` for each tissue component in the phantom ensemble. The method employs a truncated Cauchy distribution to simulate the distribution of T2* effects, where the truncation parameter `ϵ` controls the extent of the distribution tail. The frequency shifts are added to the existing `Δw` field of each phantom component, enabling accurate simulation of off-resonance effects in MRI sequences.

# Arguments
- `phantoms::Vector{Phantom}`: Vector of phantom structures containing tissue components with T2 and T2s parameters
- `T2s_method::ϵ_Truncated`: Method specification indicating the truncated Cauchy distribution approach
- `T2s_params::T2sParams`: T2* parameter structure containing the truncation parameter `ϵ` (default: T2sParams())
- `seed::Int`: Random number generator seed for reproducible results (default: 2025)

# Returns
- `Vector{Phantom}`: Updated phantom vector with added off-resonance frequency shifts in the `Δw` field
"""
function T2s_Δw(phantoms::Vector{Phantom}, T2s_method::ϵ_Truncated; T2s_params::T2sParams = T2sParams(), seed = 2025)
    N = length(phantoms[1].T2s)
    for index = 1:length(phantoms)
        γ = 1.0/phantoms[index][1].T2s[1] - 1.0/phantoms[index][1].T2[1]
        L = γ * tan((π/2) * (1 - T2s_params.ϵ))
        d_trunc = truncated(Cauchy(0.0, γ), -L, L)

        phantoms[index].Δw += rand(MersenneTwister(seed), d_trunc, N)
    end    
    return phantoms
end

"""
    T2s_Δw(phantoms::Vector{Phantom}, T2s_method::fixed_Truncated; T2s_params::T2sParams = T2sParams(), seed = 2025)

Computes and adds off-resonance frequency shifts to phantom tissue components based on the fixed truncated Cauchy distribution method.

This function models the T2* relaxation effects by computing the off-resonance frequency shift `Δw` for each tissue component in the phantom ensemble. The method employs a fixed truncated Cauchy distribution to simulate the distribution of T2* effects, where the truncation limits are specified by `T2s_params.L`. The frequency shifts are added to the existing `Δw` field of each phantom component, enabling accurate simulation of off-resonance effects in MRI sequences.

# Arguments
- `phantoms::Vector{Phantom}`: Vector of phantom structures containing tissue components with T2 and T2s parameters
- `T2s_method::fixed_Truncated`: Method specification indicating the fixed truncated Cauchy distribution approach
- `T2s_params::T2sParams`: T2* parameter structure containing the truncation limits `L` (default: T2sParams())
- `seed::Int`: Random number generator seed for reproducible results (default: 2025)

# Returns
- `Vector{Phantom}`: Updated phantom vector with added off-resonance frequency shifts in the `Δw` field
"""
function T2s_Δw(phantoms::Vector{Phantom}, T2s_method::fixed_Truncated; T2s_params::T2sParams = T2sParams(), seed = 2025)
    N = length(phantoms[1].T2s)
    for index = 1:length(phantoms)
        γ = 1.0/phantoms[index][1].T2s[1]
        d_trunc = truncated(Cauchy(0.0, γ), -T2s_params.L , T2s_params.L)

        phantoms[index].Δw += rand(MersenneTwister(seed), d_trunc, N)
    end    
    return phantoms
end

"""
    T2s_Δw(phantoms::Vector{Phantom}, T2sMethod::NoApproximation; T2s_params::T2sParams = T2sParams())

Computes and adds off-resonance frequency shifts to phantom tissue components based on the no approximation method.

This function models the T2* relaxation effects by computing the off-resonance frequency shift `Δw` for each tissue component in the phantom ensemble. The method employs no approximation, meaning the frequency shifts are not modified. This is useful for comparison purposes or when a simple model is desired.

# Arguments
- `phantoms::Vector{Phantom}`: Vector of phantom structures containing tissue components with T2 and T2s parameters
- `T2sMethod::NoApproximation`: Method specification indicating no approximation
- `T2s_params::T2sParams`: T2* parameter structure (default: T2sParams())

# Returns
- `Vector{Phantom}`: Updated phantom vector with added off-resonance frequency shifts in the `Δw` field
"""
function T2s_Δw(phantoms::Vector{Phantom}, T2sMethod::NoApproximation; T2s_params::T2sParams = T2sParams())
    # No approximation, so we do nothing
    return phantoms
end

"""
    gaussian_b0(rx::Real, ry::Real, σ::Real, Nx::Int, Ny::Int) -> Matrix{Float64}

Generate a synthetic zero-mean 2D Gaussian \$B_0\$ field inhomogeneity centered at `(rx, ry)` with spatial spread `σ`.
"""
function gaussian_b0(rx, ry, σ, Nx, Ny)
    x = repeat(collect(1:Nx), inner = Nx)
    y = repeat(collect(1:Ny), outer = Ny)
    r = zeros(Nx * Ny)
    r = 1/(2*π*σ^2) .* exp.(- ((x .-rx).^2 + (y .-ry).^2)/(2 * σ^2))
    r = reshape(r, (Nx, Ny))
    r = r .- mean(r)
    return r
end

"""
    linear_b0(fx::Real, fy::Real, Nx::Int, Ny::Int) -> Matrix{Float64}

Generate a linear 2D \$B_0\$ gradient field ranging across `[-fx/2, fx/2]` and `[-fy/2, fy/2]`.
"""
function linear_b0(fx, fy, Nx, Ny)
    b0 = zeros(Nx, Ny)
    base_freq_x = LinRange(-fx/2, fx/2, Nx)
    base_freq_y = LinRange(-fy/2, fy/2, Ny)
    for i=1:Nx
        for j=1:Ny
            b0[i, j] = base_freq_x[i] + base_freq_y[j]
        end
    end
    return b0
end

"""
    quadratic_b0(fmax_Hz::Real, Nx::Int, Ny::Int; FOVx=1.0, FOVy=1.0) -> Matrix{Float64}

Generate a 2D quadratic \$B_0\$ parabolic field distribution with peak frequency `fmax_Hz` (in rad/s).
"""
function quadratic_b0(fmax_Hz, Nx, Ny; FOVx=1.0, FOVy=1.0)
    # voxel sizes
    Δx = FOVx / Nx
    Δy = FOVy / Ny

    # centered coordinates
    xs = ((0:Nx-1) .- Nx/2) .* Δx
    ys = ((0:Ny-1) .- Ny/2) .* Δy

    # maximum radius (to normalize the parabola)
    rmax = sqrt((maximum(abs.(xs)))^2 + (maximum(abs.(ys)))^2)

    Δω = zeros(Float64, Nx, Ny)
    for ix in 1:Nx, iy in 1:Ny
        r = sqrt(xs[ix]^2 + ys[iy]^2)
        Δf = fmax_Hz * (r^2 / rmax^2)
        Δω[ix,iy] = 2π * Δf   # convert Hz → rad/s
    end

    return Δω
end

"""
    effective_time(t_samples::AbstractVector{<:Real}, t_refocus::AbstractVector{<:Real}) -> Vector{Float64}

Compute the accumulated effective phase evolution time across a sequence with refocusing pulses (accounting for 180° phase flips).
"""
function effective_time(t_samples::AbstractVector{<:Real},
                        t_refocus::AbstractVector{<:Real})
    t_eff = zeros(Float64, length(t_samples))
    sign  = 1.0
    acc   = 0.0          # accumulated signed time
    last_t = 0.0
    r_idx = 1

    for (i, t) in pairs(t_samples)
        # integrate up to each refocusing pulse before this sample
        while r_idx ≤ length(t_refocus) && t_refocus[r_idx] ≤ t
            dt = t_refocus[r_idx] - last_t
            acc   += sign * dt
            last_t = t_refocus[r_idx]
            sign  *= -1          # flip sign at each 180°
            r_idx += 1
        end

        # integrate the remaining piece up to t
        dt = t - last_t
        t_eff[i] = acc + sign * dt
    end

    return t_eff
end

"""
    get_ppm(B0_ref::Real, B0_map::AbstractArray) -> Float64

Calculate parts-per-million (ppm) peak-to-peak field inhomogeneity of `B0_map` relative to `B0_ref`.
"""
function get_ppm(B0_ref, B0_map)
    ΔB0 = (maximum(B0_map) - minimum(B0_map))/(2*π*γ)
    return 1e6*ΔB0/B0_ref
end

"""
    get_ΔB(ppm::Real, B0_ref::Real) -> Float64

Convert parts-per-million (ppm) relative deviation at field `B0_ref` to frequency shift in rad/s.
"""
function get_ΔB(ppm, B0_ref)
    return (ppm/1e6*B0_ref)*γ
end

"""
    plot_B0(xyz::AbstractMatrix, B0_map::AbstractVector)

Generate an interactive 3D scatter plot of the spatial \$B_0\$ field distribution.
"""
function plot_B0(xyz, B0_map)
    
    return Plot(
        scatter3d(
            x = xyz[:, 1],
            y = xyz[:, 2],
            z = xyz[:, 3],
            mode = "markers",
            marker = attr(
                size = 4,
                color = B0_map,        # your 4th value column
                colorscale = "Viridis",
                showscale = true
            )
        )
    )
end

function grid_mls_linear(xyz, vals, xs, ys, zs; K=20, weighted=true)

    N = size(xyz,1)
    Nx, Ny, Nz = length(xs), length(ys), length(zs)
    Bgrid = zeros(Float64, Nx, Ny, Nz)

    # Precompute for speed
    xyzT = xyz'

    @inbounds for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz
        x0 = xs[ix]; y0 = ys[iy]; z0 = zs[iz]

        # === 1) Compute squared distances ===
        d2 = (xyzT[1,:] .- x0).^2 .+
             (xyzT[2,:] .- y0).^2 .+
             (xyzT[3,:] .- z0).^2

        # === 2) Select K nearest neighbors ===
        idx = partialsortperm(d2, 1:K)

        # gather neighborhood
        X = xyz[idx, :]
        y = vals[idx]

        # center coordinates relative to grid point
        Xc = X .- [x0 y0 z0]

        # === 3) Build design matrix for local plane ===
        # columns = [1, dx, dy, dz]
        A = hcat(ones(K), Xc)

        # === 4) Weighted or unweighted LS ===
        if weighted
            w = 1.0 ./ (sqrt.(d2[idx]) .+ 1e-6)
            W = Diagonal(w)
            coeff = pinv(A' * W * A) * (A' * W * y)
        else
            coeff = A \ y
        end

        # === 5) Evaluate at (0,0,0) → intercept ===
        Bgrid[ix,iy,iz] = coeff[1]
    end

    return Bgrid
end

function extract_B0_plane(B0, plane_1, plane_2, fixed_point, axis)

    B0_plane = zeros(length(plane_1), length(plane_2))
    missing_counter = 0
    duplicated_counter = 0
    for (i, pi) in enumerate(plane_1)
        for (j, qi) in enumerate(plane_2)
            if B0_plane[i,j] != 0
                duplicated_counter += 1
                @error("at $zi, $yi, we already have a value")            
            else
                try 
                    q = B0[B0[:, axis[1]] .== fixed_point .&& B0[:, axis[2]] .== pi .&& B0[:,axis[3]] .== qi, :][:, 4][1]
                    B0_plane[i, j] = q
                catch e
                    println(" Values B0 at axis[2]=$pi and at axis[3]=$qi do not exist in B0 map.")
                    missing_counter += 1
                end            
            end        
        end
    end
    return B0_plane, missing_counter, duplicated_counter

end

"""
    simulate_off_resonance(m::AbstractMatrix{<:Complex}, Δω::AbstractMatrix{<:Real}, t_eff::AbstractVector{<:Real}) -> Vector{ComplexF64}
    simulate_off_resonance(seq::Sequence, m::AbstractMatrix{<:Complex}, Δω::AbstractMatrix{<:Real}) -> Vector{ComplexF64}

Simulate the 2D Cartesian k-space signal under an arbitrary off-resonance field \$\\Delta\\omega(\\mathbf{x})\$
via discrete evaluation across all spatial locations and readout time points.
"""
function simulate_off_resonance(m::AbstractMatrix{<:Complex},
                                     Δω::AbstractMatrix{<:Real}, t_eff::AbstractVector{<:Real})

    Nx, Ny = size(m)
    Ns = Nx * Ny
    @assert size(Δω) == (Nx, Ny)
    @assert length(t_eff) == Ns

    s = zeros(ComplexF64, Ns)

    # Precompute centered voxel indices: n_cx, n_cy ∈ [-Nx/2 .. Nx/2-1], etc.
    # (This matches the "centered" convention underlying fftshift.)
    cx = collect(0:Nx-1) .- floor(Nx/2)  # length Nx
    cy = collect(0:Ny-1) .- floor(Ny/2)  # length Ny

    for j in 1:Ns
        # map linear index j to *matrix* indices (px0, py0) in 0-based form
        px0 = (j - 1) % Nx       # column index 0..Nx-1
        py0 = (j - 1) ÷ Nx       # row index 0..Ny-1

        # centered frequency indices (like after fftshift)
        kx_c = px0 - floor(Nx/2)   # ∈ [-Nx/2 .. Nx/2-1]
        ky_c = py0 - floor(Ny/2)   # ∈ [-Ny/2 .. Ny/2-1]

        tj = t_eff[j]
        acc = 0.0 + 0.0im

        for ix in 1:Nx
            n_cx = cx[ix]         # centered spatial index
            for iy in 1:Ny
                n_cy = cy[iy]

                ρ = m[ix, iy]
                ω = Δω[ix, iy]

                # centered DFT kernel
                phase_k  = -2π * (kx_c * n_cx / Nx + ky_c * n_cy / Ny)
                phase_B0 = -ω * tj

                acc += ρ * cis(phase_k + phase_B0)
            end
        end

        s[j] = acc
    end

    return s
end

function simulate_off_resonance(seq::Sequence, m::AbstractMatrix{<:Complex},
                                     Δω::AbstractMatrix{<:Real})
    t_samples = get_adc_sampling_times(seq)
    t_samples = t_samples .- minimum(t_samples)
    t_refocusing = t_samples[seq.DEF["parameters"].Nx÷2:seq.DEF["parameters"].Nx:end]
    t_eff = effective_time(t_samples, t_refocusing)
    return simulate_off_resonance(m, Δω, t_eff)
end

"""
    fast_simulate_off_resonance(m::AbstractMatrix{<:Complex}, Δω::AbstractMatrix{<:Real}, t_eff::AbstractVector{<:Real}) -> Vector{ComplexF64}

Fast off-resonance simulation utilizing pre-planned FFTs and incremental phase modulation.
"""
function fast_simulate_off_resonance(m::AbstractMatrix{<:Complex},
                                      Δω::AbstractMatrix{<:Real},
                                      t_eff::AbstractVector{<:Real})

    Nx, Ny = size(m)
    Ns     = Nx * Ny
    @assert size(Δω) == (Nx, Ny)
    @assert length(t_eff) == Ns

    # ── Precompute spatial shifts ONCE ───────────────────────────────────────
    # ifftshift on both m and Δω: centers spatial indices to match reference loop.
    # fftshift on the FFT output: centers k-space indices.
    # Both arrays must be shifted consistently — Δω rides on m in the modulation.
    m_s  = ifftshift(complex(m))      # ComplexF64, layout shifted
    Δω_s = ifftshift(Δω)              # Float64, same layout shift

    # ── FFTW plan: reuse across all time points ──────────────────────────────
    work    = copy(m_s)               # working buffer, same size/type
    plan_ip = plan_fft!(work, flags=FFTW.MEASURE)

    # ── Sort unique t values; build incremental phasors ──────────────────────
    t_sorted  = sort(unique(t_eff))
    t_to_idx  = Dict(t => i for (i, t) in enumerate(t_sorted))
    Δts       = diff(t_sorted)

    # Incremental phasors: exp(-i·Δω·Δt) for each step between consecutive t values.
    # Precomputed once — avoids recomputing cis() over full Nx×Ny for every t.
    inc_phasors = [cis.(.- Δω_s .* Δt) for Δt in Δts]

    # ── Main loop ────────────────────────────────────────────────────────────
    fft_cache = Vector{Matrix{ComplexF64}}(undef, length(t_sorted))

    # Initial phase map at first (smallest) t value
    phase_map = cis.(.- Δω_s .* t_sorted[1])

    for i in eachindex(t_sorted)
        # Modulate into work buffer in-place, then FFT in-place
        @. work = m_s * phase_map
        plan_ip * work                           # in-place FFT, overwrites work
        fft_cache[i] = fftshift(work)            # centered k-space — allocates once per t
        # Advance phase map multiplicatively (no new cis() calls over full grid)
        if i < length(t_sorted)
            phase_map .*= inc_phasors[i]
        end
    end

    # ── Sample k-space ───────────────────────────────────────────────────────
    s = zeros(ComplexF64, Ns)
    for j in 1:Ns
        px = (j - 1) % Nx + 1    # 1-based, matches fftshift output layout
        py = (j - 1) ÷ Nx + 1
        s[j] = fft_cache[t_to_idx[t_eff[j]]][px, py]
    end

    return s
end

function spherical_harmonic_B0(Nx::Int, Ny::Int;
                               FOVx::Float64=0.22,
                               FOVy::Float64=0.22,
                               bore_radius::Float64=0.25,
                               z0::Float64=0.0,
                               B0_const::Float64=0.0,
                               cZ::Float64=0.0,
                               cX::Float64=0.0,
                               cY::Float64=0.0,
                               cZ2::Float64=0.0,
                               cX2Y2::Float64=0.0,
                               cXY::Float64=0.0,
                               cXZ::Float64=0.0,
                               cYZ::Float64=0.0)

    # Proton gyromagnetic ratio [rad/s/T]
    γ_b = 2π * γ

    # Voxel sizes
    Δx = FOVx / Nx
    Δy = FOVy / Ny

    # Centered coordinates for voxel centers
    xs = ((0:Nx-1) .- Nx/2) .* Δx
    ys = ((0:Ny-1) .- Ny/2) .* Δy
    z  = z0

    Δω = zeros(Float64, Nx, Ny)

    for ix in 1:Nx, iy in 1:Ny
        x = xs[ix]
        y = ys[iy]
        r2 = x^2 + y^2 + z^2
        r  = sqrt(r2)

        if r <= bore_radius
            # Solid harmonic basis terms
            Z   = z
            X   = x
            Y   = y
            Z2  = (3z^2 - r2) / 2
            X2Y2 = x^2 - y^2
            XY   = x*y
            XZ   = x*z
            YZ   = y*z

            ΔB = B0_const +
                 cZ   * Z   +
                 cX   * X   +
                 cY   * Y   +
                 cZ2  * Z2  +
                 cX2Y2 * X2Y2 +
                 cXY  * XY  +
                 cXZ  * XZ  +
                 cYZ  * YZ

            Δω[ix,iy] = γ_b * ΔB  # rad/s
        else
            Δω[ix,iy] = 0.0
        end
    end

    return Δω
end

function spherical_harmonic_Z2_ppm_roi(Nx::Int, Ny::Int;
                                       FOVx::Float64=0.22,
                                       FOVy::Float64=0.22,
                                       bore_radius::Float64=0.25,
                                       roi_radius::Float64=0.10,
                                       z0::Float64=0.0,
                                       B0_nominal::Float64=1.5,
                                       peak_ppm::Float64=1.0)


    ΔB_peak = peak_ppm * 1e-6 * B0_nominal   # Tesla

    Δx = FOVx / Nx
    Δy = FOVy / Ny

    xs = ((0:Nx-1) .- Nx/2) .* Δx
    ys = ((0:Ny-1) .- Ny/2) .* Δy
    z  = z0

    Z2      = zeros(Float64, Nx, Ny)
    maxabsH = 0.0

    # First pass: build Z2, find max inside ROI (and inside bore)
    for ix in 1:Nx, iy in 1:Ny
        x = xs[ix]
        y = ys[iy]
        r2 = x^2 + y^2 + z^2
        r  = sqrt(r2)

        if r <= bore_radius
            H = (3z^2 - r2) / 2  # Z2
            Z2[ix,iy] = H
            if r <= roi_radius
                maxabsH = max(maxabsH, abs(H))
            end
        else
            Z2[ix,iy] = 0.0
        end
    end

    if maxabsH == 0.0
        return zeros(Float64, Nx, Ny)
    end

    # Scale so that inside ROI we get the desired peak ppm
    cZ2 = ΔB_peak / maxabsH

    Δω = γ .* (cZ2 .* Z2)
    return Δω
end




function spherical_harmonic_Z2_ppm(Nx::Int, Ny::Int;
                                   FOVx::Float64=0.22,
                                   FOVy::Float64=0.22,
                                   bore_radius::Float64=0.25,
                                   z0::Float64=0.0,
                                   B0_nominal::Float64=1.5,
                                   peak_ppm::Float64=2.0)


    # Convert desired peak ppm → Tesla
    ΔB_peak = peak_ppm * 1e-6 * B0_nominal   # Tesla

    # Voxel sizes
    Δx = FOVx / Nx
    Δy = FOVy / Ny

    # Centered coordinates
    xs = ((0:Nx-1) .- Nx/2) .* Δx
    ys = ((0:Ny-1) .- Ny/2) .* Δy
    z  = z0

    # First build Z2 basis and find its max abs inside the bore
    Z2 = zeros(Float64, Nx, Ny)
    maxabs_H = 0.0

    for ix in 1:Nx, iy in 1:Ny
        x = xs[ix]
        y = ys[iy]
        r2 = x^2 + y^2 + z^2
        r  = sqrt(r2)

        if r <= bore_radius
            H = (3z^2 - r2) / 2       # Z2 solid harmonic
            Z2[ix,iy] = H
            maxabs_H = max(maxabs_H, abs(H))
        else
            Z2[ix,iy] = 0.0
        end
    end

    # Avoid division by zero (e.g. degenerate case)
    if maxabs_H == 0.0
        return zeros(Float64, Nx, Ny)
    end

    # Scale so that peak |ΔB| = ΔB_peak
    cZ2 = ΔB_peak / maxabs_H      # Tesla / (units of H)

    # Build Δω = γ ΔB
    Δω = γ .* (cZ2 .* Z2)         # rad/s
    return Δω
end

function spherical_harmonic_Z2_ppm_DSV(
    Nx::Int, Ny::Int;
    FOVx::Float64=0.22,
    FOVy::Float64=0.22,
    bore_radius::Float64=0.25,
    DSV_radius::Float64=0.10,
    transition_width::Float64=0.01,
    B0_nominal::Float64=1.5,
    peak_ppm_outside::Float64=2.0,
)


    # PPM objetivo fuera del DSV → Tesla
    ΔB_peak_out = peak_ppm_outside * 1e-6 * B0_nominal

    # Tamaños de voxel
    Δx = FOVx / Nx
    Δy = FOVy / Ny

    # Coordenadas centradas
    xs = ((0:Nx-1) .- Nx/2) .* Δx
    ys = ((0:Ny-1) .- Ny/2) .* Δy
    z  = 0.0

    Z2 = zeros(Float64, Nx, Ny)
    w  = zeros(Float64, Nx, Ny)

    # Función de transición suave de 0→1 en [0,1]
    smoothstep(u) = u <= 0 ? 0.0 : u >= 1 ? 1.0 : (3u^2 - 2u^3)

    # Primero construimos el Z2 y el peso radial w(r)
    maxabs_weighted_H_out = 0.0

    r_start = DSV_radius
    r_end   = DSV_radius + transition_width

    for ix in 1:Nx, iy in 1:Ny
        x = xs[ix]
        y = ys[iy]
        r2 = x^2 + y^2 + z^2
        r  = sqrt(r2)

        # Sólo consideramos dentro del bore
        if r <= bore_radius
            # Z2 sólido: (3z^2 - r^2)/2, aquí z=0 → Z2 = -r^2/2
            H = (3z^2 - r2) / 2
            Z2[ix,iy] = H

            # Peso radial: 0 dentro del DSV, 1 fuera de la capa de transición
            if r <= r_start
                w[ix,iy] = 0.0
            elseif r >= r_end
                w[ix,iy] = 1.0
            else
                u = (r - r_start) / (r_end - r_start)
                w[ix,iy] = smoothstep(u)
            end

            # Para escalar: sólo miramos la parte "outside" (donde w>0)
            weighted_H = w[ix,iy] * H
            maxabs_weighted_H_out = max(maxabs_weighted_H_out, abs(weighted_H))
        else
            Z2[ix,iy] = 0.0
            w[ix,iy]  = 0.0
        end
    end

    if maxabs_weighted_H_out == 0.0
        return zeros(Float64, Nx, Ny)
    end

    # Escala tal que el máximo |w*H| dentro del bore ≈ ΔB_peak_out
    cZ2 = ΔB_peak_out / maxabs_weighted_H_out

    Δω = zeros(Float64, Nx, Ny)
    for ix in 1:Nx, iy in 1:Ny
        ΔB = cZ2 * w[ix,iy] * Z2[ix,iy]  # Tesla
        Δω[ix,iy] = γ * ΔB               # rad/s
    end

    return Δω
end

"""
    interp_Δw_map(Δw_map::AbstractMatrix{<:Real}, FOVx::Float64, FOVy::Float64, xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}) -> Vector{Float64}

Bilinear interpolation of a regular 2D off-resonance field map `Δw_map` at scattered spatial coordinates `(xs, ys)`.
"""
function interp_Δw_map(Δw_map::AbstractMatrix{<:Real},
                            FOVx::Float64, FOVy::Float64,
                            xs::AbstractVector{<:Real},
                            ys::AbstractVector{<:Real})

    
    Nx, Ny   = size(Δw_map)
    inv_Δx   = Nx / FOVx          # reciprocal multiply — hoisted out of loop
    inv_Δy   = Ny / FOVy
    x_start  = -FOVx / 2
    y_start  = -FOVy / 2
    Nx_m2    = Nx - 2
    Ny_m2    = Ny - 2
    N        = length(xs)

    out = Vector{Float64}(undef, N)

    Threads.@threads :static for k in 1:N
        @inbounds begin
            u  = (xs[k] - x_start) * inv_Δx
            v  = (ys[k] - y_start) * inv_Δy

            i0 = clamp(unsafe_trunc(Int, u), 0, Nx_m2)   # ← see note below
            j0 = clamp(unsafe_trunc(Int, v), 0, Ny_m2)

            αx = u - i0          # NOTE: frac relative to clamped index, not floor(u)
            αy = v - j0          # this is correct when u ∈ [0, Nx-1]

            βx = 1.0 - αx
            βy = 1.0 - αy

            # All four reads happen with live αx,αy,βx,βy in registers
            out[k] = βx*βy * Δw_map[i0+1, j0+1] +
                    αx*βy * Δw_map[i0+2, j0+1] +
                    βx*αy * Δw_map[i0+1, j0+2] +
                    αx*αy * Δw_map[i0+2, j0+2]
        end
    end
    return out
end

"""
    apply_Δw_map(phantom::Phantom, Δw_map::AbstractMatrix{<:Real}, FOVx::Float64, FOVy::Float64) -> Phantom

Interpolate and apply a macroscopic 2D field map `Δw_map` onto the isochromats of `phantom`,
returning a new `Phantom` with combined microscopic and macroscopic off-resonance frequencies.
"""
function apply_Δw_map(phantom::Phantom,
                           Δw_map::AbstractMatrix{<:Real},
                           FOVx::Float64, FOVy::Float64)

    Δw_macro = interp_Δw_map(Δw_map, FOVx, FOVy, phantom.x, phantom.y)

    # Deepcopy: compiler cannot alias the new phantom's Δw with the input
    ph_new      = deepcopy(phantom)
    ph_new.Δw .+= Δw_macro   # add macroscopic on top of existing microscopic T2*

    return ph_new
end