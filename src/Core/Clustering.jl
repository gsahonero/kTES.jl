export min_pairwise_distance, match_profile, cluster_phantom

@doc raw"""
    min_pairwise_distance(X::AbstractMatrix)

Computes the minimum pairwise Euclidean distance between points in a dataset.

This function calculates the smallest distance between any two points in the dataset
by constructing a KDTree and finding the nearest neighbor for each point. It is used
to determine the spatial resolution or minimum detectable separation between features
in MRI phantom simulations.

# Arguments
- `X::AbstractMatrix`: A d×N matrix where columns represent spatial points in d-dimensional space

# Returns
- `Float64`: The minimum pairwise Euclidean distance between any two points in the dataset

# Extended Help
The function constructs a KDTree from the input point set $X \\in \\mathbb{R}^{d \\times N}$ and for each point $x_i \\in X$, 
finds its nearest neighbor $x_j \\in X \\setminus \\{x_i\\}$ using the Euclidean distance metric. The minimum squared distance 
$d_{\\min}^2 = \\min_{i,j} \\|x_i - x_j\\|^2$ is computed, and the final result is the square root of this minimum distance.

This approach is particularly useful in MRI simulation contexts where the spatial resolution of the phantom needs to be 
quantified to ensure proper sampling of tissue properties and to avoid aliasing effects in the simulated signals.
"""
function min_pairwise_distance(X::AbstractMatrix)
    # X is d×N (columns are spatial points)
    tree = KDTree(X)
    N = size(X, 2)
    mind2 = Inf

    for i in 1:N
        # returns indices and squared distances
        idxs, d2s = knn(tree, X[:, i], 2, true)

        # idxs[1] = i (the point itself), idxs[2] = nearest neighbor
        d2 = d2s[2]
        d2 < mind2 && (mind2 = d2)
    end

    return sqrt(mind2)
end

@doc raw"""
    match_profile(z_range, ss_profile_xy, z_query)

Matches a spatial profile to a query z-coordinate by finding the nearest index in the z-range.

This function is used to map spatial coordinates from one domain to another, particularly in 
MRI phantom generation where tissue profiles need to be sampled at specific z-coordinates.

# Arguments
- `z_range`: A vector of z-coordinates representing the spatial range of the profile
- `ss_profile_xy`: A vector or matrix containing the spatial profile data
- `z_query`: The z-coordinate to match against the profile range

# Returns
- `typeof(ss_profile_xy)`: The profile data corresponding to the nearest z-coordinate

# Extended Help
Given a range of z-coordinates $z_{\\text{range}} = \\{z_1, z_2, \\ldots, z_N\\}$ and a corresponding profile 
$\\mathbf{p} = \\{p_1, p_2, \\ldots, p_N\\}$, this function finds the index $i$ such that $|z_{\\text{query}} - z_i|$ 
is minimized. The function then returns $p_i$ as the matched profile value.

This operation is essential in MRI simulation frameworks where tissue properties need to be interpolated 
or matched to specific spatial coordinates, particularly when generating multi-slice phantoms or 
performing z-slice specific analysis in 3D MRI simulations.
"""
function match_profile(z_range, ss_profile_xy, z_query)
    idxs = nearest_indices(z_range, z_query)
    return ss_profile_xy[idxs]
end




"""
    cluster_phantom(phantom::Phantom, ntissues::Int, [technique::ClusteringTechnique]; kwargs...) -> (clustered_phantom, assignments, centroids)

Cluster a continuous, heterogeneous `phantom` in tissue-parameter space into `ntissues` discrete tissue classes.

# Mathematical Rationale
As described in *the k-TES paper*, realistic phantoms contain continuous parameter variations that would
otherwise require simulating thousands of separate tissue basis signals. By partitioning normalized parameter
vectors \$\\theta_i = [T_{1,i}, T_{2,i}, T_{2,i}^*, \\rho_i, \\Delta\\omega_i]\$ into \$C\$ centroids via \$k\$-means,
the number of basis signals is compressed (e.g., from 48,400 combinations down to 1,000 clusters, achieving a
97.9% compression rate with <1% NRMSE and SSIM > 0.99).

# Arguments
- `phantom::Phantom`: Digital phantom object.
- `ntissues::Int`: Number of target cluster centroids (\$C\$).
- `technique::ClusteringTechnique`: Clustering backend (default: `NormalKMeans()`; alternatives: `FastKMeans()`, `LloydKMeans()`, `SubsampledKMeans()`, `ParallelKMeansKMeans()`).

# Keyword Arguments
- `features = (:T1, :T2, :ρ, :Δw)`: Tuple of phantom property fields to include in parameter clustering.
- `weights = nothing`: Optional per-point weighting vector.
- `standardize::Bool = true`: Normalize features by mean and standard deviation before clustering.
- `maxiter::Int = 300`: Maximum clustering iterations.
- `rng = Random.default_rng()`: Random number generator for centroid initialization.

# Returns
- `clustered_phantom::Phantom`: New phantom with clustered properties assigned to each point.
- `assignments::Vector{Int}`: Cluster index assignment for each phantom point.
- `centroids::NamedTuple`: Parameter values at the cluster centroids.
"""
function cluster_phantom(phantom::Phantom, ntissues::Int, technique::NormalKMeans;
    features=(:T1, :T2, :ρ, :Δw),
    weights=nothing,
    standardize=true,
    maxiter=300,
    rng=Random.default_rng()
)
    N = length(phantom.x)
    @assert ntissues ≥ 1 "ntissues must be ≥ 1"
    @assert ntissues ≤ N "ntissues must be ≤ number of phantom points"
    @assert !isempty(features) "features cannot be empty"

    # Build feature matrix: rows = features, cols = spins
    X = Matrix{Float64}(undef, length(features), N)
    for (r, f) in enumerate(features)
        vals = getproperty(phantom, f)
        @assert length(vals) == N "Field $f has incompatible length"
        X[r, :] .= Float64.(vals)
    end

    # Standardization statistics
    μ = vec(mean(X; dims=2))
    σ = vec(std(X; dims=2, corrected=false))
    σ[σ .== 0.0] .= 1.0

    Xproc = copy(X)
    if standardize
        Xproc .= (X .- μ) ./ σ
    end

    # Optional per-feature weights
    if weights !== nothing
        @assert length(weights) == length(features) "weights must match features length"
        Xproc .= Diagonal(Float64.(weights)) * Xproc
    end

    # K-means in property space
    result = kmeans(Xproc, ntissues; maxiter=maxiter, rng=rng)
    assignments = result.assignments
    Cproc = result.centers

    # Bring centroids back to original physical units
    C = copy(Cproc)
    if weights !== nothing
        w = Float64.(weights)
        for i in eachindex(w)
            @assert w[i] != 0 "weights must be nonzero"
            C[i, :] ./= w[i]
        end
    end
    if standardize
        C .= C .* σ .+ μ
    end

    # Build clustered property arrays
    new_fields = Dict{Symbol, Any}()

    # Copy all existing fields by default
    for f in fieldnames(typeof(phantom))
        new_fields[f] = getproperty(phantom, f)
    end

    # Replace clustered fields with centroid values
    for (r, f) in enumerate(features)
        oldvals = getproperty(phantom, f)
        newvals = similar(oldvals, length(oldvals))
        for i in eachindex(oldvals)
            newvals[i] = C[r, assignments[i]]
        end
        new_fields[f] = newvals
    end

    clustered = Phantom(
        name   = phantom.name * "_clustered",
        x      = new_fields[:x],
        y      = new_fields[:y],
        z      = new_fields[:z],
        ρ      = new_fields[:ρ],
        T1     = new_fields[:T1],
        T2     = new_fields[:T2],
        T2s    = new_fields[:T2s],
        Δw     = new_fields[:Δw],
        Dλ1    = new_fields[:Dλ1],
        Dλ2    = new_fields[:Dλ2],
        Dθ     = new_fields[:Dθ],
        motion = new_fields[:motion],
    )

    centroid_tuple = NamedTuple{Tuple(features)}(Tuple(eachrow(C)))

    return clustered, assignments, centroid_tuple
end

function cluster_phantom(
    phantom::Phantom,
    ntissues::Int, technique::FastKMeans;
    features = (:T1, :T2, :ρ, :Δw),
    weights = nothing,
    standardize = true,
    maxiter = 100,
    rng = Random.default_rng(),
)
    N = length(phantom.x)
    F = length(features)

    @assert ntissues ≥ 1
    @assert ntissues ≤ N
    @assert F > 0

    # feature matrix: F × N
    X = Matrix{Float64}(undef, F, N)
    for r in 1:F
        vals = getproperty(phantom, features[r])
        @inbounds @simd for i in 1:N
            X[r, i] = Float64(vals[i])
        end
    end

    μ = zeros(Float64, F)
    σ = ones(Float64, F)

    if standardize
        for r in 1:F
            xr = @view X[r, :]
            μr = mean(xr)
            σr = std(xr; corrected=false)
            σr = iszero(σr) ? 1.0 : σr
            μ[r] = μr
            σ[r] = σr
            @inbounds @simd for i in 1:N
                X[r, i] = (X[r, i] - μr) / σr
            end
        end
    end

    if weights !== nothing
        @assert length(weights) == F
        @inbounds for r in 1:F
            wr = Float64(weights[r])
            @assert wr != 0.0
            @simd for i in 1:N
                X[r, i] *= wr
            end
        end
    end

    result = kmeans(X, ntissues; maxiter=maxiter, rng=rng)
    assignments = result.assignments
    C = copy(result.centers)

    if weights !== nothing
        @inbounds for r in 1:F
            wr = Float64(weights[r])
            @simd for c in 1:ntissues
                C[r, c] /= wr
            end
        end
    end

    if standardize
        @inbounds for r in 1:F
            μr = μ[r]
            σr = σ[r]
            @simd for c in 1:ntissues
                C[r, c] = C[r, c] * σr + μr
            end
        end
    end

    T1_new = copy(phantom.T1)
    T2_new = copy(phantom.T2)
    ρ_new  = copy(phantom.ρ)
    Δw_new = copy(phantom.Δw)

    feature_to_row = Dict(features[i] => i for i in 1:F)

    if haskey(feature_to_row, :T1)
        r = feature_to_row[:T1]
        @inbounds @simd for i in 1:N
            T1_new[i] = C[r, assignments[i]]
        end
    end
    if haskey(feature_to_row, :T2)
        r = feature_to_row[:T2]
        @inbounds @simd for i in 1:N
            T2_new[i] = C[r, assignments[i]]
        end
    end
    if haskey(feature_to_row, :ρ)
        r = feature_to_row[:ρ]
        @inbounds @simd for i in 1:N
            ρ_new[i] = C[r, assignments[i]]
        end
    end
    if haskey(feature_to_row, :Δw)
        r = feature_to_row[:Δw]
        @inbounds @simd for i in 1:N
            Δw_new[i] = C[r, assignments[i]]
        end
    end

    clustered = Phantom(
        name   = phantom.name * "_clustered",
        x      = phantom.x,
        y      = phantom.y,
        z      = phantom.z,
        ρ      = ρ_new,
        T1     = T1_new,
        T2     = T2_new,
        T2s    = phantom.T2s,
        Δw     = Δw_new,
        Dλ1    = phantom.Dλ1,
        Dλ2    = phantom.Dλ2,
        Dθ     = phantom.Dθ,
        motion = phantom.motion,
    )

    centroid_tuple = NamedTuple{Tuple(features)}(Tuple(eachrow(C)))

    return clustered, assignments, centroid_tuple
end

"""
    cluster_phantom(phantom, ntissues, technique::SubsampledKMeans; ...)

Subsampled k-means with KDTree-accelerated full-data assignment (Role 3).

Workflow:
1. Build full F×N feature matrix, standardize, apply weights.
2. Draw `technique.nsub` columns without replacement.
3. Fit k-means on the subset (init=:kmpp).
4. Optionally refine centroids on full data with a few Lloyd steps.
5. Assign all N points via KDTree nearest-neighbour query on the centroids.
6. Decode centroids back to physical units and rebuild Phantom.
"""
function cluster_phantom(
    phantom::Phantom,
    ntissues::Int,
    technique::SubsampledKMeans;
    features  = (:T1, :T2, :ρ, :Δw),
    weights   = nothing,
    standardize = true,
    maxiter   = 100,
    rng       = Random.default_rng(),
)
    N = length(phantom.x)
    F = length(features)

    @assert ntissues ≥ 1           "ntissues must be ≥ 1"
    @assert ntissues ≤ N           "ntissues must be ≤ number of phantom points"
    @assert F > 0                  "features cannot be empty"
    @assert technique.nsub ≥ ntissues "nsub must be ≥ ntissues"

    # ── 1. Build full feature matrix F×N ──────────────────────────────────────
    X = Matrix{Float64}(undef, F, N)
    for r in 1:F
        vals = getproperty(phantom, features[r])
        @assert length(vals) == N "Field $(features[r]) has incompatible length"
        @inbounds @simd for i in 1:N
            X[r, i] = Float64(vals[i])
        end
    end

    # ── 2. Standardize using full-data statistics ─────────────────────────────
    μ = zeros(Float64, F)
    σ = ones(Float64, F)
    if standardize
        for r in 1:F
            xr = @view X[r, :]
            μr = mean(xr)
            σr = std(xr; corrected=false)
            σr = iszero(σr) ? 1.0 : σr
            μ[r] = μr
            σ[r] = σr
            @inbounds @simd for i in 1:N
                X[r, i] = (X[r, i] - μr) / σr
            end
        end
    end

    # ── 3. Apply feature weights ───────────────────────────────────────────────
    if weights !== nothing
        @assert length(weights) == F "weights must match features length"
        @inbounds for r in 1:F
            wr = Float64(weights[r])
            @assert wr != 0.0 "weights must be nonzero"
            @simd for i in 1:N
                X[r, i] *= wr
            end
        end
    end

    # ── 4. Subsample ───────────────────────────────────────────────────────────
    nsub_eff = min(N, technique.nsub)
    subidx   = randperm(rng, N)[1:nsub_eff]
    Xsub     = X[:, subidx]          # copy: kmeans may mutate init centers

    # ── 5. Fit k-means on subset ───────────────────────────────────────────────
    result_sub = kmeans(Xsub, ntissues; maxiter=maxiter, init=:kmpp, rng=rng)
    Cproc      = copy(result_sub.centers)   # F × ntissues, in scaled space

    # ── 6. Optional full-data Lloyd refinement ─────────────────────────────────
    if technique.refine_full
        kmeans!(X, Cproc; maxiter=technique.maxiter_refine)
        # kmeans! updates Cproc in-place
    end

    # ── 7. KDTree assignment: all N points → nearest centroid ─────────────────
    #   NearestNeighbors expects a d×k matrix for the tree and d×n for queries.
    #   This replaces the O(N×k×d) loop with O(N log k × d).
    tree        = KDTree(Cproc)
    assignments = nn(tree, X)[1]    # nn returns (indices, distances); we want indices

    # ── 8. Decode centroids back to physical units ────────────────────────────
    C = copy(Cproc)
    if weights !== nothing
        @inbounds for r in 1:F
            wr = Float64(weights[r])
            @simd for c in 1:ntissues
                C[r, c] /= wr
            end
        end
    end
    if standardize
        @inbounds for r in 1:F
            μr, σr = μ[r], σ[r]
            @simd for c in 1:ntissues
                C[r, c] = C[r, c] * σr + μr
            end
        end
    end

    # ── 9. Write centroid values back into property arrays ────────────────────
    T1_new = copy(phantom.T1)
    T2_new = copy(phantom.T2)
    ρ_new  = copy(phantom.ρ)
    Δw_new = copy(phantom.Δw)

    feature_to_row = Dict(features[i] => i for i in 1:F)

    if haskey(feature_to_row, :T1)
        r = feature_to_row[:T1]
        @inbounds @simd for i in 1:N; T1_new[i] = C[r, assignments[i]]; end
    end
    if haskey(feature_to_row, :T2)
        r = feature_to_row[:T2]
        @inbounds @simd for i in 1:N; T2_new[i] = C[r, assignments[i]]; end
    end
    if haskey(feature_to_row, :ρ)
        r = feature_to_row[:ρ]
        @inbounds @simd for i in 1:N; ρ_new[i] = C[r, assignments[i]]; end
    end
    if haskey(feature_to_row, :Δw)
        r = feature_to_row[:Δw]
        @inbounds @simd for i in 1:N; Δw_new[i] = C[r, assignments[i]]; end
    end

    # ── 10. Reconstruct Phantom ───────────────────────────────────────────────
    clustered = Phantom(
        name   = phantom.name * "_clustered",
        x      = phantom.x,
        y      = phantom.y,
        z      = phantom.z,
        ρ      = ρ_new,
        T1     = T1_new,
        T2     = T2_new,
        T2s    = phantom.T2s,
        Δw     = Δw_new,
        Dλ1    = phantom.Dλ1,
        Dλ2    = phantom.Dλ2,
        Dθ     = phantom.Dθ,
        motion = phantom.motion,
    )

    centroid_tuple = NamedTuple{Tuple(features)}(Tuple(eachrow(C)))

    return clustered, assignments, centroid_tuple
end

function cluster_phantom(
    phantom::Phantom,
    ntissues::Int,
    technique::LloydKMeans;
    features    = (:T1, :T2, :ρ, :Δw),
    weights     = nothing,
    standardize = true,
    rng         = Random.default_rng(),
)
    N = length(phantom.x)
    F = length(features)

    @assert ntissues ≥ 1       "ntissues must be ≥ 1"
    @assert ntissues ≤ N       "ntissues must be ≤ number of phantom points"
    @assert F > 0              "features cannot be empty"

    if technique.init_centers !== nothing
        @assert size(technique.init_centers, 1) == F "init_centers must have $F rows (one per feature)"
        @assert size(technique.init_centers, 2) == ntissues "init_centers must have $ntissues columns"
    end

    # ── 1. Feature matrix F×N ─────────────────────────────────────────────────
    X = Matrix{Float64}(undef, F, N)
    for r in 1:F
        vals = getproperty(phantom, features[r])
        @assert length(vals) == N "Field $(features[r]) has incompatible length"
        @inbounds @simd for i in 1:N
            X[r, i] = Float64(vals[i])
        end
    end

    # ── 2. Standardize ────────────────────────────────────────────────────────
    μ = zeros(Float64, F)
    σ = ones(Float64, F)
    if standardize
        for r in 1:F
            xr = @view X[r, :]
            μr = mean(xr)
            σr = std(xr; corrected=false)
            σr = iszero(σr) ? 1.0 : σr
            μ[r] = μr
            σ[r] = σr
            @inbounds @simd for i in 1:N
                X[r, i] = (X[r, i] - μr) / σr
            end
        end
    end

    # ── 3. Feature weights ────────────────────────────────────────────────────
    if weights !== nothing
        @assert length(weights) == F "weights must match features length"
        @inbounds for r in 1:F
            wr = Float64(weights[r])
            @assert wr != 0.0 "weights must be nonzero"
            @simd for i in 1:N
                X[r, i] *= wr
            end
        end
    end

    # ── 4. Resolve initial centroids in processed space ───────────────────────
    #   Case A: user supplied physical-unit centroids → transform to processed space.
    #   Case B: no init → seed with kmeans++ then hand to kmeans!.
    if technique.init_centers !== nothing
        # Transform user centroids from physical to processed space
        Cinit = copy(technique.init_centers)        # F × ntissues
        if standardize
            @inbounds for r in 1:F
                @simd for c in 1:ntissues
                    Cinit[r, c] = (Cinit[r, c] - μ[r]) / σ[r]
                end
            end
        end
        if weights !== nothing
            @inbounds for r in 1:F
                wr = Float64(weights[r])
                @simd for c in 1:ntissues
                    Cinit[r, c] *= wr
                end
            end
        end
    else
        # Seed via kmeans++ (one pass of kmeans with maxiter=1 gives us the init)
        Cinit = copy(kmeans(X, ntissues; maxiter=1, init=:kmpp, rng=rng).centers)
    end

    # ── 5. Bare Lloyd sweeps with kmeans! ─────────────────────────────────────
    #   kmeans! refines Cinit in-place and returns a KmeansResult.
    result = kmeans!(X, Cinit; maxiter=technique.maxiter)
    assignments = result.assignments
    C = copy(result.centers)

    # ── 6. Decode centroids to physical units ─────────────────────────────────
    if weights !== nothing
        @inbounds for r in 1:F
            wr = Float64(weights[r])
            @simd for c in 1:ntissues
                C[r, c] /= wr
            end
        end
    end
    if standardize
        @inbounds for r in 1:F
            μr = μ[r]
            σr = σ[r]
            @simd for c in 1:ntissues
                C[r, c] = C[r, c] * σr + μr
            end
        end
    end

    # ── 7. Write centroid values into property arrays ─────────────────────────
    T1_new = copy(phantom.T1)
    T2_new = copy(phantom.T2)
    ρ_new  = copy(phantom.ρ)
    Δw_new = copy(phantom.Δw)

    feature_to_row = Dict(features[i] => i for i in 1:F)

    if haskey(feature_to_row, :T1)
        r = feature_to_row[:T1]
        @inbounds @simd for i in 1:N; T1_new[i] = C[r, assignments[i]]; end
    end
    if haskey(feature_to_row, :T2)
        r = feature_to_row[:T2]
        @inbounds @simd for i in 1:N; T2_new[i] = C[r, assignments[i]]; end
    end
    if haskey(feature_to_row, :ρ)
        r = feature_to_row[:ρ]
        @inbounds @simd for i in 1:N; ρ_new[i]  = C[r, assignments[i]]; end
    end
    if haskey(feature_to_row, :Δw)
        r = feature_to_row[:Δw]
        @inbounds @simd for i in 1:N; Δw_new[i] = C[r, assignments[i]]; end
    end

    # ── 8. Reconstruct Phantom ────────────────────────────────────────────────
    clustered = Phantom(
        name   = phantom.name * "_clustered",
        x      = phantom.x,
        y      = phantom.y,
        z      = phantom.z,
        ρ      = ρ_new,
        T1     = T1_new,
        T2     = T2_new,
        T2s    = phantom.T2s,
        Δw     = Δw_new,
        Dλ1    = phantom.Dλ1,
        Dλ2    = phantom.Dλ2,
        Dθ     = phantom.Dθ,
        motion = phantom.motion,
    )

    centroid_tuple = NamedTuple{Tuple(features)}(Tuple(eachrow(C)))

    return clustered, assignments, centroid_tuple
end

function cluster_phantom(
    phantom::Phantom,
    ntissues::Int,
    technique::ParallelKMeansKMeans;
    features    = (:T1, :T2, :ρ, :Δw),
    weights     = nothing,
    standardize = true,
    rng         = Random.default_rng(),
)
    N = length(phantom.x)
    F = length(features)

    @assert ntissues ≥ 1  "ntissues must be ≥ 1"
    @assert ntissues ≤ N  "ntissues must be ≤ number of phantom points"
    @assert F > 0         "features cannot be empty"

    # ── 1. Feature matrix F×N ─────────────────────────────────────────────────
    # ParallelKMeans convention (from docs): X is d×n, i.e. features × samples.
    # We already build X as F×N, so NO transpose is needed at the call site.
    X = Matrix{Float64}(undef, F, N)
    for r in 1:F
        vals = getproperty(phantom, features[r])
        @assert length(vals) == N "Field $(features[r]) has incompatible length"
        @inbounds @simd for i in 1:N
            X[r, i] = Float64(vals[i])
        end
    end

    # ── 2. Standardize ────────────────────────────────────────────────────────
    μ = zeros(Float64, F)
    σ = ones(Float64, F)
    if standardize
        for r in 1:F
            xr = @view X[r, :]
            μr = mean(xr)
            σr = std(xr; corrected=false)
            σr = iszero(σr) ? 1.0 : σr
            μ[r] = μr
            σ[r] = σr
            @inbounds @simd for i in 1:N
                X[r, i] = (X[r, i] - μr) / σr
            end
        end
    end

    # ── 3. Weights ────────────────────────────────────────────────────────────
    if weights !== nothing
        @assert length(weights) == F "weights must match features length"
        @inbounds for r in 1:F
            wr = Float64(weights[r])
            @assert wr != 0.0 "weights must be nonzero"
            @simd for i in 1:N
                X[r, i] *= wr
            end
        end
    end

    # ── 4. ParallelKMeans fit ─────────────────────────────────────────────────
    # X is F×N (features × samples) — correct per ParallelKMeans docs.
    # result.centers  → F × ntissues  (no transpose needed)
    # result.assignments → length-N vector of 1-based labels in 1:ntissues
    result = ParallelKMeans.kmeans(
        technique.algo,
        X,
        ntissues;
        n_threads = technique.nthreads,
        max_iters = technique.maxiter,
        tol       = technique.tol,
        rng       = rng,
    )

    assignments = Vector{Int}(result.assignments)   # length N, values in 1:ntissues
    C           = copy(result.centers)              # F × ntissues

    # Sanity checks — these should never fire with the correct X orientation
    @assert size(C, 1) == F        "centers row count $(size(C,1)) ≠ F=$F"
    @assert size(C, 2) == ntissues "centers col count $(size(C,2)) ≠ ntissues=$ntissues"
    @assert minimum(assignments) ≥ 1         "assignment label below 1"
    @assert maximum(assignments) ≤ ntissues  "assignment label exceeds ntissues"

    # ── 5. Decode centroids to physical units ─────────────────────────────────
    if weights !== nothing
        @inbounds for r in 1:F
            wr = Float64(weights[r])
            @simd for c in 1:ntissues
                C[r, c] /= wr
            end
        end
    end
    if standardize
        @inbounds for r in 1:F
            μr = μ[r]
            σr = σ[r]
            @simd for c in 1:ntissues
                C[r, c] = C[r, c] * σr + μr
            end
        end
    end

    # ── 6. Assign centroid values to property arrays ──────────────────────────
    T1_new = copy(phantom.T1)
    T2_new = copy(phantom.T2)
    ρ_new  = copy(phantom.ρ)
    Δw_new = copy(phantom.Δw)

    feature_to_row = Dict(features[i] => i for i in 1:F)

    if haskey(feature_to_row, :T1)
        r = feature_to_row[:T1]
        @inbounds @simd for i in 1:N; T1_new[i] = C[r, assignments[i]]; end
    end
    if haskey(feature_to_row, :T2)
        r = feature_to_row[:T2]
        @inbounds @simd for i in 1:N; T2_new[i] = C[r, assignments[i]]; end
    end
    if haskey(feature_to_row, :ρ)
        r = feature_to_row[:ρ]
        @inbounds @simd for i in 1:N; ρ_new[i]  = C[r, assignments[i]]; end
    end
    if haskey(feature_to_row, :Δw)
        r = feature_to_row[:Δw]
        @inbounds @simd for i in 1:N; Δw_new[i] = C[r, assignments[i]]; end
    end

    # ── 7. Reconstruct Phantom ────────────────────────────────────────────────
    clustered = Phantom(
        name   = phantom.name * "_clustered",
        x      = phantom.x,
        y      = phantom.y,
        z      = phantom.z,
        ρ      = ρ_new,
        T1     = T1_new,
        T2     = T2_new,
        T2s    = phantom.T2s,
        Δw     = Δw_new,
        Dλ1    = phantom.Dλ1,
        Dλ2    = phantom.Dλ2,
        Dθ     = phantom.Dθ,
        motion = phantom.motion,
    )

    centroid_tuple = NamedTuple{Tuple(features)}(Tuple(eachrow(C)))
    return clustered, assignments, centroid_tuple
end