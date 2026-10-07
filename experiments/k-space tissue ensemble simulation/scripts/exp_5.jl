#=
Experiment 4 of the k-TES paper.
Clustering effects on a heterogeneous phantom
Author: Guillermo Sahonero Alvarez
Affiliations: Pontificia Universidad Católica de Chile (IIBM, CIB) / iHEALTH
=#

using kTES
using CUDA
using Images # has SSIM
using BenchmarkTools
using PlotlyJS
using Statistics
using Serialization
using MAT

## Metrics
function NRMSE(img_x, img_target)
    return norm(img_x - img_target) / norm(img_target)
end
function MAD(img_x, img_target)
    return mean(abs.(img_target - img_x))/maximum(abs.(img_target))
end
## Preparing scanner
sys = Scanner()
FOVx, FOVy, slice_thickness = 0.25, 0.25, 1e-3
Nx, Ny = 128, 128
Δ = FOVx/Nx

## SPGR Sequence

RF_phases = zeros(ComplexF64, Nx)
create_RF_phases!(RF_phases, QuadraticPhases(), 117 / 180 * π)
spgr_params = SeqParams(Dict("α" => 15, "TE" => 20e-3, "TR" => 200e-3,  "Nx" => Nx, "Ny" => Ny, "FOVx" => FOVx, "FOVy" => FOVy, "t_ADC" => Nx*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "num_preparation_pulses" => 100, "preparation_technique" => Pulses(), "force_kcenter" => true, "gradient_spoilers" => [8π, 8π, 8π], "RF_spoiler_seed" => 117/180*π, "RF_phases" => RF_phases, "phases_technique" => QuadraticPhases(), "pulse_type" => Soft(), "RF_duration" => 2.3e-3, "balancing_gradients" => [false, false, false], "slice_thickness" => slice_thickness, "OF" => 1, "NSA" => 1, "seq_type" => SPGR(), "disable_slice_selection" => false, "phases_technique" => QuadraticPhases()))

spgr_seq = create_2D_seq(spgr_params, SPGR(), sys)

## Loading phantom

#data = matread("/data/subject_1.mat")
data = matread("C:\\Users\\gswr\\OneDrive\\UC-Research\\optimizing-Low-Field-MRI-sequences-for-high-field-contrast-equivalent-images\\kTES.jl\\experiments\\k-space tissue ensemble simulation\\data\\subject_1.mat")

ground_truth_phantom = phantom_from_image_maps(data["T1_map"][:, :, 120]./1000, data["T2_map"][:, :, 120]./1000, data["PD_map"][:, :, 120], 0.25, 0.25; oversampling_factor = 10)

phantom = phantom_from_image_maps(data["T1_map"][:, :, 120]./1000, data["T2_map"][:, :, 120]./1000, data["PD_map"][:, :, 120], 0.25, 0.25; oversampling_factor = 1)

# Ground truth
sim_params = Dict("return_type" => "mat", "method" => Bloch())
N_replicas = 10_000

t_koma0 = time_ns()
spgr_koma = simulate(ground_truth_phantom, spgr_seq, sys; sim_params = sim_params)
s_koma = reshape(spgr_koma[:], (spgr_seq.DEF["parameters"].Nx, spgr_seq.DEF["parameters"].Ny))
i_koma = ifftshift(ifft(fftshift(s_koma)))
abs_koma = abs.(i_koma)
t_koma1 = time_ns()
t_koma = (t_koma1 - t_koma0) / 1e9

phantom = phantom_from_image_maps(data["T1_map"][:, :, 120]./1000, data["T2_map"][:, :, 120]./1000, data["PD_map"][:, :, 120], 0.25, 0.25; oversampling_factor = 1)

display(plot_image(log.(abs.(s_koma)), title = "|S(t)| - KomaMRI"))
display(plot_image(abs.(i_koma)./length(ground_truth_phantom.x), title = "|I(x)| - KomaMRI"))


# Iterations through different number of clusters

tech = ParallelKMeansKMeans(
    algo     = Hamerly(),
    nthreads = Threads.nthreads(),
    maxiter  = 300,
    tol      = 1e-6,
)

vec_clusters = [5, 50, 100, 150, 300, 600, 1000, 5000, 10000, 20000]

NRMSEs = zeros(length(vec_clusters))
SSIMs = zeros(length(vec_clusters))
MADs = zeros(length(vec_clusters))
times_ktes = zeros(length(vec_clusters))
spgr_data = []

for (index, n_cluster) in enumerate(vec_clusters)
    @info "Processing with $n_cluster clusters"
    t0 = time_ns()
    clustered_phantom, assignments, centroids = cluster_phantom(
        phantom, n_cluster, tech;
        features    = (:T1, :T2, :ρ),
        standardize = true,
    )

    tissues, _, tissue_maps = phantom_descriptor(clustered_phantom)
    plan = build_tissue_support_plan(clustered_phantom, tissue_maps, spgr_seq; use_float32=true, any_rf = true)

    A=create_ktes_support(
        clustered_phantom, tissue_maps, spgr_seq;
        plan = plan
    )


    OSP_phantom = create_support_phantoms(clustered_phantom, spgr_seq, Normal(), sys; T2s_params = T2sParams(method=NoApproximation()), replicas = N_replicas, σ=1/20.0)

    acq_params = default_acq_params()
    acq_params.noise_technique = ReducedNoise()
    spgr_ktes = simulate_acquisition(OSP_phantom, spgr_seq, acq_params, sys, sim_params, A, N_replicas, clustered_phantom, tissue_maps, plan)
    t1 = time_ns()
    times_ktes[index] = (t1 - t0) / 1e9
    push!(spgr_data, spgr_ktes)

    # Comparison results using spgr
    NRMSEs[index] = round(NRMSE(spgr_ktes[3]./length(phantom.x), i_koma./length(ground_truth_phantom.x))*100, digits=4)
    SSIMs[index] = round(assess_ssim(abs.(spgr_ktes[3])./maximum(abs.(spgr_ktes[3])), abs_koma./maximum(abs_koma)), digits = 4)
    MADs[index] = round(MAD(spgr_ktes[2]./length(phantom.x), s_koma./length(ground_truth_phantom.x))*100, digits = 4)

end


## Two plots visualization
function clustering_performance_plots(
    clusters,
    nrmse,
    mad,
    ssim;
    logx::Bool = false,
    font_size::Int = 15,
)

    ############################
    # Plot 1: NRMSE
    ############################

    nrmse_trace = scatter(
        x = clusters,
        y = nrmse,
        name = "NRMSE (%)",
        mode = "lines+markers",
        marker = attr(
            symbol = "circle",
            size = 8,
            color = "black",
        ),
        line = attr(
            color = "black",
            width = 2,
        ),
    )

    layout_nrmse = Layout(

        width = 700,
        height = 450,

        template = "simple_white",

        paper_bgcolor = "white",
        plot_bgcolor = "white",

        font = attr(size = font_size),

        margin = attr(
            l = 80,
            r = 30,
            t = 20,
            b = 70,
        ),

        xaxis = attr(
            title = "Number of clusters",
            type = logx ? "log" : "linear",
            showline = true,
            linewidth = 1.5,
            linecolor = "black",
            mirror = true,
            ticks = "outside",
            tickwidth = 1.5,
            ticklen = 6,
            showgrid = true,
            zeroline = false
        ),

        yaxis = attr(
            title = "NRMSE (%)",
            range = [0, 1.05 * maximum(nrmse)],
            showline = true,
            linewidth = 1.5,
            linecolor = "black",
            mirror = true,
            ticks = "outside",
            tickwidth = 1.5,
            ticklen = 6,
            showgrid = true,
            zeroline = false,
        ),

        legend = attr(
            x = 0.02,
            y = 0.98,
            bgcolor = "rgba(255,255,255,0.8)",
            borderwidth = 0,
        ),
    )

    nrmse_plot = Plot([nrmse_trace], layout_nrmse)

    ############################
    # Plot 2: MAD + (1 - SSIM)
    ############################

    mad_trace = scatter(
        x = clusters,
        y = mad,
        name = "MAD",
        mode = "lines+markers",
        marker = attr(
            symbol = "square",
            size = 8,
            color = "#D55E00",
        ),
        line = attr(
            color = "#D55E00",
            width = 2,
        ),
    )

    ssim_trace = scatter(
        x = clusters,
        y = 1 .- ssim,
        name = "1 − SSIM",
        mode = "lines+markers",
        marker = attr(
            symbol = "diamond",
            size = 8,
            color = "#0072B2",
        ),
        line = attr(
            color = "#0072B2",
            width = 2,
            dash = "dash",
        ),
    )

    ymax = 1.05 * max(maximum(mad), maximum(1 .- ssim))

    layout_error = Layout(

        width = 700,
        height = 450,

        template = "simple_white",

        paper_bgcolor = "white",
        plot_bgcolor = "white",

        font = attr(size = font_size),

        margin = attr(
            l = 80,
            r = 30,
            t = 20,
            b = 70,
        ),

        xaxis = attr(
            title = "Number of clusters",
            type = logx ? "log" : "linear",
            showline = true,
            linewidth = 1.5,
            linecolor = "black",
            mirror = true,
            ticks = "outside",
            tickwidth = 1.5,
            ticklen = 6,
            showgrid = true,
            zeroline = false
        ),

        yaxis = attr(
            title = "Error",
            range = [0, ymax],
            showline = true,
            linewidth = 1.5,
            linecolor = "black",
            mirror = true,
            ticks = "outside",
            tickwidth = 1.5,
            ticklen = 6,
            showgrid = true,
            zeroline = false,
        ),

        legend = attr(
            x = 0.02,
            y = 0.98,
            bgcolor = "rgba(255,255,255,0.8)",
            borderwidth = 0,
        ),
    )

    error_plot = Plot([mad_trace, ssim_trace], layout_error)

    return nrmse_plot, error_plot
end

nrmse_plot, error_plot = clustering_performance_plots(vec_clusters, NRMSEs, MADs, SSIMs; logx = true, font_size = 15)
nrmse_plot |> display
error_plot |> display

savefig(nrmse_plot, "results/nrmse_exp5.svg")
savefig(error_plot, "results/error_exp5.svg")

## New joint plot
using PlotlyJS

# --- 1. Identify Key Indices for Image Panels ---
target_clusters = [5, 100, 1000, 20000]
target_indices = [findfirst(==(k), vec_clusters) for k in target_clusters]

# Extract magnitude images normalized by phantom size
imgs_top = [abs.(spgr_data[idx][3]) ./ length(phantom.x) for idx in target_indices]
img_ref  = abs_koma ./ length(ground_truth_phantom.x)

# Normalize display intensity consistently
max_val = maximum(img_ref)
imgs_top_norm = [img ./ max_val for img in imgs_top]
img_ref_norm  = img_ref ./ max_val

# Maintain the verified anatomical orientation
orient(img) = reverse(reverse(img, dims=1)', dims=2)

# --- 2. Color Palette & Tick Setup ---
color_nrmse = "black"
color_mad   = "#D55E00"  # Vermilion
color_ssim  = "#0072B2"  # Blue

x_tick_vals = [5, 10, 20, 50, 100, 200, 300, 500, 600, 1000, 2000, 5000, 10000, 20000]
x_tick_text = ["5", "10", "2", "5", "100", "2", "", "5", "", "1000", "2", "5", "10k", "2"]

# --- 3. Subplot Grid Configuration (2 Rows, 6 Columns) ---
# Row 1: A (col 1), B (col 2), C (col 3), D (col 4), [Gap] (col 5), Ref (col 6)
# Row 2: Panel E (cols 1-3), Panel F (cols 4-6)
specs = [
    Spec()           Spec()           Spec()           Spec()           missing          Spec();
    Spec(colspan=3)  missing          missing          Spec(colspan=3)  missing          missing
]

fig = make_subplots(
    rows = 2, cols = 6,
    specs = specs,
    horizontal_spacing = 0.045, # Compact horizontal spacing maximizes image size
    vertical_spacing = 0.02,
    row_heights = [0.38, 0.62]
)

# --- 4. Populate Top Row (Heatmaps) ---
for (col_idx, img) in enumerate(imgs_top_norm)
    add_trace!(fig,
        heatmap(
            z = orient(img),
            colorscale = "Greys",
            showscale = false,
            hoverinfo = "none"
        ),
        row = 1, col = col_idx
    )
end

# Reference image in column 6
add_trace!(fig,
    heatmap(
        z = orient(img_ref_norm),
        colorscale = "Greys",
        showscale = false,
        hoverinfo = "none"
    ),
    row = 1, col = 6
)

# --- 5. Populate Bottom Row ---
# Panel E: NRMSE trace (Row 2, spanning cols 1-3)
add_trace!(fig,
    scatter(
        x = vec_clusters,
        y = NRMSEs,
        mode = "lines+markers",
        name = "NRMSE",
        showlegend = false,
        line = attr(color = color_nrmse, width = 2.5),
        marker = attr(symbol = "circle", size = 7, color = color_nrmse)
    ),
    row = 2, col = 1
)

# Panel F: MAD trace (Row 2, spanning cols 4-6)
add_trace!(fig,
    scatter(
        x = vec_clusters,
        y = MADs,
        mode = "lines+markers",
        name = "MAD",
        showlegend = true,
        line = attr(color = color_mad, width = 2.5),
        marker = attr(symbol = "square", size = 7, color = color_mad)
    ),
    row = 2, col = 4
)

# Panel F: 1 - SSIM trace (Row 2, spanning cols 4-6)
add_trace!(fig,
    scatter(
        x = vec_clusters,
        y = 1 .- SSIMs,
        mode = "lines+markers",
        name = "1 − SSIM",
        showlegend = true,
        line = attr(color = color_ssim, width = 2.5, dash = "dash"),
        marker = attr(symbol = "diamond", size = 7, color = color_ssim)
    ),
    row = 2, col = 4
)

# --- 6. Detect Exact Bottom Axes IDs ---
axis_e_id = string(fig.plot.data[6][:xaxis])
axis_f_id = string(fig.plot.data[7][:xaxis])

axis_e_num = replace(axis_e_id, "x" => "")
axis_f_num = replace(axis_f_id, "x" => "")

# --- 7. Annotations ---
panel_labels = ["A", "B", "C", "D"]
cluster_subtitles = ["5 clusters", "100 clusters", "1000 clusters", "20000 clusters"]

annotations = []

# Headers and Subtitles for Top Row
for i in 1:4
    push!(annotations, attr(
        text = "<b>$(panel_labels[i])</b>",
        xref = (i == 1 ? "x domain" : "x$i domain"),
        yref = (i == 1 ? "y domain" : "y$i domain"),
        x = 0.5, y = 1.05,
        xanchor = "center", yanchor = "bottom",
        showarrow = false, font = attr(size = 18, color = "black")
    ))
    push!(annotations, attr(
        text = cluster_subtitles[i],
        xref = (i == 1 ? "x domain" : "x$i domain"),
        yref = (i == 1 ? "y domain" : "y$i domain"),
        x = 0.5, y = -0.05,
        xanchor = "center", yanchor = "top",
        showarrow = false, font = attr(size = 14, color = "black")
    ))
end

# Reference Image Title (axis 5)
push!(annotations, attr(
    text = "<b>Reference Image</b>",
    xref = "x5 domain", yref = "y5 domain",
    x = 0.5, y = 1.05,
    xanchor = "center", yanchor = "bottom",
    showarrow = false, font = attr(size = 18, color = "black")
))

# Panel Letters: E and F
push!(annotations,
    attr(text = "<b>E</b>", xref = "$(axis_e_id) domain", yref = "y$(axis_e_num) domain", x = 0.94, y = 0.94,
         showarrow = false, font = attr(size = 20, color = "black")),
    attr(text = "<b>F</b>", xref = "$(axis_f_id) domain", yref = "y$(axis_f_num) domain", x = 0.94, y = 0.94,
         showarrow = false, font = attr(size = 20, color = "black"))
)

# Point Annotations A, B, C, D in Panel E
point_ann = [
    (target_clusters[1], NRMSEs[target_indices[1]], "A", -10, -18),
    (target_clusters[2], NRMSEs[target_indices[2]], "B", -14, -14),
    (target_clusters[3], NRMSEs[target_indices[3]], "C", 0, -22),
    (target_clusters[4], NRMSEs[target_indices[4]], "D", 14, 0)
]

for (px, py, txt, ox, oy) in point_ann
    push!(annotations, attr(
        x = log10(px), y = py,
        xref = axis_e_id, yref = "y$(axis_e_num)",
        text = txt,
        showarrow = true,
        arrowhead = 0,
        arrowcolor = "rgba(0,0,0,0)",
        ax = ox, ay = oy,
        font = attr(size = 14, color = "black")
    ))
end

# Centered Unified X-Axis Title
push!(annotations, attr(
    text = "Number of clusters",
    xref = "paper", yref = "paper",
    x = 0.5, y = -0.13,
    xanchor = "center", yanchor = "top",
    showarrow = false, font = attr(size = 18, color = "black")
))

# --- 8. Layout & Dynamic Axis Mapping ---
img_axis_common = attr(
    showticklabels = false,
    showgrid = false,
    zeroline = false,
    showline = false,
    ticks = "",
    constrain = "domain"
)

log_axis_template = attr(
    type = "log",
    tickvals = x_tick_vals,
    ticktext = x_tick_text,
    ticks = "outside",
    tickwidth = 1.2,
    ticklen = 5,
    showline = true,
    linecolor = "black",
    linewidth = 1,
    mirror = true,
    showgrid = true,
    gridcolor = "#ECECEC",
    tickfont = attr(size = 13, color = "black")
)

# Explicit vertical domain limits enforce proper square image heights
y_img_domain  = [0.58, 0.98]
y_plot_domain = [0.00, 0.51]

layout_dict = Dict{Symbol, Any}(
    :template => templates.plotly_white,
    :font => attr(family = "Arial, Helvetica, sans-serif", color = "black"),
    :margin => attr(t = 30, b = 95, l = 85, r = 55),
    :annotations => annotations,

    # Image axes 1 to 5
    :xaxis => merge(img_axis_common, attr(scaleanchor="y", scaleratio=1)),
    :yaxis => merge(img_axis_common, attr(domain = y_img_domain)),
    :xaxis2 => merge(img_axis_common, attr(scaleanchor="y2", scaleratio=1)),
    :yaxis2 => merge(img_axis_common, attr(domain = y_img_domain)),
    :xaxis3 => merge(img_axis_common, attr(scaleanchor="y3", scaleratio=1)),
    :yaxis3 => merge(img_axis_common, attr(domain = y_img_domain)),
    :xaxis4 => merge(img_axis_common, attr(scaleanchor="y4", scaleratio=1)),
    :yaxis4 => merge(img_axis_common, attr(domain = y_img_domain)),
    :xaxis5 => merge(img_axis_common, attr(scaleanchor="y5", scaleratio=1)),
    :yaxis5 => merge(img_axis_common, attr(domain = y_img_domain)),

    # Legend located inside top-right of Panel F
    :legend => attr(
        x = 0.85,
        y = 0.49,
        xanchor = "right",
        yanchor = "top",
        bgcolor = "rgba(255,255,255,0.85)",
        borderwidth = 0,
        font = attr(size = 14)
    )
)

# Apply dynamic log axis definitions
layout_dict[Symbol("xaxis$(axis_e_num)")] = log_axis_template
layout_dict[Symbol("yaxis$(axis_e_num)")] = attr(
    domain = y_plot_domain,
    title = attr(text = "NRMSE (%)", font = attr(size = 17, color = "black"), standoff = 12),
    range = [4.8, 23.2],
    tickvals = [6, 8, 10, 12, 14, 16, 18, 20, 22],
    showline = true, linecolor = "black", linewidth = 1, mirror = true,
    ticks = "outside", tickwidth = 1.2, ticklen = 5,
    showgrid = true, gridcolor = "#ECECEC",
    tickfont = attr(size = 14, color = "black")
)

layout_dict[Symbol("xaxis$(axis_f_num)")] = log_axis_template
layout_dict[Symbol("yaxis$(axis_f_num)")] = attr(
    domain = y_plot_domain,
    title = attr(text = "Error", font = attr(size = 17, color = "black"), standoff = 14),
    range = [-0.015, 0.45],
    tickvals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    ticktext = ["0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4"],
    showline = true, linecolor = "black", linewidth = 1, mirror = true,
    ticks = "outside", tickwidth = 1.2, ticklen = 5,
    showgrid = true, gridcolor = "#ECECEC",
    tickfont = attr(size = 14, color = "black")
)

relayout!(fig; layout_dict...)

display(fig)
savefig(fig, "PDF_exp_5.pdf", width = 1040, height = 620)