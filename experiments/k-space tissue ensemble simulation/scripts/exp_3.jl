#=
Experiment 3 of the k-TES paper.

Validation of k-TES under spatially invariant conditions
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

## Metrics
function NRMSE(img_x, img_target)
    return norm(img_x - img_target) / norm(img_target)
end
function MAD(img_x, img_target)
    return mean(abs.(img_target - img_x))/maximum(abs.(img_target))
end

## Preparing scanner
sys = Scanner()
FOVx, FOVy, slice_thickness = 0.25, 0.25, 2e-3
Nx, Ny = 128, 128
Δ = FOVx/Nx

## SPGR Sequence

RF_phases = zeros(ComplexF64, Nx)
create_RF_phases!(RF_phases, QuadraticPhases(), 117 / 180 * π)
spgr_params = SeqParams(Dict("α" => 15, "TE" => 5e-3, "TR" => 20e-3,  "Nx" => Nx, "Ny" => Ny, "FOVx" => FOVx, "FOVy" => FOVy, "t_ADC" => Nx*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "num_preparation_pulses" => 100, "preparation_technique" => Pulses(), "force_kcenter" => true, "gradient_spoilers" => [8π, 8π, 8π], "RF_spoiler_seed" => 117/180*π, "RF_phases" => RF_phases, "phases_technique" => QuadraticPhases(), "pulse_type" => Soft(), "RF_duration" => 2.3e-3, "balancing_gradients" => [false, false, false], "slice_thickness" => slice_thickness, "OF" => 1, "NSA" => 1, "seq_type" => SPGR(), "disable_slice_selection" => false, "phases_technique" => QuadraticPhases()))

spgr_seq = create_2D_seq(spgr_params, SPGR(), sys)

## Simulation Metrics using long SPGR sequence
spgr_long_params = deepcopy(spgr_params)
spgr_long_params.TE=20e-3
spgr_long_params.TR=200e-3
spgr_long_seq = create_2D_seq(spgr_long_params, SPGR(), sys)

us_vec = [1, 2, 3, 4, 5, 6, 7, 8]
replicas_vec = [10, 100, 1000, 10000]
spgr_long_NRMSEs = zeros(length(us_vec), length(replicas_vec))
spgr_long_SSIMs = zeros(length(us_vec), length(replicas_vec))
spgr_long_MADs = zeros(length(us_vec), length(replicas_vec))
spgr_long_data = []
koma_long_spgr_data = []
phantom_lengths = zeros(length(us_vec))

sim_params = Dict("return_type" => "mat", "method" => Bloch())
for (index_us, us) in enumerate(us_vec)
    phantom = brain_phantom2D(; us = us)
    phantom_lengths[index_us] = length(phantom.x)
    spgr_long_koma = simulate(phantom, spgr_long_seq, sys; sim_params = sim_params)
    s_long_koma = reshape(spgr_long_koma[:], (spgr_long_seq.DEF["parameters"].Nx, spgr_long_seq.DEF["parameters"].Ny))
    i_long_koma = ifftshift(ifft(fftshift(s_long_koma)))
    abs_long_koma = abs.(i_long_koma)
    for (index_replica, replica_i) in enumerate(replicas_vec)
        tissues, _, tissue_maps = phantom_descriptor(phantom)
        plan = build_tissue_support_plan(phantom, tissue_maps, spgr_long_seq; use_float32=true, any_rf = true)

        A=create_ktes_support(
        phantom, tissue_maps, spgr_long_seq;
        plan = plan
        )

        OSP_phantom = create_support_phantoms(phantom, spgr_long_seq, Normal(), sys; T2s_params = T2sParams(method=NoApproximation()), replicas = replica_i, σ=1/20.0)

        acq_params = default_acq_params()
        acq_params.noise_technique = ReducedNoise()
        spgr_long_ktes = simulate_acquisition(OSP_phantom, spgr_long_seq, acq_params, sys, sim_params, A, replica_i, phantom, tissue_maps, plan)

        spgr_long_NRMSEs[index_us, index_replica] = round(NRMSE(spgr_long_ktes[3], i_long_koma)*100, digits=6)
        spgr_long_SSIMs[index_us, index_replica] = assess_ssim(abs.(spgr_long_ktes[3])./maximum(abs.(spgr_long_ktes[3])), abs_long_koma./maximum(abs.(abs_long_koma)))
        spgr_long_MADs[index_us, index_replica] = round(MAD(spgr_long_ktes[2], s_long_koma)*100, digits=6)
        push!(spgr_long_data, spgr_long_ktes)
        push!(koma_long_spgr_data, s_long_koma)
    end
end

## Simulation: Metrics using long ETL TSE sequence

tse_long_params = SeqParams(Dict("TR" => 2500e-3, "EST" =>40e-3, "ETL" => 16,"Nx" => Nx, "Ny" => Ny, "FOVx" => FOVx, "FOVy" => FOVy, "t_ADC" => Nx*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "OF" => 1, "seq_type" => TSE(), "pulse_type" => Soft(), "NSA" => 1, "gradient_spoilers" => [8π, 8π, 8π], "force_kcenter" => false, "RF_duration" => 2.3e-3, "slice_thickness" => slice_thickness, "disable_slice_selection" => false));

tse_long_seq = create_2D_seq(tse_long_params, TSE(), sys)

us_vec = [1, 2, 3, 4, 5, 6, 7, 8]
replicas_vec = [10, 100, 1000, 10000]
tse_long_NRMSEs = zeros(length(us_vec), length(replicas_vec))
tse_long_SSIMs = zeros(length(us_vec), length(replicas_vec))
tse_long_MADs = zeros(length(us_vec), length(replicas_vec))
tse_long_data = []
koma_long_tse_data = []
phantom_lengths = zeros(length(us_vec))

sim_params = Dict("return_type" => "mat", "method" => Bloch())
for (index_us, us) in enumerate(us_vec)
    phantom = brain_phantom2D(; us = us)
    phantom_lengths[index_us] = length(phantom.x)
    tse_long_koma = simulate(phantom, tse_long_seq, sys; sim_params = sim_params)
    s_long_koma = reshape(tse_long_koma[:], (tse_long_seq.DEF["parameters"].Nx, tse_long_seq.DEF["parameters"].Ny))
    #i_long_koma = ifftshift(ifft(fftshift(s_long_koma)))
    i_long_koma = non_cartesian_ifft(tse_long_seq, s_long_koma, Nx, Ny)
    abs_long_koma = abs.(i_long_koma)
    for (index_replica, replica_i) in enumerate(replicas_vec)
        tissues, _, tissue_maps = phantom_descriptor(phantom)
        plan = build_tissue_support_plan(phantom, tissue_maps, tse_long_seq; use_float32=true, any_rf = true)

        A=create_ktes_support(
        phantom, tissue_maps, tse_long_seq;
        plan = plan
        )

        OSP_phantom = create_support_phantoms(phantom, tse_long_seq, Normal(), sys; T2s_params = T2sParams(method=NoApproximation()), replicas = replica_i, σ=1/20.0)

        acq_params = default_acq_params()
        acq_params.noise_technique = ReducedNoise()
        acq_params.reconstruction_technique = non_cartesian_ifft
        tse_long_ktes = simulate_acquisition(OSP_phantom, tse_long_seq, acq_params, sys, sim_params, A, replica_i, phantom, tissue_maps, plan)

        tse_long_NRMSEs[index_us, index_replica] = round(NRMSE(tse_long_ktes[3], i_long_koma)*100, digits=6)
        tse_long_SSIMs[index_us, index_replica] = assess_ssim(abs.(tse_long_ktes[3])./maximum(abs.(tse_long_ktes[3])), abs_long_koma./maximum(abs.(abs_long_koma)))
        tse_long_MADs[index_us, index_replica] = round(MAD(tse_long_ktes[2], s_long_koma)*100, digits=6)
        push!(tse_long_data, tse_long_ktes)
        push!(koma_long_tse_data, s_long_koma)
    end
end

## Joint visualization
using PlotlyJS

# --- 1. Colors, Symbols & Typography ---
colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
symbols = ["circle", "square", "triangle-up", "diamond"]

FONT_FAMILY = "Arial, Helvetica, sans-serif"
LABEL_SIZE  = 18
TICK_SIZE   = 18
TITLE_SIZE  = 19
TAG_SIZE    = 22

# 2x3 Subplot Grid
fig = make_subplots(
    rows = 2, cols = 3,
    shared_xaxes = true,
    horizontal_spacing = 0.08,
    vertical_spacing   = 0.10
)

# --- 2. Populate Traces with Custom Marker Symbols ---
function add_metrics_row!(fig, row_idx, ssim_data, nrmse_data, mad_data, replicas_vec, phantom_lengths)
    for (i, J) in enumerate(replicas_vec)
        color_i  = colors[mod1(i, length(colors))]
        symbol_i = symbols[mod1(i, length(symbols))]
        show_in_legend = (row_idx == 1)
        
        # Col 1: SSIM
        add_trace!(fig, scatter(
            x = phantom_lengths, y = ssim_data[:, i],
            mode = "lines+markers", name = "J = $J",
            line = attr(color = color_i, width = 2.5),
            marker = attr(size = 8, color = color_i, symbol = symbol_i),
            showlegend = show_in_legend
        ), row = row_idx, col = 1)

        # Col 2: NRMSE
        add_trace!(fig, scatter(
            x = phantom_lengths, y = nrmse_data[:, i],
            mode = "lines+markers", name = "J = $J",
            line = attr(color = color_i, width = 2.5),
            marker = attr(size = 8, color = color_i, symbol = symbol_i),
            showlegend = false
        ), row = row_idx, col = 2)

        # Col 3: MAD
        add_trace!(fig, scatter(
            x = phantom_lengths, y = mad_data[:, i],
            mode = "lines+markers", name = "J = $J",
            line = attr(color = color_i, width = 2.5),
            marker = attr(size = 8, color = color_i, symbol = symbol_i),
            showlegend = false
        ), row = row_idx, col = 3)
    end
end

add_metrics_row!(fig, 1, spgr_long_SSIMs, spgr_long_NRMSEs, spgr_long_MADs, replicas_vec, phantom_lengths)
add_metrics_row!(fig, 2, tse_long_SSIMs, tse_long_NRMSEs, tse_long_MADs, replicas_vec, phantom_lengths)

# --- 3. Axis Formatting Functions ---

## NRMSE/SSIM/MAD Curves plotting

# Helper to generate integer-power log tick marks with explicit superscripts
function get_log10_ticks(min_val, max_val)
    p_min = floor(Int, log10(min_val))
    p_max = ceil(Int, log10(max_val))
    powers = p_min:p_max
    vals = 10.0 .^ powers
    texts = ["10<sup>$p</sup>" for p in powers]
    return vals, texts
end

function configure_x_axis(show_ticks::Bool; title_text="", data_range=(minimum(phantom_lengths), maximum(phantom_lengths)))
    x_vals, x_texts = get_log10_ticks(data_range[1], data_range[2])
    
    return attr(
        title = title_text != "" ? attr(text=title_text, font=attr(size=LABEL_SIZE, family=FONT_FAMILY), standoff=10) : nothing,
        type = "log",
        tickmode = "array",
        tickvals = x_vals,
        ticktext = show_ticks ? x_texts : fill("", length(x_vals)),
        showline = true, linewidth = 1.5, linecolor = "black",
        mirror = true, ticks = "outside", tickwidth = 1.5, tickcolor = "black",
        tickfont = attr(size=TICK_SIZE, family=FONT_FAMILY),
        showgrid = true, gridcolor = "#E5E5E5",
        showticklabels = show_ticks
    )
end

# ADAPTIVE SSIM FORMATTER (Dynamically zoomed)
function configure_ssim_y_axis(ssim_matrix)
    valid_vals = filter(!isnan, vec(ssim_matrix))
    min_val = minimum(valid_vals)
    
    if min_val >= 0.95
        lower_bound = floor(min_val * 100) / 100
        if lower_bound >= min_val
            lower_bound -= 0.005
        end
        upper_bound = 1.002
    else
        lower_bound = floor(min_val * 20) / 20
        if lower_bound >= min_val
            lower_bound -= 0.02
        end
        upper_bound = 1.005
    end

    return attr(
        type = "linear",
        range = [lower_bound, upper_bound],
        autorange = false,
        showline = true, linewidth = 1.5, linecolor = "black",
        mirror = true, ticks = "outside", tickwidth = 1.5, tickcolor = "black",
        tickfont = attr(size=TICK_SIZE, family=FONT_FAMILY),
        showgrid = true, gridcolor = "#E5E5E5"
    )
end

# Linear Axis for MAD (%)
function configure_mad_y_axis(mad_matrix)
    max_val = maximum(filter(!isnan, vec(mad_matrix)))
    return attr(
        type = "linear",
        range = [0.0, max_val * 1.12],
        autorange = false,
        showline = true, linewidth = 1.5, linecolor = "black",
        mirror = true, ticks = "outside", tickwidth = 1.5, tickcolor = "black",
        tickfont = attr(size=TICK_SIZE, family=FONT_FAMILY),
        showgrid = true, gridcolor = "#E5E5E5"
    )
end

# Precision Log Y-Axis for NRMSE
function configure_log_y_axis_from_data(nrmse_matrix, custom_ticks)
    valid_vals = filter(x -> x > 0 && !isnan(x), vec(nrmse_matrix))
    min_d, max_d = minimum(valid_vals), maximum(valid_vals)
    
    visible_ticks = filter(t -> t >= min_d * 0.7 && t <= max_d * 1.3, custom_ticks)
    
    return attr(
        type = "log",
        tickmode = "array",
        tickvals = visible_ticks,
        ticktext = string.(visible_ticks),
        range = [log10(min_d * 0.85), log10(max_d * 1.15)],
        autorange = false,
        showline = true, linewidth = 1.5, linecolor = "black",
        mirror = true, ticks = "outside", tickwidth = 1.5, tickcolor = "black",
        tickfont = attr(size=TICK_SIZE, family=FONT_FAMILY),
        showgrid = true, gridcolor = "#E5E5E5"
    )
end

spgr_nrmse_ticks = [1, 2, 5, 10, 20, 40]
tse_nrmse_ticks  = [1, 2, 3, 5, 7, 10]

# --- 4. Apply Layout ---
relayout!(fig,
    font = attr(family=FONT_FAMILY, color="black"),
    
    # Horizontal Legend with symbols
    legend = attr(
        orientation = "h",
        x = 0.5, y = 1.06,
        xanchor = "center", yanchor = "bottom",
        font = attr(size=20, family=FONT_FAMILY),
        bgcolor = "rgba(255,255,255,0.9)",
        borderwidth = 0
    ),

    # --- Row 1 (SPGR) ---
    xaxis  = configure_x_axis(false),
    yaxis  = configure_ssim_y_axis(spgr_long_SSIMs),
    xaxis2 = configure_x_axis(false),
    yaxis2 = configure_log_y_axis_from_data(spgr_long_NRMSEs, spgr_nrmse_ticks), 
    xaxis3 = configure_x_axis(false),
    yaxis3 = configure_mad_y_axis(spgr_long_MADs),

    # --- Row 2 (TSE) ---
    xaxis4 = configure_x_axis(true),
    yaxis4 = configure_ssim_y_axis(tse_long_SSIMs),
    xaxis5 = configure_x_axis(true, title_text="Isochromats"),
    yaxis5 = configure_log_y_axis_from_data(tse_long_NRMSEs, tse_nrmse_ticks), 
    xaxis6 = configure_x_axis(true),
    yaxis6 = configure_mad_y_axis(tse_long_MADs),

    margin = attr(t=85, b=85, l=150, r=30),
    template = templates.plotly_white,

    annotations = [
        # --- Column Header Titles ---
        attr(text="<b>SSIM</b>", x=0.13, y=1.05, xref="paper", yref="paper", showarrow=false, font=attr(size=TITLE_SIZE)),
        attr(text="<b>NRMSE (%)</b>", x=0.50, y=1.05, xref="paper", yref="paper", showarrow=false, font=attr(size=TITLE_SIZE)),
        attr(text="<b>MAD (%)</b>", x=0.87, y=1.05, xref="paper", yref="paper", showarrow=false, font=attr(size=TITLE_SIZE)),

        # --- Rotated Global Row Titles ---
        attr(text="<b>SPGR</b>", x=-0.07, y=0.78, xref="paper", yref="paper", showarrow=false, font=attr(size=TAG_SIZE, color="#1A365D"), textangle=-90, xanchor="center", yanchor="middle"),
        attr(text="<b>TSE</b>",  x=-0.07, y=0.22, xref="paper", yref="paper", showarrow=false, font=attr(size=TAG_SIZE, color="#1A365D"), textangle=-90, xanchor="center", yanchor="middle"),

        # --- Panel Tags (A and B) ---
        attr(text="<b>A</b>", x=-0.08, y=1.01, xref="paper", yref="paper", showarrow=false, font=attr(size=TAG_SIZE), xanchor="right", yanchor="bottom"),
        attr(text="<b>B</b>", x=-0.08, y=0.48, xref="paper", yref="paper", showarrow=false, font=attr(size=TAG_SIZE), xanchor="right", yanchor="bottom")
    ]
)

display(fig)
savefig(fig, "PDF_exp3.pdf", width=1400, height=800)

## Cropped reconstructions
using PlotlyJS
using Statistics: quantile

FONT_FAMILY = "Arial, Helvetica, sans-serif"
FONT_COLOR  = "black"

"""
Formatea conteos en notación científica con superíndices HTML (e.g., 6506 -> "6.5 × 10<sup>3</sup>").
Si prefieres solo potencias puras de 10 (e.g. 10<sup>4</sup>), ajusta la función según convenga.
"""
function format_scientific_html(n::Real; digits=1)
    n == 0 && return "0"
    exponent = floor(Int, log10(abs(n)))
    mantissa = n / (10.0^exponent)
    mantissa_str = string(round(mantissa, digits=digits))
    
    # Si mantissa es exactamente 1.0 (o muy cercano), se puede mostrar simplemente 10^exp
    if isapprox(mantissa, 1.0; atol=0.05)
        return "10<sup>$exponent</sup>"
    end
    return "$(mantissa_str) × 10<sup>$exponent</sup>"
end

function plot_comparison_grid(;
    spgr_long_data,
    tse_long_data,
    koma_long_spgr_data,
    koma_long_tse_data,
    tse_long_seq,
    replicas_vec,
    phantom_lengths,
    Nx, Ny,
    target_isochromats=[6506, 415856, 8421084],
    target_replica=10000,
    crop_start=64,
    crop_end=128,
    width=1200,
    height=540,
    font_family=FONT_FAMILY,
    font_color=FONT_COLOR
)
    bgcolor = "rgba(0,0,0,0)"
    plot_bgcolor = "rgba(0,0,0,0)"

    replica_index = findfirst(replicas_vec .== target_replica)
    isnothing(replica_index) && error("Replica value $target_replica not found in replicas_vec.")

    us_indices = [
        let idx = findfirst(isapprox.(phantom_lengths, target; atol=1))
            isnothing(idx) ? error("Isochromat count $target not found in phantom_lengths.") : idx
        end
        for target in target_isochromats
    ]

    n_cols = 6
    traces = GenericTrace[]

    # Geometría horizontal
    gap_middle = 0.04
    left_margin_w = 0.14
    block_w = (1.0 - left_margin_w - gap_middle - 0.02) / 2
    x_start_A = left_margin_w
    x_start_B = x_start_A + block_w + gap_middle

    col_w_A = block_w / 3
    col_w_B = block_w / 3
    cell_w = col_w_A - 0.005

    col_domains = []
    for c in 1:3
        push!(col_domains, [x_start_A + (c - 1) * col_w_A, x_start_A + c * col_w_A - 0.005])
    end
    for c in 1:3
        push!(col_domains, [x_start_B + (c - 1) * col_w_B, x_start_B + c * col_w_B - 0.005])
    end

    # Geometría vertical (conservación de aspecto 1:1)
    cell_h = cell_w * (width / height)
    row_gap = 0.035
    y_bottom = 0.16
    y_mid1 = y_bottom + cell_h
    y_top1 = y_mid1 + row_gap
    y_top2 = y_top1 + cell_h

    row_domains = [
        [y_top1, y_top2],  # Fila 1: k-TES
        [y_bottom, y_mid1] # Fila 2: KomaMRI
    ]

    # Formateo científico en las etiquetas inferiores
    col_isochromats = [format_scientific_html(phantom_lengths[idx]) for idx in us_indices]
    images_row1 = []
    images_row2 = []

    for (c, us_idx) in enumerate(repeat(us_indices, 2))
        vis_idx = (us_idx - 1) * length(replicas_vec) + replica_index
        is_spgr = (c <= 3)

        if is_spgr
            img_ktes = abs.(spgr_long_data[vis_idx][3])
            img_koma = abs.(ifftshift(ifft(fftshift(koma_long_spgr_data[vis_idx]))))
        else
            img_ktes = abs.(tse_long_data[vis_idx][3])
            img_koma = abs.(non_cartesian_ifft(tse_long_seq, koma_long_tse_data[vis_idx], Nx, Ny))
        end

        crop_x = crop_start:min(crop_end, size(img_ktes, 1))
        crop_y = crop_start:min(crop_end, size(img_ktes, 2))

        mat_ktes = reverse(collect(transpose(img_ktes[crop_x, crop_y])), dims=1)
        mat_koma = reverse(collect(transpose(img_koma[crop_x, crop_y])), dims=1)

        push!(images_row1, mat_ktes)
        push!(images_row2, mat_koma)
    end

    layout_dict = Dict{Symbol, Any}(
        :paper_bgcolor => bgcolor,
        :plot_bgcolor => plot_bgcolor,
        :font => attr(family=font_family, color=font_color),
        :margin => attr(t=75, b=60, l=15, r=15),
        :showlegend => false,
        :width => width,
        :height => height
    )

    subplot_idx = 1
    for r in 1:2
        for c in 1:n_cols
            img = (r == 1) ? images_row1[c] : images_row2[c]
            z_min = 0.0
            z_max = quantile(img[:], 0.999)

            x_axis_key = subplot_idx == 1 ? :xaxis : Symbol("xaxis$subplot_idx")
            y_axis_key = subplot_idx == 1 ? :yaxis : Symbol("yaxis$subplot_idx")

            layout_dict[x_axis_key] = attr(
                domain=col_domains[c],
                showticklabels=false,
                showgrid=false,
                zeroline=false,
                constrain="domain"
            )
            layout_dict[y_axis_key] = attr(
                domain=row_domains[r],
                showticklabels=false,
                showgrid=false,
                zeroline=false,
                scaleanchor=(subplot_idx == 1 ? "x" : "x$subplot_idx"),
                scaleratio=1,
                constrain="domain",
                autorange="reversed"
            )

            push!(traces, heatmap(;
                z=img,
                xaxis=(subplot_idx == 1 ? "x" : "x$subplot_idx"),
                yaxis=(subplot_idx == 1 ? "y" : "y$subplot_idx"),
                colorscale="Greys",
                zmin=z_min,
                zmax=z_max,
                showscale=false,
                hoverinfo="skip"
            ))
            subplot_idx += 1
        end
    end

    row1_mid_y = (row_domains[1][1] + row_domains[1][2]) / 2
    row2_mid_y = (row_domains[2][1] + row_domains[2][2]) / 2
    headers_y = row_domains[1][2] + 0.03
    numbers_y = row_domains[2][1] - 0.04

    fattr(sz; bold=true) = attr(family=font_family, size=sz, color=font_color)

    annotations = [
        # Rótulos de fila
        attr(x=0.06, y=row1_mid_y, xref="paper", yref="paper", text="<b>k-TES</b>", showarrow=false, font=fattr(18), xanchor="center"),
        attr(x=0.06, y=row2_mid_y, xref="paper", yref="paper", text="<b>KomaMRI</b>", showarrow=false, font=fattr(18), xanchor="center"),

        # Encabezados A y B
        attr(x=x_start_A - 0.01, y=headers_y + 0.02, xref="paper", yref="paper", text="<b>A</b>", showarrow=false, font=fattr(28), xanchor="left"),
        attr(x=(col_domains[1][1] + col_domains[3][2]) / 2, y=headers_y, xref="paper", yref="paper",
             text="<b>SPGR<br>T1-Weighted</b>", showarrow=false, font=fattr(18), xanchor="center", yanchor="bottom"),

        attr(x=x_start_B - 0.01, y=headers_y + 0.02, xref="paper", yref="paper", text="<b>B</b>", showarrow=false, font=fattr(28), xanchor="left"),
        attr(x=(col_domains[4][1] + col_domains[6][2]) / 2, y=headers_y, xref="paper", yref="paper",
             text="<b>TSE<br>T2-Weighted</b>", showarrow=false, font=fattr(18), xanchor="center", yanchor="bottom"),

        # Rótulo inferior
        attr(x=(x_start_A + col_domains[6][2]) / 2, y=0.07, xref="paper", yref="paper",
             text="<b>Number of isochromats</b>", showarrow=false, font=fattr(18), xanchor="center")
    ]

    # Números de isocromatas por columna (Paneles A y B) en formato científico
    for c in 1:3
        x_pos_A = (col_domains[c][1] + col_domains[c][2]) / 2
        x_pos_B = (col_domains[c + 3][1] + col_domains[c + 3][2]) / 2
        push!(annotations, attr(x=x_pos_A, y=numbers_y, xref="paper", yref="paper", text=col_isochromats[c], showarrow=false, font=fattr(15), xanchor="center"))
        push!(annotations, attr(x=x_pos_B, y=numbers_y, xref="paper", yref="paper", text=col_isochromats[c], showarrow=false, font=fattr(15), xanchor="center"))
    end

    layout_dict[:annotations] = annotations

    config = PlotConfig(;
        displaylogo=false,
        toImageButtonOptions=attr(format="svg").fields,
        modeBarButtonsToRemove=["zoom", "pan", "resetScale2d", "autoScale"]
    )

    return PlotlyBase.Plot(traces, Layout(; layout_dict...); config)
end

p = plot_comparison_grid(;
    spgr_long_data=spgr_long_data,
    tse_long_data=tse_long_data,
    koma_long_spgr_data=koma_long_spgr_data,
    koma_long_tse_data=koma_long_tse_data,
    tse_long_seq=tse_long_seq,
    replicas_vec=replicas_vec,
    phantom_lengths=phantom_lengths,
    Nx=Nx, Ny=Ny,
    target_isochromats=[6506, 415856, 6653696],
    target_replica=10000,
    crop_start=64,
    crop_end=128
)

display(p)
savefig(p, "results/PDF_exp3_1.pdf", width=1400, height=800)