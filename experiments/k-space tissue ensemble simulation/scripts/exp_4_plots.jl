#=

Author: Guillermo Sahonero Alvarez
Affiliations: Pontificia Universidad Católica de Chile (IIBM, CIB) / iHEALTH
=#

using PlotlyJS
using kTES
using Serialization

## Reading GPU results
# Read using: 
gpu_read_data = deserialize("results/benchmark_results.bin")

gpu_koma_allocs = gpu_read_data["koma_allocs"]
gpu_koma_memory = gpu_read_data["koma_memory"]
gpu_koma_gctimes = gpu_read_data["koma_gctimes"]
gpu_koma_times = gpu_read_data["koma_times"]
gpu_ktes_allocs = gpu_read_data["ktes_allocs"]
gpu_ktes_memory = gpu_read_data["ktes_memory"]
gpu_ktes_gctimes = gpu_read_data["ktes_gctimes"]
gpu_ktes_times = gpu_read_data["ktes_times"]
n_phantoms = size(gpu_read_data["koma_times"], 1)
gpu_spins_per_phantom = zeros(n_phantoms)
try
    global gpu_spins_per_phantom = gpu_read_data["spins_per_phantom"]
catch e
    global n_phantoms = size(gpu_read_data["koma_times"], 1)
    for i=1:n_phantoms
        gpu_spins_per_phantom[i] = length(brain_phantom2D(; us = i).x)
    end
end
## Reading CPU with single thread
# Read using: 
cpu1_read_data = deserialize("results/benchmark_results_cpu_1.bin") # for one core results
cpu1_koma_allocs = cpu1_read_data["koma_allocs"]
cpu1_koma_memory = cpu1_read_data["koma_memory"]
cpu1_koma_gctimes = cpu1_read_data["koma_gctimes"]
cpu1_koma_times = cpu1_read_data["koma_times"]
cpu1_ktes_allocs = cpu1_read_data["ktes_allocs"]
cpu1_ktes_memory = cpu1_read_data["ktes_memory"]
cpu1_ktes_gctimes = cpu1_read_data["ktes_gctimes"]
cpu1_ktes_times = cpu1_read_data["ktes_times"]
n_phantoms = size(cpu1_read_data["koma_times"], 1)

cpu1_spins_per_phantom = zeros(n_phantoms)
try
    global cpu1_spins_per_phantom = cpu1_read_data["spins_per_phantom"]
catch e
    for i=1:n_phantoms
        global cpu1_spins_per_phantom[i] = length(brain_phantom2D(; us = i).x)
    end
end
## Reading CPU with "auto" thread
# Read using: 
cpu_auto_read_data = deserialize("results/benchmark_results_cpu_auto.bin") # for one core results
cpu_auto_koma_allocs = cpu_auto_read_data["koma_allocs"]
cpu_auto_koma_memory = cpu_auto_read_data["koma_memory"]
cpu_auto_koma_gctimes = cpu_auto_read_data["koma_gctimes"]
cpu_auto_koma_times = cpu_auto_read_data["koma_times"]
cpu_auto_ktes_allocs = cpu_auto_read_data["ktes_allocs"]
cpu_auto_ktes_memory = cpu_auto_read_data["ktes_memory"]
cpu_auto_ktes_gctimes = cpu_auto_read_data["ktes_gctimes"]
cpu_auto_ktes_times = cpu_auto_read_data["ktes_times"]
n_phantoms = size(cpu_auto_read_data["koma_times"], 1)

cpu_auto_spins_per_phantom = zeros(n_phantoms)
try
    global cpu_auto_spins_per_phantom = cpu_auto_read_data["spins_per_phantom"]
catch e
    for i=1:n_phantoms
        global cpu_auto_spins_per_phantom[i] = length(brain_phantom2D(; us = i).x)
    end
end

## Plots
function format_metric_tick(v::Real)
    if v >= 1e6
        val = v / 1e6
        return isinteger(val) ? "$(Int(val))M" : "$(round(val, digits=1))M"
    elseif v >= 1e3
        val = v / 1e3
        return isinteger(val) ? "$(Int(val))k" : "$(round(val, digits=1))k"
    else
        return isinteger(v) ? "$(Int(v))" : string(v)
    end
end

function benchmark_trace(
    x,
    y::AbstractVector;
    name::AbstractString,
    color::AbstractString,
    marker::AbstractString,
    dash::AbstractString = "solid",
    yaxis::AbstractString = "y",
)
    scatter(
        x = x,
        y = y,
        mode = "lines+markers",
        name = name,
        line = attr(
            color = color,
            width = 2,
            dash = dash,
        ),
        marker = attr(
            size = 7,
            color = color,
            symbol = marker,
        ),
        yaxis = yaxis,
    )
end

function benchmark_trace(
    x,
    y::AbstractMatrix;
    name::AbstractString,
    color::AbstractString,
    marker::AbstractString,
    dash::AbstractString = "solid",
    yaxis::AbstractString = "y",
)
    μ = vec(mean(y, dims = 2))
    σ = vec(std(y, dims = 2))

    fillcolor = color == "black" ?
        "rgba(0,0,0,0.15)" :
        "rgba(213,94,0,0.15)"

    lower = scatter(
        x = x,
        y = μ .- σ,
        mode = "lines",
        line = attr(width = 0),
        hoverinfo = "skip",
        showlegend = false,
        yaxis = yaxis,
    )

    upper = scatter(
        x = x,
        y = μ .+ σ,
        mode = "lines",
        fill = "tonexty",
        fillcolor = fillcolor,
        line = attr(width = 0),
        hoverinfo = "skip",
        showlegend = false,
        yaxis = yaxis,
    )

    mean_trace = scatter(
        x = x,
        y = μ,
        mode = "lines+markers",
        name = name,
        line = attr(
            color = color,
            width = 2,
            dash = dash,
        ),
        marker = attr(
            size = 7,
            color = color,
            symbol = marker,
        ),
        yaxis = yaxis,
    )

    return [lower, upper, mean_trace]
end

# =============================================================================
# Main plotting function
# =============================================================================

function benchmark_rate_plot(
    x,
    y1,
    y2;
    xlabel::AbstractString,
    ylabel1::AbstractString,
    ylabel2::AbstractString,
    legend1::AbstractString,
    legend2::AbstractString,
    logx::Bool = true,
    logy1::Bool = false,
    logy2::Bool = false,
    font_size::Int = 15,
    custom_ticks::Union{Nothing, AbstractVector} = nothing,
)
    traces = vcat(
        benchmark_trace(
            x,
            y1;
            name = legend1,
            color = "black",
            marker = "circle",
            yaxis = "y",
        ),
        benchmark_trace(
            x,
            y2;
            name = legend2,
            color = "#D55E00",
            marker = "square",
            dash = "dash",
            yaxis = "y2",
        ),
    )

    xaxis_attrs = if logx
        raw_vals = if custom_ticks !== nothing
            Float64.(custom_ticks)
        else
            min_val = minimum(x)
            max_val = maximum(x)
            min_exp = floor(Int, log10(min_val))
            max_exp = ceil(Int, log10(max_val))

            vals = Float64[]
            for e in min_exp:max_exp
                for mult in (1.0, 3.0)
                    v = mult * 10.0^e
                    if v >= 0.7 * min_val && v <= 1.3 * max_val
                        push!(vals, v)
                    end
                end
            end
            vals
        end

        attr(
            title = xlabel,
            type = "log",
            tickmode = "array",
            tickvals = raw_vals,
            ticktext = [format_metric_tick(v) for v in raw_vals],
            tickfont = attr(size = font_size),
            showline = true,
            linewidth = 1.5,
            linecolor = "black",
            mirror = true,
            ticks = "outside",
            tickwidth = 1.5,
            ticklen = 6,
            showgrid = true,
            zeroline = false,
        )
    else
        attr(
            title = xlabel,
            type = "linear",
            tickfont = attr(size = font_size),
            showline = true,
            linewidth = 1.5,
            linecolor = "black",
            mirror = true,
            ticks = "outside",
            tickwidth = 1.5,
            ticklen = 6,
            showgrid = true,
            zeroline = false,
        )
    end

    layout = Layout(
        width = 700,
        height = 450,

        template = "simple_white",

        paper_bgcolor = "white",
        plot_bgcolor = "white",

        font = attr(size = font_size),

        margin = attr(
            l = 80,
            r = 80,
            t = 20,
            b = 70,
        ),

        xaxis = xaxis_attrs,

        yaxis = attr(
            title = ylabel1,
            tickfont = attr(size = font_size),
            showline = true,
            linewidth = 1.5,
            linecolor = "black",
            mirror = true,
            ticks = "outside",
            tickwidth = 1.5,
            ticklen = 6,
            showgrid = true,  # Horizontal grid lines
            zeroline = false,
            type = logy1 ? "log" : "linear",
        ),

        yaxis2 = attr(
            title = ylabel2,
            tickfont = attr(size = font_size),
            overlaying = "y",
            side = "right",
            showline = true,
            linewidth = 1.5,
            linecolor = "#D55E00",
            ticks = "outside",
            tickwidth = 1.5,
            ticklen = 6,
            showgrid = false, # Kept false so secondary axis grid does not conflict
            zeroline = false,
            type = logy2 ? "log" : "linear",
        ),

        legend = attr(
            x = 0.02,
            y = 0.98,
            bgcolor = "rgba(255,255,255,0.8)",
            borderwidth = 0,
            font = attr(size = font_size),
        ),

        shapes = [
            attr(
                type = "line",
                xref = "paper",
                x0 = 0,
                x1 = 1,
                yref = "y",
                y0 = 1,
                y1 = 1,
                line = attr(
                    color = "gray",
                    width = 1.5,
                    dash = "dot",
                ),
            ),
            attr(
                type = "line",
                xref = "paper",
                x0 = 0,
                x1 = 1,
                yref = "y2",
                y0 = 1,
                y1 = 1,
                line = attr(
                    color = "#D55E00",
                    width = 1.5,
                    dash = "dot",
                ),
            ),
        ],
    )

    return Plot(traces, layout)
end

gpu_plot_fig = benchmark_rate_plot(gpu_spins_per_phantom, gpu_koma_times./gpu_ktes_times, gpu_koma_memory./gpu_ktes_memory;
    xlabel = "Number of isochromats in the reference phantom",
    ylabel1 = "Runtime speedup",
    ylabel2 = "Memory allocation ratio",
    legend1 = "Speedup",
    legend2 = "Allocation ratio",
    logx = true,
    font_size = 15,
)

cpu1_plot_fig = benchmark_rate_plot(cpu1_spins_per_phantom, cpu1_koma_times./cpu1_ktes_times, cpu1_koma_memory./cpu1_ktes_memory;
    xlabel = "Number of isochromats in the reference phantom",
    ylabel1 = "Runtime speedup",
    ylabel2 = "Memory allocation ratio",
    legend1 = "Speedup",
    legend2 = "Allocation ratio",
    logx = true,
    font_size = 15,
)

cpu_auto_plot_fig = benchmark_rate_plot(cpu_auto_spins_per_phantom, cpu_auto_koma_times./cpu_auto_ktes_times, cpu_auto_koma_memory./cpu_auto_ktes_memory;
    xlabel = "Number of isochromats in the reference phantom",
    ylabel1 = "Runtime speedup",
    ylabel2 = "Memory allocation ratio",
    legend1 = "Speedup",
    legend2 = "Allocation ratio",
    logx = true,
    font_size = 15,
)

savefig(gpu_plot_fig, "results/exp_4_gpu_plot.pdf")
savefig(cpu1_plot_fig, "results/exp_4_cpu_1_plot.pdf")
savefig(cpu_auto_plot_fig, "results/exp_4_cpu_auto_plot.pdf")