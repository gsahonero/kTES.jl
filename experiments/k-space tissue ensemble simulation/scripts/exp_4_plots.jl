#=
Author: Guillermo Sahonero Alvarez
Affiliations: Pontificia Universidad Católica de Chile (IIBM, CIB) / iHEALTH
=#

using PlotlyJS
import PlotlyKaleido
using kTES
using Serialization
using Statistics

PlotlyKaleido.restart(mathjax=false)

# =============================================================================
# Configuration
# =============================================================================

font_family = "Arial, Helvetica, sans-serif"

results_dir = raw"C:\Users\gswr\OneDrive\UC-Research\optimizing-Low-Field-MRI-sequences-for-high-field-contrast-equivalent-images\kTES.jl\experiments\k-space tissue ensemble simulation\results"

output_dir = "results"
mkpath(output_dir)

# =============================================================================
# Reading GPU results
# =============================================================================

gpu_read_data = deserialize(
    joinpath(results_dir, "benchmark_results.bin"),
)

gpu_koma_allocs = gpu_read_data["koma_allocs"]
gpu_koma_memory = gpu_read_data["koma_memory"]
gpu_koma_gctimes = gpu_read_data["koma_gctimes"]
gpu_koma_times = gpu_read_data["koma_times"]

gpu_ktes_allocs = gpu_read_data["ktes_allocs"]
gpu_ktes_memory = gpu_read_data["ktes_memory"]
gpu_ktes_gctimes = gpu_read_data["ktes_gctimes"]
gpu_ktes_times = gpu_read_data["ktes_times"]

gpu_spins_per_phantom = if haskey(gpu_read_data, "spins_per_phantom")
    gpu_read_data["spins_per_phantom"]
else
    [
        length(brain_phantom2D(; us=i).x)
        for i in 1:size(gpu_koma_times, 1)
    ]
end

# =============================================================================
# Reading CPU results: single thread
# =============================================================================

cpu1_read_data = deserialize(
    joinpath(results_dir, "benchmark_results_cpu_1.bin"),
)

cpu1_koma_allocs = cpu1_read_data["koma_allocs"]
cpu1_koma_memory = cpu1_read_data["koma_memory"]
cpu1_koma_gctimes = cpu1_read_data["koma_gctimes"]
cpu1_koma_times = cpu1_read_data["koma_times"]

cpu1_ktes_allocs = cpu1_read_data["ktes_allocs"]
cpu1_ktes_memory = cpu1_read_data["ktes_memory"]
cpu1_ktes_gctimes = cpu1_read_data["ktes_gctimes"]
cpu1_ktes_times = cpu1_read_data["ktes_times"]

cpu1_spins_per_phantom = if haskey(cpu1_read_data, "spins_per_phantom")
    cpu1_read_data["spins_per_phantom"]
else
    [
        length(brain_phantom2D(; us=i).x)
        for i in 1:size(cpu1_koma_times, 1)
    ]
end

# =============================================================================
# Reading CPU results: automatic threading
# =============================================================================

cpu_auto_read_data = deserialize(
    joinpath(results_dir, "benchmark_results_cpu_auto.bin"),
)

cpu_auto_koma_allocs = cpu_auto_read_data["koma_allocs"]
cpu_auto_koma_memory = cpu_auto_read_data["koma_memory"]
cpu_auto_koma_gctimes = cpu_auto_read_data["koma_gctimes"]
cpu_auto_koma_times = cpu_auto_read_data["koma_times"]

cpu_auto_ktes_allocs = cpu_auto_read_data["ktes_allocs"]
cpu_auto_ktes_memory = cpu_auto_read_data["ktes_memory"]
cpu_auto_ktes_gctimes = cpu_auto_read_data["ktes_gctimes"]
cpu_auto_ktes_times = cpu_auto_read_data["ktes_times"]

cpu_auto_spins_per_phantom = if haskey(cpu_auto_read_data, "spins_per_phantom")
    cpu_auto_read_data["spins_per_phantom"]
else
    [
        length(brain_phantom2D(; us=i).x)
        for i in 1:size(cpu_auto_koma_times, 1)
    ]
end

# =============================================================================
# Helper formatting and benchmark trace generators
# =============================================================================

function format_metric_tick(v::Real)
    if v >= 1e6
        val = v / 1e6
        return isinteger(val) ?
            string(Int(val), "M") :
            string(round(val, digits=1), "M")
    elseif v >= 1e3
        val = v / 1e3
        return isinteger(val) ?
            string(Int(val), "k") :
            string(round(val, digits=1), "k")
    else
        return isinteger(v) ? string(Int(v)) : string(v)
    end
end

function benchmark_trace(
    x,
    y::AbstractVector;
    name::AbstractString,
    color::AbstractString,
    marker::AbstractString,
    dash::AbstractString="solid",
    yaxis::AbstractString="y",
)
    return scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=name,
        line=attr(
            color=color,
            width=2,
            dash=dash,
        ),
        marker=attr(
            size=6,
            color=color,
            symbol=marker,
        ),
        yaxis=yaxis,
    )
end

function benchmark_trace(
    x,
    y::AbstractMatrix;
    name::AbstractString,
    color::AbstractString,
    marker::AbstractString,
    dash::AbstractString="solid",
    yaxis::AbstractString="y",
)
    μ = vec(mean(y, dims=2))
    σ = vec(std(y, dims=2))

    fillcolor = color == "black" ?
        "rgba(0,0,0,0.15)" :
        "rgba(213,94,0,0.15)"

    lower = scatter(
        x=x,
        y=μ .- σ,
        mode="lines",
        line=attr(width=0),
        hoverinfo="skip",
        showlegend=false,
        yaxis=yaxis,
    )

    upper = scatter(
        x=x,
        y=μ .+ σ,
        mode="lines",
        fill="tonexty",
        fillcolor=fillcolor,
        line=attr(width=0),
        hoverinfo="skip",
        showlegend=false,
        yaxis=yaxis,
    )

    mean_trace = scatter(
        x=x,
        y=μ,
        mode="lines+markers",
        name=name,
        line=attr(
            color=color,
            width=2,
            dash=dash,
        ),
        marker=attr(
            size=6,
            color=color,
            symbol=marker,
        ),
        yaxis=yaxis,
    )

    return [lower, upper, mean_trace]
end

# =============================================================================
# Individual benchmark plot generator
# =============================================================================

function benchmark_rate_plot(
    x,
    y1,
    y2;
    xlabel::AbstractString="",
    ylabel1::AbstractString="",
    ylabel2::AbstractString="",
    legend1::AbstractString="Speedup",
    legend2::AbstractString="Allocation ratio",
    logx::Bool=true,
    logy1::Bool=false,
    logy2::Bool=false,
    font_size::Int=14,
    font_family::AbstractString="Arial, Helvetica, sans-serif",
    show_legend::Bool=true,
    show_xticks::Bool=true,
    custom_ticks::Union{Nothing,AbstractVector}=nothing,
)
    traces = vcat(
        benchmark_trace(
            x,
            y1;
            name=legend1,
            color="black",
            marker="circle",
            yaxis="y",
        ),
        benchmark_trace(
            x,
            y2;
            name=legend2,
            color="#D55E00",
            marker="square",
            dash="dash",
            yaxis="y2",
        ),
    )

    xaxis_attrs = attr(
        title=attr(
            text=xlabel,
            font=attr(
                family=font_family,
                size=font_size,
                color="black",
            ),
        ),
        type=logx ? "log" : "linear",
        tickangle=0,
        tickfont=attr(
            family=font_family,
            size=font_size,
            color="black",
        ),
        showticklabels=show_xticks,
        showline=true,
        linewidth=1.2,
        linecolor="black",
        mirror=true,
        ticks="outside",
        tickwidth=1.2,
        ticklen=5,
        showgrid=true,
        zeroline=false,
    )

    if logx
        raw_vals = if custom_ticks !== nothing
            Float64.(custom_ticks)
        else
            min_val = minimum(x)
            max_val = maximum(x)
            min_exp = floor(Int, log10(min_val))
            max_exp = ceil(Int, log10(max_val))

            vals = Float64[]

            for e in min_exp:max_exp
                for mult in (1.0, 3.0, 5.0)
                    v = mult * 10.0^e

                    if v >= 0.7 * min_val && v <= 1.3 * max_val
                        push!(vals, v)
                    end
                end
            end

            vals
        end

        xaxis_attrs[:tickmode] = "array"
        xaxis_attrs[:tickvals] = raw_vals
        xaxis_attrs[:ticktext] = [
            format_metric_tick(v) for v in raw_vals
        ]
    end

    layout = Layout(
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=attr(
            family=font_family,
            size=font_size,
            color="black",
        ),
        showlegend=show_legend,
        xaxis=xaxis_attrs,
        yaxis=attr(
            title=attr(
                text=ylabel1,
                font=attr(
                    family=font_family,
                    size=font_size,
                    color="black",
                ),
            ),
            tickfont=attr(
                family=font_family,
                size=font_size,
                color="black",
            ),
            showline=true,
            linewidth=1.2,
            linecolor="black",
            mirror=true,
            ticks="outside",
            tickwidth=1.2,
            ticklen=5,
            showgrid=true,
            zeroline=false,
            type=logy1 ? "log" : "linear",
        ),
        yaxis2=attr(
            title=attr(
                text=ylabel2,
                font=attr(
                    family=font_family,
                    size=font_size,
                    color="#D55E00",
                ),
            ),
            tickfont=attr(
                family=font_family,
                size=font_size,
                color="#D55E00",
            ),
            overlaying="y",
            side="right",
            showline=true,
            linewidth=1.2,
            linecolor="#D55E00",
            ticks="outside",
            tickwidth=1.2,
            ticklen=5,
            showgrid=false,
            zeroline=false,
            type=logy2 ? "log" : "linear",
        ),
        legend=attr(
            x=0.03,
            y=0.95,
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=0,
            font=attr(
                family=font_family,
                size=font_size,
            ),
        ),
        shapes=[
            attr(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=1,
                y1=1,
                line=attr(
                    color="gray",
                    width=1.2,
                    dash="dot",
                ),
            ),
            attr(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                yref="y2",
                y0=1,
                y1=1,
                line=attr(
                    color="#D55E00",
                    width=1.2,
                    dash="dot",
                ),
            ),
        ],
    )

    return Plot(traces, layout)
end

# =============================================================================
# Individual plots and exports
# =============================================================================

gpu_plot_fig = benchmark_rate_plot(
    gpu_spins_per_phantom,
    gpu_koma_times ./ gpu_ktes_times,
    gpu_koma_memory ./ gpu_ktes_memory;
    xlabel="Number of isochromats in the reference phantom",
    ylabel1="Runtime speedup",
    ylabel2="Memory allocation ratio",
    legend1="Speedup",
    legend2="Allocation ratio",
    logx=true,
    font_size=15,
    font_family=font_family,
)

cpu1_plot_fig = benchmark_rate_plot(
    cpu1_spins_per_phantom,
    cpu1_koma_times ./ cpu1_ktes_times,
    cpu1_koma_memory ./ cpu1_ktes_memory;
    xlabel="Number of isochromats in the reference phantom",
    ylabel1="Runtime speedup",
    ylabel2="Memory allocation ratio",
    legend1="Speedup",
    legend2="Allocation ratio",
    logx=true,
    font_size=15,
    font_family=font_family,
)

cpu_auto_plot_fig = benchmark_rate_plot(
    cpu_auto_spins_per_phantom,
    cpu_auto_koma_times ./ cpu_auto_ktes_times,
    cpu_auto_koma_memory ./ cpu_auto_ktes_memory;
    xlabel="Number of isochromats in the reference phantom",
    ylabel1="Runtime speedup",
    ylabel2="Memory allocation ratio",
    legend1="Speedup",
    legend2="Allocation ratio",
    logx=true,
    font_size=15,
    font_family=font_family,
)

savefig(
    gpu_plot_fig,
    joinpath(output_dir, "exp_4_gpu_plot.pdf"),
)

savefig(
    cpu1_plot_fig,
    joinpath(output_dir, "exp_4_cpu_1_plot.pdf"),
)

savefig(
    cpu_auto_plot_fig,
    joinpath(output_dir, "exp_4_cpu_auto_plot.pdf"),
)

# =============================================================================
# Joint figure
# =============================================================================

function assemble_benchmark_figure(
    gpu_fig::Plot,
    cpu1_fig::Plot,
    cpu_auto_fig::Plot;
    title_gpu="GPU (NVIDIA RTX 5070)",
    title_cpu1="CPU (single thread)",
    title_cpu_auto="CPU (24 threads)",
    global_xlabel="Number of isochromats in the reference phantom",
    global_ylabel_left="Runtime speedup",
    global_ylabel_right="Memory allocation ratio",
    font_family::AbstractString="Arial, Helvetica, sans-serif",
    width=1400,
    height=800,
)
    # -------------------------------------------------------------------------
    # Property helpers
    # -------------------------------------------------------------------------

    function get_val(obj, key::Symbol, default=nothing)
        if obj isa AbstractDict
            return get(obj, key, get(obj, string(key), default))
        elseif hasproperty(obj, :fields) && obj.fields isa AbstractDict
            return get(
                obj.fields,
                key,
                get(obj.fields, string(key), default),
            )
        elseif hasproperty(obj, key)
            return getproperty(obj, key)
        end

        return default
    end

    function set_val!(obj, key::Symbol, val)
        if obj isa AbstractDict
            obj[key] = val
        elseif hasproperty(obj, :fields) && obj.fields isa AbstractDict
            obj.fields[key] = val
        else
            setproperty!(obj, key, val)
        end
    end

    function delete_key!(obj, key::Symbol)
        if obj isa AbstractDict
            delete!(obj, key)
            delete!(obj, string(key))
        elseif hasproperty(obj, :fields) && obj.fields isa AbstractDict
            delete!(obj.fields, key)
            delete!(obj.fields, string(key))
        end
    end

    # -------------------------------------------------------------------------
    # 1. Trace retargeting
    # -------------------------------------------------------------------------

    function extract_traces(
        fig::Plot,
        x_id::String,
        y_id1::String,
        y_id2::String;
        keep_legend::Bool=false,
    )
        out = GenericTrace[]

        for t in fig.data
            tc = deepcopy(t)

            set_val!(tc, :xaxis, x_id)

            cur_y = string(get_val(tc, :yaxis, "y"))
            set_val!(
                tc,
                :yaxis,
                cur_y == "y2" ? y_id2 : y_id1,
            )

            if !keep_legend
                set_val!(tc, :showlegend, false)
            end

            push!(out, tc)
        end

        return out
    end

    all_traces = vcat(
        extract_traces(
            gpu_fig, "x", "y", "y2";
            keep_legend=true,
        ),
        extract_traces(
            cpu1_fig, "x2", "y3", "y4";
            keep_legend=false,
        ),
        extract_traces(
            cpu_auto_fig, "x3", "y5", "y6";
            keep_legend=false,
        ),
    )

    # -------------------------------------------------------------------------
    # 2. Subplot domains and axes
    # -------------------------------------------------------------------------

    dom_x_left = [0.09, 0.54]
    dom_y_left = [0.15, 0.88]

    dom_x_right = [0.67, 0.92]
    dom_y_top = [0.57, 0.88]
    dom_y_bot = [0.15, 0.46]

    get_ax(p::Plot, sym::Symbol) =
        deepcopy(get_val(p.layout, sym, attr()))

    # GPU: left panel

    ax_x1 = get_ax(gpu_fig, :xaxis)
    ax_y1 = get_ax(gpu_fig, :yaxis)
    ax_y2 = get_ax(gpu_fig, :yaxis2)

    # Swap 300k for 250k.

    tv1 = get_val(ax_x1, :tickvals, nothing)
    tt1 = get_val(ax_x1, :ticktext, nothing)

    if tv1 !== nothing && tt1 !== nothing
        tv1 = Float64.(tv1)
        tt1 = String.(tt1)

        for i in eachindex(tv1)
            if (
                isapprox(tv1[i], 300_000, rtol=0.05) ||
                tt1[i] == "300k" ||
                tt1[i] == "300K"
            )
                tv1[i] = 250_000.0
                tt1[i] = "250k"
            end
        end

        set_val!(ax_x1, :tickvals, tv1)
        set_val!(ax_x1, :ticktext, tt1)
    end

    set_val!(ax_x1, :domain, dom_x_left)
    set_val!(ax_x1, :title, "")
    set_val!(ax_x1, :tickangle, 0)
    set_val!(
        ax_x1,
        :tickfont,
        attr(family=font_family, size=15, color="black"),
    )

    set_val!(ax_y1, :domain, dom_y_left)
    set_val!(ax_y1, :title, "")
    set_val!(ax_y1, :anchor, "x")
    delete_key!(ax_y1, :overlaying)
    set_val!(
        ax_y1,
        :tickfont,
        attr(family=font_family, size=16, color="black"),
    )

    set_val!(ax_y2, :domain, dom_y_left)
    set_val!(ax_y2, :title, "")
    set_val!(ax_y2, :anchor, "x")
    set_val!(ax_y2, :overlaying, "y")
    set_val!(
        ax_y2,
        :tickfont,
        attr(family=font_family, size=16, color="#D55E00"),
    )

    # Compact x-axis configuration for the CPU panels.

    function configure_compact_x(ax, domain, anchor_y)
        set_val!(ax, :domain, domain)
        set_val!(ax, :anchor, anchor_y)
        set_val!(ax, :title, "")
        set_val!(ax, :tickangle, 0)
        set_val!(ax, :showticklabels, true)
        set_val!(
            ax,
            :tickfont,
            attr(family=font_family, size=12, color="black"),
        )

        tv = get_val(ax, :tickvals, nothing)
        tt = get_val(ax, :ticktext, nothing)

        if tv !== nothing && length(tv) > 4
            idx = 1:2:length(tv)

            set_val!(ax, :tickvals, tv[idx])

            if tt !== nothing
                set_val!(ax, :ticktext, tt[idx])
            end
        end
    end

    # CPU single thread: top-right panel

    ax_x2 = get_ax(cpu1_fig, :xaxis)
    ax_y3 = get_ax(cpu1_fig, :yaxis)
    ax_y4 = get_ax(cpu1_fig, :yaxis2)

    configure_compact_x(ax_x2, dom_x_right, "y3")

    set_val!(ax_y3, :domain, dom_y_top)
    set_val!(ax_y3, :title, "")
    set_val!(ax_y3, :anchor, "x2")
    delete_key!(ax_y3, :overlaying)
    set_val!(
        ax_y3,
        :tickfont,
        attr(family=font_family, size=12, color="black"),
    )

    set_val!(ax_y4, :domain, dom_y_top)
    set_val!(ax_y4, :title, "")
    set_val!(ax_y4, :anchor, "x2")
    set_val!(ax_y4, :overlaying, "y3")
    set_val!(
        ax_y4,
        :tickfont,
        attr(family=font_family, size=12, color="#D55E00"),
    )

    # CPU automatic threading: bottom-right panel

    ax_x3 = get_ax(cpu_auto_fig, :xaxis)
    ax_y5 = get_ax(cpu_auto_fig, :yaxis)
    ax_y6 = get_ax(cpu_auto_fig, :yaxis2)

    configure_compact_x(ax_x3, dom_x_right, "y5")

    set_val!(ax_y5, :domain, dom_y_bot)
    set_val!(ax_y5, :title, "")
    set_val!(ax_y5, :anchor, "x3")
    delete_key!(ax_y5, :overlaying)
    set_val!(
        ax_y5,
        :tickfont,
        attr(family=font_family, size=12, color="black"),
    )

    set_val!(ax_y6, :domain, dom_y_bot)
    set_val!(ax_y6, :title, "")
    set_val!(ax_y6, :anchor, "x3")
    set_val!(ax_y6, :overlaying, "y5")
    set_val!(
        ax_y6,
        :tickfont,
        attr(family=font_family, size=12, color="#D55E00"),
    )

    # -------------------------------------------------------------------------
    # 3. Reference shapes: y = 1
    # -------------------------------------------------------------------------

    function extract_shapes(
        p::Plot,
        x_dom,
        y_id1,
        y_id2;
        width=1.5,
    )
        raw_shapes = get_val(p.layout, :shapes, Any[])
        out = Any[]

        for s in raw_shapes
            sc = deepcopy(s)

            set_val!(sc, :xref, "paper")
            set_val!(sc, :x0, x_dom[1])
            set_val!(sc, :x1, x_dom[2])

            cur_yref = string(get_val(sc, :yref, "y"))
            set_val!(
                sc,
                :yref,
                cur_yref == "y2" ? y_id2 : y_id1,
            )

            line_obj = get_val(sc, :line, nothing)

            if line_obj !== nothing
                set_val!(line_obj, :width, width)
            end

            push!(out, sc)
        end

        return out
    end

    all_shapes = vcat(
        extract_shapes(
            gpu_fig, dom_x_left, "y", "y2";
            width=1.5,
        ),
        extract_shapes(
            cpu1_fig, dom_x_right, "y3", "y4";
            width=1.2,
        ),
        extract_shapes(
            cpu_auto_fig, dom_x_right, "y5", "y6";
            width=1.2,
        ),
    )

    # -------------------------------------------------------------------------
    # 4. Subplot titles and global labels
    # -------------------------------------------------------------------------

    annotations = [
        attr(
            text=title_gpu,
            xref="paper",
            yref="paper",
            x=(dom_x_left[1] + dom_x_left[2]) / 2,
            y=0.91,
            showarrow=false,
            xanchor="center",
            yanchor="bottom",
            font=attr(
                family=font_family,
                size=22,
                color="black",
            ),
        ),
        attr(
            text=title_cpu1,
            xref="paper",
            yref="paper",
            x=(dom_x_right[1] + dom_x_right[2]) / 2,
            y=0.91,
            showarrow=false,
            xanchor="center",
            yanchor="bottom",
            font=attr(
                family=font_family,
                size=19,
                color="black",
            ),
        ),
        attr(
            text=title_cpu_auto,
            xref="paper",
            yref="paper",
            x=(dom_x_right[1] + dom_x_right[2]) / 2,
            y=0.48,
            showarrow=false,
            xanchor="center",
            yanchor="bottom",
            font=attr(
                family=font_family,
                size=19,
                color="black",
            ),
        ),

        # Global x-axis label

        attr(
            text=global_xlabel,
            xref="paper",
            yref="paper",
            x=0.50,
            y=0.02,
            showarrow=false,
            xanchor="center",
            yanchor="bottom",
            font=attr(
                family=font_family,
                size=22,
                color="black",
            ),
        ),

        # Global left y-axis label

        attr(
            text=global_ylabel_left,
            xref="paper",
            yref="paper",
            x=0.02,
            y=(dom_y_left[1] + dom_y_left[2]) / 2,
            showarrow=false,
            textangle=-90,
            xanchor="center",
            yanchor="middle",
            font=attr(
                family=font_family,
                size=22,
                color="black",
            ),
        ),

        # Global right y-axis label

        attr(
            text=global_ylabel_right,
            xref="paper",
            yref="paper",
            x=0.98,
            y=(dom_y_left[1] + dom_y_left[2]) / 2,
            showarrow=false,
            textangle=90,
            xanchor="center",
            yanchor="middle",
            font=attr(
                family=font_family,
                size=22,
                color="#D55E00",
            ),
        ),
    ]

    # -------------------------------------------------------------------------
    # 5. Combined layout
    # -------------------------------------------------------------------------

    layout_dict = Dict{Symbol,Any}(
        :width => width,
        :height => height,
        :paper_bgcolor => "white",
        :plot_bgcolor => "white",
        :template => "simple_white",
        :font => attr(
            family=font_family,
            size=14,
            color="black",
        ),
        :margin => attr(
            l=85,
            r=85,
            t=60,
            b=80,
        ),
        :shapes => all_shapes,
        :annotations => annotations,
        :legend => attr(
            x=dom_x_left[1] + 0.02,
            y=dom_y_left[2] - 0.02,
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=0,
            font=attr(
                family=font_family,
                size=16,
            ),
        ),

        # Left panel

        :xaxis => ax_x1,
        :yaxis => ax_y1,
        :yaxis2 => ax_y2,

        # Top-right panel

        :xaxis2 => ax_x2,
        :yaxis3 => ax_y3,
        :yaxis4 => ax_y4,

        # Bottom-right panel

        :xaxis3 => ax_x3,
        :yaxis5 => ax_y5,
        :yaxis6 => ax_y6,
    )

    return Plot(all_traces, Layout(layout_dict))
end

# =============================================================================
# Base plots for the joint figure
# =============================================================================

gpu_plot_fig = benchmark_rate_plot(
    gpu_spins_per_phantom,
    gpu_koma_times ./ gpu_ktes_times,
    gpu_koma_memory ./ gpu_ktes_memory;
    font_size=14,
    font_family=font_family,
)

cpu1_plot_fig = benchmark_rate_plot(
    cpu1_spins_per_phantom,
    cpu1_koma_times ./ cpu1_ktes_times,
    cpu1_koma_memory ./ cpu1_ktes_memory;
    font_size=12,
    font_family=font_family,
    show_xticks=true,
)

cpu_auto_plot_fig = benchmark_rate_plot(
    cpu_auto_spins_per_phantom,
    cpu_auto_koma_times ./ cpu_auto_ktes_times,
    cpu_auto_koma_memory ./ cpu_auto_ktes_memory;
    font_size=12,
    font_family=font_family,
    show_xticks=true,
)

# =============================================================================
# Assemble, display, and export the joint figure
# =============================================================================

final_fig = assemble_benchmark_figure(
    gpu_plot_fig,
    cpu1_plot_fig,
    cpu_auto_plot_fig;
    title_gpu="GPU (NVIDIA RTX 5070)",
    title_cpu1="CPU (single thread)",
    title_cpu_auto="CPU (24 threads)",
    font_family=font_family,
    width=1400,
    height=800,
)

display(final_fig)

savefig(
    final_fig,
    joinpath(output_dir, "PDF_exp4.pdf");
    width=1400,
    height=800,
)