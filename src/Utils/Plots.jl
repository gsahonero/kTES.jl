export joint_plots, multiple_plots, plot_optimization_trajectory, plot_scan_times, plot_seqs_param, scatter_opt_traj_3D_plot, contour_opt_traj_plot

"""
    joint_plots(plots::Vector{Matrix{ComplexF64}}, titles::Vector{String}, title_text::String)
    joint_plots(plots::Vector{PlotlyJS.SyncPlot}, titles::Vector{String}, nrows::Integer, ncols::Integer; kwargs...)

Display multiple 2D complex reconstruction images or subplots in a synchronized Plotly grid.

# Arguments
- `plots`: Vector of 2D complex matrices or `PlotlyJS.SyncPlot` instances.
- `titles`: Subplot titles.
- `title_text` / `(nrows, ncols)`: Main title or subplot grid arrangement.

# Keywords
- `shared_yaxes = false`: Synchronize vertical axis scales across subplots.
- `horizontal_spacing = 0.08`: Gap between adjacent subplots.
- `showlegend = true`: Display trace legends.
- `shared_colorbar = false`: Use a unified color scale across all panels.
"""
function joint_plots(plots::Vector{Matrix{ComplexF64}}, titles::Vector{String}, title_text::String)
    cols = length(plots)
    p = make_subplots(rows = 1, cols = cols, shared_xaxes = true, shared_yaxes = "rows", vertical_spacing = 0.03)
    v_plots = vcat(plots...)
    cmax = maximum(abs.(v_plots))
    cmin = minimum(abs.(v_plots))
    annotations = []
    for i in eachindex(plots)
        p_i = plot_image(abs.(plots[i]), title = titles[i], zmin = cmin, zmax = cmax)
        #p_i = plot_image(abs.(plots[i]), title = titles[i], zmin = minimum(abs.(plots[i])), zmax = maximum(abs.(plots[i])))
        PlotlyBase.add_trace!(p, p_i.plot.data[1], row = 1, col = i)
        annotation = Dict(
            :text => titles[i],
            :xref => "x"*string(i)*" domain",
            :yref => "y domain",
            :x => 0.5,
            :y => 1.05,
            :showarrow => false,
            :row => 1,
            :col => i
        )
       push!(annotations, annotation) 
    end
    relayout!(p, yaxis_scaleanchor = "x", annotations=annotations, title_text = title_text)
    return p    
end


function joint_plots(
    plots::Vector{PlotlyJS.SyncPlot},
    titles::Vector{String},
    nrows::Integer,
    ncols::Integer;
    shared_yaxes::Bool = false,
    horizontal_spacing::Real = 0.08,
    showlegend::Bool = true,
    shared_colorbar::Bool = false,
)

    # helper: "x", "x2", "x3"...  /  "xaxis", "xaxis2"...
    axname(i, base) = i == 1 ? base : string(base, i)

    nsub = length(plots)

    sp = make_subplots(
        rows = nrows,
        cols = ncols,
        shared_yaxes = shared_yaxes,
        horizontal_spacing = horizontal_spacing,
        subplot_titles = permutedims(titles),
    )

    # global and per-subplot z ranges
    gmin = +Inf
    gmax = -Inf
    zmins = fill(+Inf, nsub)
    zmaxs = fill(-Inf, nsub)

    # ---------- 1st pass: compute z ranges ------------------------------------
    for i in 1:nsub
        for tr in plots[i].plot.data
            if haskey(tr, :z)
                z = tr[:z]
                zmins[i] = min(zmins[i], minimum(z))
                zmaxs[i] = max(zmaxs[i], maximum(z))
                gmin = min(gmin, zmins[i])
                gmax = max(gmax, zmaxs[i])
            end
        end
    end

    # ---------- 2nd pass: add traces -----------------------------------------
    for idx in 1:nsub
        row = div(idx - 1, ncols) + 1
        col = mod(idx - 1, ncols) + 1

        for trace in plots[idx].plot.data
            # clean inherited keys
            haskey(trace, :coloraxis)  && delete!(trace, :coloraxis)
            haskey(trace, :colorscale) && delete!(trace, :colorscale)
            haskey(trace, :colorbar)   && delete!(trace, :colorbar)

            trace[:colorscale] = "Greys"

            if shared_colorbar
                trace[:coloraxis] = "coloraxis"
                trace[:showscale] = false
            else
                # one coloraxis per subplot: coloraxis, coloraxis2, ...
                cax_name = axname(idx, "coloraxis")
                trace[:coloraxis] = cax_name
                trace[:showscale] = false
            end

            add_trace!(sp, trace; row = row, col = col)
        end
    end

    # ---------- build layout updates -----------------------------------------
    layout_updates = Dict{Symbol,Any}()

    # enforce 1:1 aspect ratio in every subplot:
    #   xaxis[i] anchored to y[i] and constrained to its domain
    for i in 1:nsub
        layout_updates[Symbol(axname(i, "xaxis"))] =
            attr(scaleanchor = axname(i, "y"),
                 scaleratio  = 1,
                 constrain   = "domain")
    end

    if shared_colorbar
        # single shared grayscale colorbar on the right
        layout_updates[:coloraxis] = attr(
            cmin = gmin, cmax = gmax, cauto = false,
            colorscale = "Greys",
            colorbar = attr(
                x = 1.02,
                y = 0.46,         # a bit lower
                yanchor = "middle",
                len = 0.75,       # shorter bar
                thickness = 14,
                xref = "paper", yref = "paper",
            )
        )
        layout_updates[:margin] = attr(r = 90)

    else
        # independent colorbars per subplot
        row_height = 1.0 / nrows

        for idx in 1:nsub
            row = div(idx - 1, ncols) + 1
            col = mod(idx - 1, ncols) + 1

            # actual x-domain of this subplot
            xax  = Symbol(axname(idx, "xaxis"))
            xdom = sp.plot.layout[xax][:domain]  # [xmin, xmax]

            # x-position of colorbar:
            x_cb = begin
                if col < ncols
                    # put bar roughly in the middle of the gap to the next subplot,
                    # but slightly closer to the current image
                    next_idx = (row - 1) * ncols + (col + 1)
                    next_dom = sp.plot.layout[Symbol(axname(next_idx, "xaxis"))][:domain]
                    gap = next_dom[1] - xdom[2]
                    xdom[2] + 0.35 * gap
                else
                    # last column: a bit to the right of the subplot
                    xdom[2] + 0.03
                end
            end

            # y-position / length for this row (paper coords)
            y_center_row = 1.0 - (row - 0.5) * row_height
            y_cb   = y_center_row - 0.02     # slight downward shift
            len_cb = 0.45 * row_height       # shorter than row height

            cname_sym = Symbol(axname(idx, "coloraxis"))

            layout_updates[cname_sym] = attr(
                cmin = zmins[idx], cmax = zmaxs[idx], cauto = false,
                colorscale = "Greys",
                colorbar = attr(
                    x = x_cb, y = y_cb,
                    xanchor = "left", yanchor = "middle",
                    len = len_cb,
                    thickness = 12,
                    xref = "paper", yref = "paper",
                    xpad = 0, ypad = 0,
                )
            )
        end

        layout_updates[:margin] = attr(r = 90)
    end

    layout_updates[:showlegend] = showlegend

    relayout!(sp; layout_updates...)

    return sp
end





"""
    multiple_plots(images::Vector{Matrix{Float64}}, titles::Vector{String}, main_title::String, aspect_ratio::Float64, plot_params::Dict)

Plot multiple 2D magnitude images in a tiled layout with optional global colorbar matching.
"""
function multiple_plots(images::Vector{Matrix{Float64}}, titles::Vector{String}, main_title::String, aspect_ratio::Float64, plot_params::Dict)
    plotly()
    plots = []
    index = 1    
    if haskey(plot_params, "show_axis")
        show_axis = plot_params["show_axis"]
    else
        show_axis = false
    end
    if haskey(plot_params, "show_grid")
        show_grid = plot_params["show_grid"]
    else
        show_grid = false
    end

    

    if plot_params["colorbar"] == "global"
        num_images = size(images)
        grid_x = Int(floor(sqrt(num_images[1]), digits=0))
        grid_y = Int(ceil(num_images[1]/grid_x, digits=0))
        println("Fitting images into a grid of x=", grid_x, " and y=", grid_y)
        clims = extrema(vcat(images...))
        if aspect_ratio != 1
            if haskey(plot_params, "force_aspect_ratio")
                if plot_params["force_aspect_ratio"] == true
                    println("Warning, forcing aspect ratio to be another value rather than 1.")
                else
                    println("Making aspect ratio equal to 1, use plot_params[\"force_aspect_ratio\"]=true to avoid this.")
                    aspect_ratio = 1
                end
            else
                println("Making aspect ratio equal to 1, use plot_params[\"force_aspect_ratio\"]=true to avoid this.")
                aspect_ratio = 1
            end
        end
        
        for image in images
            plot_i = Plots.heatmap(image', color=:grays, aspect_ratio=aspect_ratio, showaxis = show_axis, grid = show_grid, title = titles[index], plot_titlevspan = 0.1, top_margin=10*Plots.px, clims=clims, colorbar=false)
            push!(plots, plot_i)
            index = index + 1
        end
        trace = clims[1]:0.01:clims[2]

        display(Plots.plot(plots..., Plots.heatmap((trace).*ones(size(trace)[1],1), legend=:none, c=:grays,  xticks=:none, yticks=(1:size(trace)[1]/10:size(trace)[1], string.(round.(LinRange(clims[1], clims[2], 10), digits=1)))), layout=@layout[grid(Int(grid_x),Int(grid_y)) a{0.03w}], plot_title=main_title))
    else
        for image in images
            plot_i = Plots.heatmap(image', color=:grays, aspect_ratio=aspect_ratio, showaxis = show_axis, grid = show_grid, title = titles[index], plot_titlevspan = 0.1, top_margin=10*Plots.px)
            push!(plots, plot_i)
            index = index + 1
        end
        
          display(Plots.plot(plots..., plot_title = main_title, plot_titlevspan=0.1))
    end
end

"""
    plot_optimization_trajectory(acqs, logs)

Visualize sequence parameter optimization trajectories in 3D parameter space (`α`, `TE`, `TR`).
"""
function plot_optimization_trajectory(acqs, logs)

    α_s = zeros(length(acqs))
    TE_s = zeros(length(acqs)) 
    TR_s = zeros(length(acqs))
    index = 1

    for acq_i in acqs
        α_s[index] = acq_i[5].DEF["parameters"].α
        TE_s[index] = acq_i[5].DEF["parameters"].TE
        TR_s[index] = acq_i[5].DEF["parameters"].TR
        index += 1
    end 

    custom_colorscale = [ [0.0, "green"], [1.0, "red"]]

    optimization_trajectory_plot=PlotlyJS.plot( 
    [   
        PlotlyJS.scatter3d( 
        x = α_s, 
        y = TE_s*1e3, 
        z = TR_s*1e3, 
        mode = "lines+markers", 
        marker = attr(size = 4,
                color = logs,
                colorscale = custom_colorscale = [
                    (minimum(logs), "green"),  
                    (maximum(logs), "red")    
                ],  ), 
        line = attr(width = 2) ),
        PlotlyJS.scatter3d( 
            x = [α_s[1]], 
            y = [TE_s[1]*1e3], 
            z = [TR_s[1]*1e3], 
            mode = "markers", 
            marker = attr(size = 6, symbol = "x", color = "red"), # Starting point
            name = "Start" 
        ), 
        PlotlyJS.scatter3d( 
            x = [α_s[end]], 
            y = [TE_s[end]*1e3], 
            z = [TR_s[end]*1e3], 
            mode = "markers", 
            marker = attr(size = 3, symbol = "square", color = "green"), # Ending point
            name = "End" 
        )
        ], 
            Layout( title = "Optimization trajectory", 
            scene = attr( xaxis = attr(title = "α", range = [minimum(α_s) - 1, maximum(α_s) + 1]), 
            yaxis = attr(title = "TE [ms]", range = [minimum(TE_s*1e3) - 10, maximum(TE_s*1e3) + 10]), 
            zaxis = attr(title = "TR [ms]", range = [minimum(TR_s*1e3) - 10, maximum(TR_s*1e3) + 10]) ) ) 
    )
    return optimization_trajectory_plot
end

"""
    plot_scan_times(acqs)

Plot sequence acquisition scan time evolution across optimization iterations.
"""
function plot_scan_times(acqs)
    times = zeros(length(acqs))
    index = 1

    for acq_i in acqs 
        times[index] = get_scan_time(acq_i[5])
        index += 1
    end
    p = PlotlyJS.plot(
            PlotlyJS.scatter(
                x = 1:length(times),
                y = times,
                mode = "lines+markers"
            ),
            Layout(
                title = "Scan time per iteration",
                xaxis = attr(title = "Number of iterations"),
                yaxis = attr(
                    title = "Time",
                    range = [0, maximum(times) * 1.1]
                )
            )
        )
    return p
end

"""
    plot_seqs_param(acqs, parameter::String)

Plot parameter values (e.g. flip angle, TR, TE, BW) across optimization iterations.
"""
function plot_seqs_param(acqs, parameter::String)
    # TODO this code is the same as extract params from acqs
    params = extract_params_from_acqs(acqs, parameter)
    #params = zeros(length(acqs))
    #index = 1
    #for acq_i in acqs
    #    params[index] = getfield(acq_i[5].DEF["parameters"], Symbol(parameter))
    #    index += 1
    #end 

    # NOTE this is asssuming that sys.ADC_Δt is the same
    if parameter == "OF"
        params = 1.0 ./(params * 2e-6 * acqs[1][5].DEF["parameters"].Nx)
        parameter = "BW"
    end
    p = PlotlyJS.plot(
        PlotlyJS.scatter(
            x = 0:length(params) - 1,
            y = params,
            mode = "lines+markers"
        ),
        Layout(
            title = "Evolution of "*parameter,
            xaxis = attr(title = "Number of iterations"),
            yaxis = attr(
                title = parameter,
                range = [0, maximum(params) * 1.1]
            )
        )
    )
    return params, p
end

"""
    scatter_opt_traj_3D_plot(x_data, y_data, z_data, x_line, y_line, z_line, x_label, y_label, z_label)

Interactive 3D scatter plot of sampled parameter points with overlayed optimization trajectory and start/end markers.
"""
function scatter_opt_traj_3D_plot(x_data::Vector{Float64}, y_data::Vector{Float64}, z_data::Vector{Float64}, x_line::Vector{Float64}, y_line::Vector{Float64}, z_line::Vector{Float64}, x_label::String, y_label::String, z_label::String)

    parameter_scatter_3D = PlotlyJS.scatter3d(
        x=x_data,
        y=y_data,
        z=z_data,
        mode="markers",
        name="Explored points",
        marker=attr(
            size=4,
            color=(z_data .- minimum(z_data))./(maximum(z_data) - minimum(z_data)),  
            colorscale=[
                (0.0, "green"),  
                (1.0, "red")    
            ],  
            opacity=0.8,
            cmin=0, cmax=1
        ),
        type="scatter3d"
    )
    
    first_point = PlotlyJS.scatter3d(
        x = [x_line[1]],
        y = [y_line[1]],
        z = [z_line[1]],
        mode = "markers",
        marker = attr(
            size = 5,
            color = "red",
            symbol = "diamond"
        ),
        name = "Start Point"
    )

    # Last point marker
    last_point = PlotlyJS.scatter3d(
        x = [x_line[end]],
        y = [y_line[end]],
        z = [z_line[end]],
        mode = "markers",
        marker = attr(
            size = 5,
            color = "green",
            symbol = "diamond"
        ),
        name = "End Point"
    )

    line_trace_3D = PlotlyJS.scatter3d(
        x = x_line,
        y = y_line,
        z = z_line,
        mode = "lines+markers",
        marker = attr(size = 2, color = "red"),
        line = attr(width = 2, color = "blue"),
        name = "Optimization Trajectory"
    )
    
    opt_plot = PlotlyJS.plot(
        [parameter_scatter_3D, line_trace_3D, first_point, last_point],
        Layout(
            margin=attr(l=0, r=0, b=0, t=0), 
            title = "Optimization space", 
            scene = attr( 
                xaxis = attr(title = x_label), 
                yaxis = attr(title = y_label), 
                zaxis = attr(title = z_label)
                )
            )
        )

    return opt_plot
end

"""
    contour_opt_traj_plot(x_data, y_data, z_data, x_line, y_line, x_label, y_label; n_contours = 10, boundaries = nothing)

2D contour plot of objective loss surface with overlayed optimization trajectory and feasible region boundaries.
"""
function contour_opt_traj_plot(x_data::Vector{Float64}, y_data::Vector{Float64}, z_data::Vector{Float64}, x_line::Vector{Float64}, y_line::Vector{Float64}, x_label::String, y_label::String; n_contours = 10, boundaries = nothing)

    contour_plot = PlotlyJS.contour(
        x = x_data,
        y = y_data,
        z = z_data,
        ncontours = n_contours,
        connectgaps = false,
        name = "Optimization space"
    )

    line_trace = PlotlyJS.scatter(
        x = x_line,
        y = y_line,
        mode = "lines+markers",
        marker = attr(size = 3, color = "red"),
        line = attr(width = 2, color = "blue"),
        name = "Opt. trajectory"
    )

    first_point = PlotlyJS.scatter(
        x = [x_line[1]],
        y = [y_line[1]],
        mode = "markers",
        marker = attr(
            size = 10,
            color = "red",
            symbol = "star"
        ),
        name = "Start Point"
    )

    # Last point marker
    last_point = PlotlyJS.scatter(
        x = [x_line[end]],
        y = [y_line[end]],
        mode = "markers",
        marker = attr(
            size = 10,
            color = "green",
            symbol = "star"
        ),
        name = "End Point"
    )

    if !isnothing(boundaries)
        plots = vcat([contour_plot, line_trace, first_point, last_point], boundaries)
    else
        plots = [contour_plot, line_trace, first_point, last_point]
    end

    p = PlotlyJS.plot(plots, 
            Layout(
                title = "Contour Optimization plot",
                xaxis = attr(title = x_label), 
                yaxis = attr(title = y_label),
                showlegend = true,
                legend = attr(
                    x = 0.85,
                    y = 1.05,
                    xanchor = "center",
                    yanchor = "bottom",
                    orientation = "h"
                )
            )
        );
    return p
end