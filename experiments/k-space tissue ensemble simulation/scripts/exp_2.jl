#=
Experiment 2 of the k-TES paper.
Microscopic inhomogeneity modeling.
Author: Guillermo Sahonero Alvarez
Affiliations: Pontificia Universidad Católica de Chile (IIBM, CIB) / iHEALTH
=#

using kTES
using CUDA
using LinearAlgebra
using PlotlyJS

## Preparing scanner
sys = Scanner()
FOVx, FOVy, slice_thickness = 1e-3, 1e-3, 1e-3
Nx, Ny = 128, 128
Δ = FOVx/Nx

## Sequence definition
N_ADC = 100_000
seq_fid = create_FID_sequence(N_ADC, sys)
time = N_ADC * sys.ADC_Δt
## Create phantom
points = 100
xs, ys, zs = create_2d_grid_space(Δ, points)
tissue_prop = Dict(
    "T1"=>1.100, "T2"=>0.1, 
    "ρ" => 1, "Δw"=>0,
    "T2s" => 0.05
    )

cube = cube_phantom(xs, ys, zs, [0,0,0], Δ, tissue_prop)

## Create exponential model
times = 0:sys.ADC_Δt:time-sys.ADC_Δt
T2_decay(t) = exp.(-t./tissue_prop["T2s"])

##  Quantitative comparison
function NRMSE(img_x, img_target)
    return norm(img_x - img_target) / norm(img_target)
end

## Simulating tissue basis signal with FID sequence 
replicas_vec = [10, 100, 1000, 10000, 100000]
NRMSEs = zeros(length(replicas_vec))
signals = []
sim_params = Dict("return_type" => "mat", "method" => Bloch())
for (index_replica, replica_i) in enumerate(replicas_vec)
    N_replicas = replica_i
    OSP_phantom = create_support_phantoms(cube, FOVx, FOVy, slice_thickness, Nx, Ny, Normal(), T2s_params = T2sParams(method = ϵ_Truncated()), replicas = N_replicas)
    T2s_kTES = simulate(OSP_phantom[1], seq_fid, sys; sim_params = sim_params)./replica_i
    kTES_signal = T2s_kTES[:]./maximum(abs.(T2s_kTES[:]))
    NRMSEs[index_replica] = round(NRMSE(T2_decay(times), abs.(kTES_signal))*100, digits=6)
    push!(signals, kTES_signal)
end

## Simulating KomaMRI response
T2s_koma = simulate(cube, seq_fid, sys; sim_params)
T2s_koma = T2s_koma./maximum(abs.(T2s_koma[:]))

## Plotting k-TES and KomaMRI results
using PlotlyJS

# 1. Define Colorblind-Safe Publication Palette (Okabe-Ito / Nature Standard)
colors = [
    "#0072B2", # Blue
    "#E69F00", # Orange
    "#009E73", # Green
    "#D55E00", # Vermilion
    "#CC79A7"  # Reddish Purple
]

# 2. Initialize 1x2 Grid
fig = make_subplots(
    rows=1, cols=2,
    horizontal_spacing=0.10
)

# --- PANEL A: T2* Decay Profile ---

# Signal traces for replicas
for (i, J) in enumerate(replicas_vec)
    add_trace!(fig,
        scatter(
            x = times,
            y = abs.(signals[i]),
            mode = "lines",
            name = "J = $J",
            line = attr(color = colors[mod1(i, length(colors))], width = 2)
        ),
        row=1, col=1
    )
end

# Analytical Exponential Model Trace
add_trace!(fig,
    scatter(
        x = times,
        y = T2_decay.(times),
        mode = "lines",
        name = "Exponential model",
        line = attr(color = "black", dash = "dash", width = 2.5)
    ),
    row=1, col=1
)

# KomaMRI Reference Trace
add_trace!(fig,
    scatter(
        x = times,
        y = abs.(T2s_koma[:]),
        mode = "lines",
        name = "KomaMRI",
        line = attr(color = "#D55E00", width = 2.5)
    ),
    row=1, col=1
)


# --- PANEL B: NRMSE vs J ---

# Clean logarithmic decades for J (10^1 to 10^5)
tickvals_j = [10, 100, 1000, 10000, 100000]
ticktext_j = ["10<sup>1</sup>", "10<sup>2</sup>", "10<sup>3</sup>", "10<sup>4</sup>", "10<sup>5</sup>"]

add_trace!(fig,
    scatter(
        x = replicas_vec,
        y = NRMSEs,
        mode = "lines+markers",
        showlegend = false, 
        line = attr(color = "#0072B2", width = 2.5),
        marker = attr(symbol = "circle", size = 8, color = "#0072B2") 
    ),
    row=1, col=2
)


# --- GLOBAL FORMATTING & LAYOUT ---

# Shared Axis Subtemplate
axis_template = attr(
    showline = true, 
    linecolor = "black", 
    linewidth = 1, 
    mirror = true, 
    ticks = "outside", 
    tickwidth = 1,
    tickfont = attr(size = 13, color = "black")
)

relayout!(fig,
    template = templates.plotly_white,
    font = attr(family="Arial, Helvetica, sans-serif", size=14, color="black"),
    
    # Balanced publication margins
    margin = attr(t=60, b=75, l=85, r=30),
    
    # Inset Legend inside Panel A (clean & borderless)
    legend = attr(
        x = 0.42, 
        y = 0.96, 
        xanchor = "right", 
        yanchor = "top",
        bgcolor = "rgba(255,255,255,0.85)",
        borderwidth = 0, 
        font = attr(size=13)
    ),

    # Panel A Axes (Time in seconds)
    xaxis = merge(axis_template, attr(
        title = attr(text="Time (s)", font=attr(size=16, color="black"), standoff=15),
        range = [0, 0.2]
    )),
    yaxis = merge(axis_template, attr(
        title = attr(text="Normalized Signal", font=attr(size=16, color="black"), standoff=15),
        range = [0, 1.05]
    )),

    # Panel B Axes (Log J vs NRMSE)
    xaxis2 = merge(axis_template, attr(
        title = attr(text="J", font=attr(size=16, color="black"), standoff=15),
        type = "log",
        tickvals = tickvals_j,
        ticktext = ticktext_j
    )),
    yaxis2 = merge(axis_template, attr(
        title = attr(text="NRMSE (%)", font=attr(size=16, color="black"), standoff=15)
    )),

    # Clean Panel Labels (A, B)
    annotations = [
        attr(
            text="<b>A</b>", 
            x=-0.1, y=1.05, 
            xref="x domain", yref="y domain", 
            showarrow=false, font=attr(size=18, color="black")
        ),
        attr(
            text="<b>B</b>", 
            x=-0.1, y=1.05, 
            xref="x2 domain", yref="y2 domain", 
            showarrow=false, font=attr(size=18, color="black")
        )
    ]
)

# Display and Export as Vector PDF
display(fig)
savefig(fig, "results/PDF_exp2.pdf", width=1050, height=520)