#= 
Experiment 1 of the k-TES paper.
Accuracy of the Tissue Basis Signal using SPGR and TSE
Author: Guillermo Sahonero Alvarez
Affiliations: Pontificia Universidad Católica de Chile (IIBM, CIB) / iHEALTH
=# 

using kTES
using Images
using CUDA
using Suppressor


## Preparing scanner & acq. parameters
sys = Scanner()
FOV = 0.25 # in meters
Nx = 128
Ny = 128
Δ = FOV / Nx
slice_thickness = 2e-3;

## Sequence definition
RF_phases = zeros(ComplexF64, Nx)
create_RF_phases!(RF_phases, QuadraticPhases(), 117 / 180 * π)
spgr_params = SeqParams(Dict("α" => 15, "TE" => 25e-3, "TR" => 400e-3,  "Nx" => Nx, "Ny" => Ny, "FOVx" => FOV, "FOVy" => FOV, "t_ADC" => Nx*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "num_preparation_pulses" => 100, "preparation_technique" => Pulses(), "force_kcenter" => true, "gradient_spoilers" => [8π, 8π, 8π], "RF_spoiler_seed" => 117/180*π, "RF_phases" => RF_phases, "phases_technique" => QuadraticPhases(), "pulse_type" => Soft(), "RF_duration" => 2.3e-3, "balancing_gradients" => [false, false, false], "slice_thickness" => slice_thickness, "OF" => 1, "NSA" => 1, "seq_type" => SPGR(), "disable_slice_selection" => false, "phases_technique" => QuadraticPhases(), "RF_spoiler_seed" => 117 / 180 * π))
spgr_seq = create_2D_seq(spgr_params, SPGR(), sys)

tse_params = SeqParams(Dict("TR" => 1400e-3, "EST" =>10e-3, "ETL" => 2,"Nx" => Nx, "Ny" => Ny, "FOVx" => FOV, "FOVy" => FOV, "t_ADC" => Nx*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "OF" => 1, "seq_type" => TSE(), "pulse_type" => Soft(), "NSA" => 1, "gradient_spoilers" => [0, 0, 8π], "force_kcenter" => false, "RF_duration" => 1.6e-3, "slice_thickness" => slice_thickness, "disable_slice_selection" => false));
tse_seq = create_2D_seq(tse_params, TSE(), sys)
## Comparison functions
function NRMSE(img_x, img_target)
    return norm(img_x - img_target) / norm(img_target)
end

function MAD(img_x, img_target)
    return mean(abs.(img_target - img_x))/maximum(abs.(img_target))
end

## Preparing phantoms
tissue_prop = Dict(
    "T1"=>1.100, "T2"=>0.1, 
    "ρ" => 1, "Δw"=>0,
    "T2s" => 0.1
    )

Q_vec = [32, 64, 128, 256, 512, 1024, 2048]
phantoms = []
load_pre_generated_phantoms = true
if load_pre_generated_phantoms
    phantoms = load("data/phantoms.jld2", "phantoms")
else
    println("Generating phantoms. This process takes a lot of time. Alternatively, you can load pre-generated phantoms from phantoms.jld2.")
    for Q_i in Q_vec
        xs, ys, zs = create_2d_grid_space(Δ, Q_i)
        cube = cube_phantom(xs, ys, zs, [0,0,0], Δ, tissue_prop)
        push!(phantoms, cube)
    end
    save("data/phantoms.jld2", "phantoms", phantoms)
end

replicas_vec = [10, 100, 1000, 10000]

## Experiment with TSE

σ_vec = [1/10, 1/16, 1/20, 1/25, 1/32, 1/40, 1/50]

NRMSE_data_TSE = []
SSIM_data_TSE = []
MAD_data_TSE = []
tse_full_data = []
tse_full_koma_data = []

for (index_σ, σ_i) in enumerate(σ_vec)

    NRMSEs = zeros(length(Q_vec), length(replicas_vec))
    SSIMs = zeros(length(Q_vec), length(replicas_vec))
    MADs = zeros(length(Q_vec), length(replicas_vec))
    tse_data = []
    tse_koma_data = []

    for (index_q, Q_i) in enumerate(Q_vec)
        sim_params = Dict("return_type"=>"mat", "method" => Bloch())
        tse_koma = simulate(phantoms[index_q], tse_seq, sys; sim_params = sim_params)
        s_koma = reshape(tse_koma[:], (Nx, Ny))
        i_koma = non_cartesian_ifft(tse_seq, s_koma, Nx, Ny)
        abs_koma = abs.(i_koma)
        
        for (index_replica, replicas_i) in enumerate(replicas_vec)
            acq_params = default_acq_params()
            acq_params.reconstruction_technique = non_cartesian_ifft
            tissues, _, tissue_maps = phantom_descriptor(phantoms[index_q])
            plan = build_tissue_support_plan(phantoms[index_q], tissue_maps, tse_seq)
            A = ComplexF32.(create_ktes_support(phantoms[index_q], tissue_maps, tse_seq; plan = plan))        
            OSP_phantom = create_support_phantoms(phantoms[index_q], tse_seq, Normal(), sys; T2s_params = T2sParams(method=NoApproximation()), replicas = replicas_i, σ = σ_i)
            tse_ktes = simulate_acquisition(OSP_phantom, tse_seq, acq_params, sys, sim_params, A, replicas_i, phantoms[index_q], tissue_maps, plan)

            NRMSEs[index_q, index_replica] = round(NRMSE(tse_ktes[3], i_koma)*100, digits=6)
            SSIMs[index_q, index_replica] = round(assess_ssim(abs.(tse_ktes[3])./maximum(abs.(tse_ktes[3])), abs_koma./maximum(abs_koma)), digits = 6)
            MADs[index_q, index_replica] = round(MAD(tse_ktes[2], s_koma)*100, digits=6)
            push!(tse_data, tse_ktes)
            push!(tse_koma_data, (s_koma, i_koma))
        end
    end
    push!(NRMSE_data_TSE, NRMSEs)
    push!(SSIM_data_TSE, SSIMs)
    push!(MAD_data_TSE, MADs)
    push!(tse_full_data, tse_data)
    push!(tse_full_koma_data, tse_koma_data)
end


## Experiment with SPGR
σ_vec = [1/10, 1/16, 1/20, 1/25, 1/32, 1/40, 1/50]
NRMSE_data = []
SSIM_data = []
MAD_data = []
spgr_full_data = []
spgr_full_koma_data = []

for (index_σ, σ_i) in enumerate(σ_vec)

    NRMSEs = zeros(length(Q_vec), length(replicas_vec))
    NRMSEs_std = zeros(length(Q_vec), length(replicas_vec))
    SSIMs = zeros(length(Q_vec), length(replicas_vec))
    SSIMs_std = zeros(length(Q_vec), length(replicas_vec))
    MADs = zeros(length(Q_vec), length(replicas_vec))
    MADs_std = zeros(length(Q_vec), length(replicas_vec))
    spgr_data = []
    spgr_koma_data = []

    for (index_q, Q_i) in enumerate(Q_vec)
        sim_params = Dict("return_type"=>"mat", "method" => Bloch())
        spgr_koma = simulate(phantoms[index_q], spgr_seq, sys; sim_params = sim_params)
        s_koma = reshape(spgr_koma[:], (Nx, Ny))
        i_koma = ifftshift(ifft(fftshift(s_koma)))
        abs_koma = abs.(i_koma)
        
        for (index_replica, replicas_i) in enumerate(replicas_vec)
            acq_params = default_acq_params()
            tissues, _, tissue_maps = phantom_descriptor(phantoms[index_q])
            plan = build_tissue_support_plan(phantoms[index_q], tissue_maps, spgr_seq)
            A = ComplexF32.(create_ktes_support(phantoms[index_q], tissue_maps, spgr_seq; plan = plan))        
            OSP_phantom = create_support_phantoms(phantoms[index_q], spgr_seq, Normal(), sys; T2s_params = T2sParams(method=NoApproximation()), replicas = replicas_i, σ = σ_i)
            spgr_ktes = simulate_acquisition(OSP_phantom, spgr_seq, acq_params, sys, sim_params, A, replicas_i, phantoms[index_q], tissue_maps, plan)

            NRMSEs[index_q, index_replica] = round(NRMSE(spgr_ktes[3], i_koma)*100, digits=6)
            SSIMs[index_q, index_replica] = round(assess_ssim(abs.(spgr_ktes[3])./maximum(abs.(spgr_ktes[3])), abs_koma./maximum(abs_koma)), digits = 6)
            MADs[index_q, index_replica] = round(MAD(spgr_ktes[2], s_koma)*100, digits=6)
            push!(spgr_data, spgr_ktes)
            push!(spgr_koma_data, (s_koma, i_koma))
        end
    end
    push!(NRMSE_data, NRMSEs)
    push!(SSIM_data, SSIMs)
    push!(MAD_data, MADs)
    push!(spgr_full_data, spgr_data)
    push!(spgr_full_koma_data, spgr_koma_data)
end


## Visualizing all in the same figure
using PlotlyJS
using Statistics

# 1. Define Common Variables
σ_vec = [1/10, 1/16, 1/20, 1/25, 1/32, 1/40, 1/50]

σ_x = σ_vec .* FOV/Nx
σ_y = σ_vec .* FOV/Ny
σ_z = σ_vec .* slice_thickness
support_volume = 8 .*σ_x .* 8 .*σ_y .* 8 .*σ_z
voxel_volume = FOV/Nx * FOV/Ny * slice_thickness
σ_vec = (support_volume./voxel_volume).*100
println("Support Volume / Voxel Volume (%) = ", σ_vec)

x_tick_vals = σ_vec
x_tick_text = ["2", "0.8", "0.4", "0.25", "0.2", "0.16", "0.125", "0.1", "0.08"]
x_tick_text = string.(round.(x_tick_vals, digits=2))

tick_vals_y = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50]
tick_text_y = ["0.1", "0.25", "0.5", "1", "2", "5", "10", "20", "50"]

replicas_vec = [10, 100, 1000, 10000]

colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
symbols = ["circle", "square", "triangle-up", "diamond"] 

# 2. Process Data
means_nrmse     = zeros(length(σ_vec), length(replicas_vec)) 
means_mad       = zeros(length(σ_vec), length(replicas_vec)) 
means_nrmse_tse = zeros(length(σ_vec), length(replicas_vec)) 
means_mad_tse   = zeros(length(σ_vec), length(replicas_vec)) 

for i in 1:length(σ_vec) 
    means_nrmse[i,:]     = mean(NRMSE_data[i], dims=1) 
    means_mad[i,:]       = mean(MAD_data[i], dims=1) 
    means_nrmse_tse[i,:] = mean(NRMSE_data_TSE[i], dims=1) 
    means_mad_tse[i,:]   = mean(MAD_data_TSE[i], dims=1) 
end 

# 3. Subplot Grid
fig = make_subplots(
    rows=2, cols=2,
    shared_xaxes=true, 
    shared_yaxes=true, 
    vertical_spacing=0.08, 
    horizontal_spacing=0.05
)

# 4. Populate Traces
for i in eachindex(replicas_vec)
    t_attr = attr(
        line=attr(color=colors[i], width=2.5), 
        marker=attr(symbol=symbols[i], size=8, color=colors[i])
    )

    add_trace!(fig, scatter(x=σ_vec, y=means_nrmse[:, i], mode="lines+markers", 
        name="J = $(replicas_vec[i])", showlegend=true; t_attr...), row=1, col=1)
    
    add_trace!(fig, scatter(x=σ_vec, y=means_nrmse_tse[:, i], mode="lines+markers", 
        showlegend=false; t_attr...), row=1, col=2)

    add_trace!(fig, scatter(x=σ_vec, y=means_mad[:, i], mode="lines+markers", 
        showlegend=false; t_attr...), row=2, col=1)

    add_trace!(fig, scatter(x=σ_vec, y=means_mad_tse[:, i], mode="lines+markers", 
        showlegend=false; t_attr...), row=2, col=2)
end

# Axis Attributes
common_x = attr(
    type="log", 
    tickvals=x_tick_vals, 
    ticktext=x_tick_text,
    ticks="outside", 
    showline=true, 
    linecolor="black", 
    linewidth=1,
    mirror=true
)

common_y = attr(
    type="log", 
    tickvals=tick_vals_y, 
    ticktext=tick_text_y, 
    ticks="outside", 
    showline=true, 
    linecolor="black", 
    linewidth=1,
    mirror=true
)

# 5. Global Layout Formatting
relayout!(fig, 
    template = templates.plotly_white, 
    font = attr(family="Arial, Helvetica, sans-serif", color="black", size=14),
    margin = attr(t=75, b=110, l=90, r=30), 
    
    legend = attr(
        orientation="h", 
        yanchor="bottom", y=1.08, 
        xanchor="center", x=0.5, 
        font=attr(size=15), 
        bgcolor="rgba(0,0,0,0)"
    ),
    
    xaxis  = merge(common_x, attr(showticklabels=false)),
    xaxis2 = merge(common_x, attr(showticklabels=false)),
    
    # Bottom subplots keep clean angled ticks without individual x-titles
    xaxis3 = merge(common_x, attr(tickangle=-35, tickfont=attr(size=15, color="black"))),
    xaxis4 = merge(common_x, attr(tickangle=-35, tickfont=attr(size=15, color="black"))),
    
    # Left subplots get Y-axis titles
    yaxis  = merge(common_y, attr(title=attr(text="NRMSE (%)", font=attr(size=18, color="black")), tickfont=attr(size=14))),
    yaxis2 = merge(common_y, attr(showticklabels=false)),
    yaxis3 = merge(common_y, attr(title=attr(text="MAD (%)", font=attr(size=18, color="black")), tickfont=attr(size=14))),
    yaxis4 = merge(common_y, attr(showticklabels=false)),

    annotations = [
        # Subplot Panel Tags (A, B, C, D)
        attr(text="<b>A</b>", x=0.02, y=0.94, xref="x domain",  yref="y domain",  showarrow=false, font=attr(size=20)),
        attr(text="<b>B</b>", x=0.02, y=0.94, xref="x2 domain", yref="y2 domain", showarrow=false, font=attr(size=20)),
        attr(text="<b>C</b>", x=0.02, y=0.94, xref="x3 domain", yref="y3 domain", showarrow=false, font=attr(size=20)),
        attr(text="<b>D</b>", x=0.02, y=0.94, xref="x4 domain", yref="y4 domain", showarrow=false, font=attr(size=20)),
        
        # Column Titles
        attr(text="<b>SPGR</b>", x=0.5, y=1.02, xref="x domain",  yref="y domain",  xanchor="center", yanchor="bottom", showarrow=false, font=attr(size=18)),
        attr(text="<b>TSE</b>",  x=0.5, y=1.02, xref="x2 domain", yref="y2 domain", xanchor="center", yanchor="bottom", showarrow=false, font=attr(size=18)),

        # --- SINGLE CENTERED BOTTOM X-TITLE ---
        attr(
            text="|V<sub>d</sub>| / Voxel Volume (%)", 
            x=0.5, y=-0.12, 
            xref="paper", yref="paper", 
            xanchor="center", yanchor="top", 
            showarrow=false, 
            font=attr(size=18, color="black")
        )
    ]
)

# 6. Save and display plot
display(fig)
savefig(fig, "results/PDF_exp1.pdf", width=1050, height=750)
