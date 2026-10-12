#=
Experiment 4 - CPUof the k-TES paper.

Computational performance with different phantoms using CPU
Make sure the number of threads available to Julia is set. 
Author: Guillermo Sahonero Alvarez
Affiliations: Pontificia Universidad Católica de Chile (IIBM, CIB) / iHEALTH
=#

using kTES
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
FOVx, FOVy, slice_thickness = 0.25, 0.25, 1e-3
Nx, Ny = 128, 128
Δ = FOVx/Nx

## SPGR Sequence

RF_phases = zeros(ComplexF64, Nx)
create_RF_phases!(RF_phases, QuadraticPhases(), 117 / 180 * π)
spgr_params = SeqParams(Dict("α" => 15, "TE" => 5e-3, "TR" => 20e-3,  "Nx" => Nx, "Ny" => Ny, "FOVx" => FOVx, "FOVy" => FOVy, "t_ADC" => Nx*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "num_preparation_pulses" => 100, "preparation_technique" => Pulses(), "force_kcenter" => true, "gradient_spoilers" => [8π, 8π, 8π], "RF_spoiler_seed" => 117/180*π, "RF_phases" => RF_phases, "phases_technique" => QuadraticPhases(), "pulse_type" => Soft(), "RF_duration" => 2.3e-3, "balancing_gradients" => [false, false, false], "slice_thickness" => slice_thickness, "OF" => 1, "NSA" => 1, "seq_type" => SPGR(), "disable_slice_selection" => false, "phases_technique" => QuadraticPhases()))

spgr_seq = create_2D_seq(spgr_params, SPGR(), sys)

## Simulation speed assessment

us_vec = [1, 2, 3, 4, 5, 6, 7, 8]
tests = 10
koma_allocs = zeros(length(us_vec))
koma_gctimes = zeros(length(us_vec), tests)
koma_memory = zeros(length(us_vec))
koma_times = zeros(length(us_vec), tests)

ktes_allocs = zeros(length(us_vec))
ktes_gctimes = zeros(length(us_vec), tests)
ktes_memory = zeros(length(us_vec))
ktes_times = zeros(length(us_vec), tests)

spins_per_phantom = zeros(length(us_vec))
sim_params = Dict("return_type" => "mat", "method" => Bloch())
N_replicas = 10_000
phantom = Phantom()

for (index, us) in enumerate(us_vec)

    global phantom = brain_phantom2D(; us = us)

    koma_profile = @benchmark begin
    spgr_koma = simulate(phantom, spgr_seq, sys; sim_params = sim_params)
    s_koma = reshape(spgr_koma[:], (spgr_seq.DEF["parameters"].Nx, spgr_seq.DEF["parameters"].Ny))
    i_koma = ifftshift(ifft(fftshift(s_koma)))
    abs_koma = abs.(i_koma)
    end samples=tests evals=1 seconds=1e9

    ktes_profile = @benchmark begin
    tissues, _, tissue_maps = phantom_descriptor(phantom)
    plan = build_tissue_support_plan(phantom, tissue_maps, spgr_seq; use_float32=true, any_rf = true)

    A=create_ktes_support(
        phantom, tissue_maps, spgr_seq;
        plan = plan
    )

    OSP_phantom = create_support_phantoms(phantom, spgr_seq, Normal(), sys; T2s_params = T2sParams(method=NoApproximation()), replicas = N_replicas, σ=1/20.0)

    acq_params = default_acq_params()
    acq_params.noise_technique = ReducedNoise()
    spgr_ktes = simulate_acquisition(OSP_phantom, spgr_seq, acq_params, sys, sim_params, A, N_replicas, phantom, tissue_maps, plan)
    end  samples=tests evals=1 seconds=1e9

    koma_allocs[index] = koma_profile.allocs
    koma_gctimes[index, :] = koma_profile.gctimes
    koma_memory[index] = koma_profile.memory
    koma_times[index, :] = koma_profile.times
    
    ktes_allocs[index] = ktes_profile.allocs
    ktes_gctimes[index, :] = ktes_profile.gctimes
    ktes_memory[index] = ktes_profile.memory
    ktes_times[index,:] = ktes_profile.times
    spins_per_phantom[index] = length(phantom.x)
end

## Saving results
benchmark_results = Dict("koma_times"=> koma_times,"koma_allocs" =>koma_allocs, "koma_memory"=>koma_memory, "koma_gctimes"=>koma_gctimes, "ktes_times"=>ktes_times, "ktes_allocs" => ktes_allocs, "ktes_memory"=>ktes_memory, "ktes_gctimes"=>ktes_gctimes, "spins_per_phantom"=>spins_per_phantom)
serialize("results/benchmark_results_cpu_auto.bin", benchmark_results)