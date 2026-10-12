using kTES
using Test

@testset "kTES.jl" begin
    @testset "Phantom creation" begin
        phantom_HF = brain_phantom_HF()
        @test isa(phantom_HF, Phantom)
        phantom_LF = brain_phantom_LF()
        @test isa(phantom_LF, Phantom)
    end
    @testset "Create SPGR sequence" begin
        FOVx, FOVy = 0.25, 0.25
        Nx, Ny, OF = 128, 128, 2
        α, TE, TR = 25.0, 100e-3, 230e-3
        sys = Scanner()
        RF_phases = zeros(ComplexF64, Nx)
        create_RF_phases!(RF_phases, QuadraticPhases(), 117 / 180 * π)
        SPGR_params = SeqParams(Dict("α" => α, "TE" => TE, "TR" => TR,  "Nx" => Nx, "Ny" => Ny, "FOVx" => FOVx, "FOVy" => FOVy, "t_ADC" => Nx*2*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "num_preparation_pulses" => 100, "preparation_technique" => Pulses(), "force_kcenter" => true, "gradient_spoilers" => [8π, 8π, 8π], "RF_spoiler_seed" => 117/180*π, "RF_phases" => RF_phases, "phases_technique" => QuadraticPhases(), "pulse_type" => Soft(), "RF_duration" => 2.3e-3, "balancing_gradients" => [false, false, false], "slice_thickness" => 3e-3, "OF" => 2, "NSA" => 1, "seq_type" => SPGR(), "disable_slice_selection" => false, "phases_technique" => QuadraticPhases(), "RF_spoiler_seed" => 117 / 180 * π))
        SPGR_seq = create_2D_seq(SPGR_params, SPGR(), sys)
        @test isa(SPGR_seq, Sequence)
    end
    @testset "One-spin phantom generation" begin
        phantom_HF = brain_phantom_HF()
        FOVx, FOVy = 0.25, 0.25
        Nx, Ny, OF = 128, 128, 2
        α, TE, TR = 25.0, 100e-3, 230e-3
        sys = Scanner()
        RF_phases = zeros(ComplexF64, Nx)
        create_RF_phases!(RF_phases, QuadraticPhases(), 117 / 180 * π)
        SPGR_params = SeqParams(Dict("α" => α, "TE" => TE, "TR" => TR,  "Nx" => Nx, "Ny" => Ny, "FOVx" => FOVx, "FOVy" => FOVy, "t_ADC" => Nx*2*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "num_preparation_pulses" => 100, "preparation_technique" => Pulses(), "force_kcenter" => true, "gradient_spoilers" => [8π, 8π, 8π], "RF_spoiler_seed" => 117/180*π, "RF_phases" => RF_phases, "phases_technique" => QuadraticPhases(), "pulse_type" => Soft(), "RF_duration" => 2.3e-3, "balancing_gradients" => [false, false, false], "slice_thickness" => 3e-3, "OF" => 2, "NSA" => 1, "seq_type" => SPGR(), "disable_slice_selection" => false, "phases_technique" => QuadraticPhases(), "RF_spoiler_seed" => 117 / 180 * π))
        SPGR_seq = create_2D_seq(SPGR_params, SPGR(), sys)
        one_spin_phantoms = create_support_phantoms(phantom_HF, SPGR_seq, sys; T2s_params = T2sParams(method=NoApproximation()), replicas=10000)
        @test isa(one_spin_phantoms, Vector{Phantom})
        @test length(one_spin_phantoms) == size(tissue_detector(phantom_HF)[1])[1]
    end
    @testset "Canvas generation" begin
        phantom_HF = brain_phantom_HF()
        FOVx, FOVy = 0.25, 0.25
        Nx, Ny, OF = 128, 128, 2
        α, TE, TR = 25.0, 100e-3, 230e-3
        sys = Scanner()
        RF_phases = zeros(ComplexF64, Nx)
        create_RF_phases!(RF_phases, QuadraticPhases(), 117 / 180 * π)
        SPGR_params = SeqParams(Dict("α" => α, "TE" => TE, "TR" => TR,  "Nx" => Nx, "Ny" => Ny, "FOVx" => FOVx, "FOVy" => FOVy, "t_ADC" => Nx*2*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "num_preparation_pulses" => 100, "preparation_technique" => Pulses(), "force_kcenter" => true, "gradient_spoilers" => [8π, 8π, 8π], "RF_spoiler_seed" => 117/180*π, "RF_phases" => RF_phases, "phases_technique" => QuadraticPhases(), "pulse_type" => Soft(), "RF_duration" => 2.3e-3, "balancing_gradients" => [false, false, false], "slice_thickness" => 3e-3, "OF" => 2, "NSA" => 1, "seq_type" => SPGR(), "disable_slice_selection" => false, "phases_technique" => QuadraticPhases(), "RF_spoiler_seed" => 117 / 180 * π))
        SPGR_seq = create_2D_seq(SPGR_params, SPGR(), sys)
        A =  ComplexF32.(create_ktes_support(phantom_HF, SPGR_seq, sys) .+ 0im)
        @test isa(A, Array{ComplexF32})
    end
    @testset "k-TES Simulation" begin
        phantom_HF = brain_phantom_HF(; us = 4)
        FOVx, FOVy = 0.25, 0.25
        Nx, Ny, OF = 128, 128, 2
        α, TE, TR = 25.0, 100e-3, 230e-3
        sys = Scanner()
        RF_phases = zeros(ComplexF64, Nx)
        create_RF_phases!(RF_phases, QuadraticPhases(), 117 / 180 * π)
        SPGR_params = SeqParams(Dict("α" => α, "TE" => TE, "TR" => TR,  "Nx" => Nx, "Ny" => Ny, "FOVx" => FOVx, "FOVy" => FOVy, "t_ADC" => Nx*2*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "num_preparation_pulses" => 100, "preparation_technique" => Pulses(), "force_kcenter" => true, "gradient_spoilers" => [8π, 8π, 8π], "RF_spoiler_seed" => 117/180*π, "RF_phases" => RF_phases, "phases_technique" => QuadraticPhases(), "pulse_type" => Soft(), "RF_duration" => 2.3e-3, "balancing_gradients" => [false, false, false], "slice_thickness" => 3e-3, "OF" => 2, "NSA" => 1, "seq_type" => SPGR(), "disable_slice_selection" => false, "phases_technique" => QuadraticPhases(), "RF_spoiler_seed" => 117 / 180 * π))
        SPGR_seq = create_2D_seq(SPGR_params, SPGR(), sys)
        N_replicas = 100000
        one_spin_phantoms = create_support_phantoms(phantom_HF, SPGR_seq, sys; T2s_params = T2sParams(method=NoApproximation()), replicas=N_replicas)
        acq_params = default_acq_params()
        A =  ComplexF32.(create_ktes_support(phantom_HF, SPGR_seq, sys; spin_σ = 1e-14) .+ 0im)
        sim_data = simulate_acquisition(one_spin_phantoms, SPGR_params, acq_params, Dict("return_type" => "mat", "sim_method" => Bloch()), A, sys; replicas = N_replicas, full_phantom = phantom_HF, T2s_params = T2sParams(method = NoApproximation()))
        @test isa(sim_data, Tuple{Array{ComplexF64, 3}, Matrix{ComplexF64}, Matrix{ComplexF64}, Float64, <:Sequence})
    end
    @testset "k-TES Simulation" begin
        phantom = brain_phantom2D();
        sys = Scanner();
        seq_params = default_SPGR_params(sys);
        seq = create_GE_2D(seq_params, sys);
        data = simulate_acquisition(phantom, seq, sys)
        display(plot_image(abs.(data[3])))
    end
end