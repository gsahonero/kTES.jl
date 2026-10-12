using kTES

circle_off=circle_phantom([0.0, 0.0, 0.0], 3e-2, Dict("T1"=>1100e-3, "T2"=>300e-3, "T2s"=>250e-3, "Δw"=>200e3*2*pi, "ρ"=>1.0));
circle=circle_phantom([5.0e-2, 5.0e-2, 0.0], 3e-2, Dict("T1"=>1100e-3, "T2"=>300e-3, "T2s"=>250e-3, "Δw"=>0.0, "ρ"=>1.0e-5));


spgr_params = SeqParams(Dict("α" => 15, "TE" => 35e-3, "TR" => 2000e-3,  "Nx" => Nx, "Ny" => Ny, "FOVx" => FOVx, "FOVy" => FOVy, "t_ADC" => Nx*sys.ADC_Δt, "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax), "emulated_noise_sequence" => true, "num_preparation_pulses" => 100, "preparation_technique" => Pulses(), "force_kcenter" => true, "gradient_spoilers" => [8π, 8π, 8π], "RF_spoiler_seed" => 117/180*π, "RF_phases" => RF_phases, "phases_technique" => QuadraticPhases(), "pulse_type" => Soft(), "RF_duration" => 2.3e-3, "balancing_gradients" => [false, false, false], "slice_thickness" => slice_thickness, "OF" => 1, "NSA" => 1, "seq_type" => SPGR(), "disable_slice_selection" => false, "phases_technique" => QuadraticPhases()))

spgr_seq = create_2D_seq(spgr_params, SPGR(), sys)
sim_params = Dict("return_type" => "mat", "method" => Bloch())
full_phantom = circle_off + circle
koma_sim_off = simulate(full_phantom, spgr_seq, sys; sim_params = sim_params)
plot_image(abs.(ifftshift(ifft(fftshift(reshape(koma_sim_off[:], (spgr_seq.DEF["parameters"].Nx, spgr_seq.DEF["parameters"].Ny)))))))

koma_sim = simulate(circle, spgr_seq, sys; sim_params = sim_params)
plot_image(abs.(ifftshift(ifft(fftshift(reshape(koma_sim[:], (spgr_seq.DEF["parameters"].Nx, spgr_seq.DEF["parameters"].Ny)))))))

acq_params = default_acq_params()
acq_params.noise_technique = ReducedNoise()
A = create_ktes_support(full_phantom, spgr_seq)
OSP_phantom = create_support_phantoms(full_phantom, spgr_seq, sys)
ktes_sim = simulate_acquisition(OSP_phantom, spgr_params, acq_params, sim_params, A, sys)
plot_image(abs.(ktes_sim[3]))
