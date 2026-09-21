export default_SPGR_params, create_GE_1D, create_2D_seq

"""
    default_SPGR_params(sys::Scanner) -> SeqParams

Construct a default [`SeqParams`](@ref) structure for 2D Spoiled Gradient Echo (SPGR / FLASH) acquisition.

# Default Parameters
- Flip angle \$\\alpha = 45^\\circ\$
- \$T_E = 25\\text{ ms}\$, \$T_R = 500\\text{ ms}\$
- Matrix size: \$128 \\times 128\$, Field of View: \$0.35\\text{ m} \\times 0.35\\text{ m}\$
- Quadratic RF spoiling seed \$\\Delta\\phi = 117^\\circ\$
- Crusher gradient spoilers: \$[8\\pi, 8\\pi, 8\\pi]\$
- Number of signal averages \$\\text{NSA} = 2\$
- Oversampling factor \$\\text{OF} = 2\$
"""
function default_SPGR_params(sys::Scanner)
    RF_phases = zeros(ComplexF64, 128)
    create_RF_phases!(RF_phases, QuadraticPhases(), 117 / 180 * π)
    parameters = SeqParams( Dict(
        "α" => 45.0,
        "TE" => 25e-3,
        "TR" => 500e-3,
        "t_ADC" => 128 * sys.ADC_Δt * 1, # Nx * Δt * OF
        "FOVx" => 0.35,
        "FOVy" => 0.35,
        "Nx" => 128,
        "Ny" => 128,
        "NSA" => 2,
        "num_preparation_pulses" => 1000,
        "preparation_technique" => Pulses(),
        "disable_slice_selection" => false,
        "force_kcenter" => true,
        "gradient_spoilers" => [8π, 8π, 8π], # was 8 pi
        "RF_spoiler_seed" => 117 / 180 * π,
        "RF_phases" => RF_phases,
        "phases_technique" => QuadraticPhases(),
        "pulse_type" => Soft(),
        "RF_duration" => 2.3e-3,
        "balancing_gradients" => [false, false, false],
        "slice_thickness" => 3e-3,
        "t_ramp" => estimate_t_ramp(sys.Gmax, sys.Smax),
        "t_c" => 0.0,
        "t_f" => 0.0,
        "emulated_noise_sequence" => false,
        "OF" => 2,
        "seq_type" => SPGR()
    ))
    return parameters
end


"""
    create_balancing_gradients(sequence::Sequence, balance_strategy::BalancingGradients) -> Sequence

Construct rewinder gradient waveforms to rephase zeroth gradient moments along all axes.
"""
function create_balancing_gradients(sequence::Sequence, balance_strategy::BalancingGradients)
    balancing_element = Sequence([Grad(-sequence.DEF["parameters"].balancing_gradients[1]*G_deph_negative_amplitude, sequence.DEF["parameters"].RF_duration, sequence.DEF["parameters"].t_ramp); Grad(sequence.DEF["parameters"].balancing_gradients[2]*selected_G_phase_negative_amplitude, sequence.DEF["parameters"].RF_duration, sequence.DEF["parameters"].t_ramp); Grad(-sequence.DEF["parameters"].balancing_gradients[3]*Gz, sequence.DEF["parameters"].RF_duration/2, sequence.DEF["parameters"].t_ramp);;])
    return balancing_element
end

"""
    create_GE_1D(sequence::Sequence, sys::Scanner; traj_i=1) -> Sequence

Generate a single 1D gradient echo readout line (TR block) corresponding to phase encoding index `traj_i`.
"""
function create_GE_1D(sequence::Sequence, sys::Scanner; traj_i = 1)    
    # NOTE This should be simpler, there might be several variables that are unnecesarily computed all the time.
    p = sequence.DEF["parameters"]
    if p.emulated_noise_sequence
        OF = 1
    else
        OF = p.OF
    end
    rotative_maximum_gradient = 1/sqrt(2)*sys.Gmax
    RF_subseq = create_RF_pulse(sequence, p.pulse_type, sys; traj_i = traj_i)

    slice_selection_rephase = Sequence([Grad(0e-3, p.RF_duration, p.t_ramp); Grad(0e-3, p.RF_duration, p.t_ramp); Grad(!p.disable_slice_selection * RF_subseq.GR[3].A, p.RF_duration, p.t_ramp);;], [RF_subseq.RF[1];;])

    G_slice_negative_amplitude = - RF_subseq.GR[3].A *0.5
    if p.force_kcenter & iseven(p.Ny)
        G_phase_negative_amplitude = ((p.Ny - 1) / (2 * γ * p.FOVy)) * (1 / (p.RF_duration + p.t_ramp))
    else
        G_phase_negative_amplitude = (p.Ny / (2 * γ * p.FOVy)) * (1 / (p.RF_duration + p.t_ramp))
    end

    #t_s = (p.oversampling_factor * p.Nx-1)*sys.ADC_Δt


    if p.force_kcenter && iseven(p.Nx)
        G_reph_positive_amplitude = 2 * ((p.Nx) - 1) / (2 * p.FOVx * γ * p.t_ADC)
    else
        G_reph_positive_amplitude = 2 * (p.Nx - 1)/(2 * p.FOVx * γ * p.t_ADC)
    end

    if p.force_kcenter && iseven(p.Nx)
        G_deph_negative_amplitude = 1 / (p.RF_duration + p.t_ramp) * (G_reph_positive_amplitude / (p.Nx - 1) * (p.Nx) * p.t_ADC / 2 + G_reph_positive_amplitude * p.t_ramp / 2)
    else
        G_deph_negative_amplitude = 1 / (p.RF_duration + p.t_ramp) * (G_reph_positive_amplitude * p.t_ADC / 2 + G_reph_positive_amplitude * p.t_ramp / 2)
    end

    G_0 = (γ * (p.RF_duration * G_phase_negative_amplitude + p.t_ramp * G_phase_negative_amplitude) - 1 / p.FOVy) * 1 / (γ * (p.RF_duration + p.t_ramp))
    ΔG_y = G_phase_negative_amplitude - G_0

    # Selected phase goes from top to bottom, e.g., if there are 128 phases, then the first one is 1

    if p.force_kcenter && iseven(p.Ny)
    selected_G_phase_negative_amplitude = (p.Ny/2 - (traj_i - 1)) * ΔG_y
    else
        selected_G_phase_negative_amplitude = ((p.Ny - 1) / 2 - (traj_i - 1)) * ΔG_y
    end

    slice_selection_dephase = Sequence([Grad(-G_deph_negative_amplitude, p.RF_duration, p.t_ramp); Grad(-selected_G_phase_negative_amplitude, p.RF_duration, p.t_ramp); Grad(!p.disable_slice_selection * G_slice_negative_amplitude, p.RF_duration, p.t_ramp);;])
    slice_selection_element = slice_selection_rephase + slice_selection_dephase

    #! Delay left for TE positioning
    left_delay = round(p.TE - p.t_ADC/2 - p.t_ramp - p.t_ramp - p.RF_duration - p.t_ramp - p.t_ramp - p.RF_duration/2, digits = 9)
    left_delay_element = Sequence([Grad(0e-3, left_delay, 0); Grad(0e-3, left_delay, 0); Grad(0e-3, left_delay, 0);;])
    #! Readout and ADC block
    #N_ADC = p.oversampling_factor * p.Nx
    N_ADC = p.Nx * OF # round(p.t_ADC / sys.ADC_Δt, digits = 3)
    if N_ADC%floor(N_ADC)!=0
        @warn "N_ADC=$N_ADC is not integer. It will be rounded, but the sequence should be checked."
        @info "For debugging: Nx=$(p.Nx), t_ADC=$(p.t_ADC)"
        N_ADC = ceil(N_ADC)
    end
    readout_element = Sequence([Grad(G_reph_positive_amplitude, p.t_ADC, p.t_ramp); Grad(0e-3, p.t_ADC, p.t_ramp); Grad(0e-3, p.t_ADC, p.t_ramp);;], [RF(0, p.t_ADC);;], [ADC(N_ADC, p.t_ADC, p.t_ramp, 0, angle(p.RF_phases[traj_i]))])

    spoiler_element = Sequence()
    balancing_element = Sequence()
    ## Gz is not considered as this is a 2D sequence
    if sum(p.balancing_gradients)>0
        balancing_element = Sequence([Grad(-p.balancing_gradients[1]*G_deph_negative_amplitude, p.RF_duration, p.t_ramp); Grad(p.balancing_gradients[2]*selected_G_phase_negative_amplitude, p.RF_duration, p.t_ramp); Grad(-p.balancing_gradients[3]*Gz, p.RF_duration/2, p.t_ramp);;])
    else
        balancing_element = Sequence([Grad(0, p.RF_duration, p.t_ramp); Grad(selected_G_phase_negative_amplitude, p.RF_duration, p.t_ramp); Grad(0, p.RF_duration, p.t_ramp);;])
    end
    if sum(p.gradient_spoilers)>0
        gradient_existence = p.gradient_spoilers.>0
        spoiler_amplitude = sys.Gmax#rotative_maximum_gradient*0.5

        t_sp_x = ((p.gradient_spoilers[1]) / (p.FOVx / (p.Nx) * 2π * γ * spoiler_amplitude)) - p.t_ramp
        t_sp_y = ((p.gradient_spoilers[2])/(p.FOVy/p.Ny * 2π * γ * spoiler_amplitude)) - p.t_ramp
        t_sp_z = ((p.gradient_spoilers[3])/(p.slice_thickness * 2π * γ * spoiler_amplitude)) - p.t_ramp
        spoiler_element = Sequence([Grad(gradient_existence[1]*spoiler_amplitude, t_sp_x, p.t_ramp); Grad(gradient_existence[2]*spoiler_amplitude, t_sp_y, p.t_ramp); Grad(gradient_existence[3]*spoiler_amplitude, t_sp_z, p.t_ramp);;])
    end
    
    right_delay = round(p.TR - p.RF_duration/2 - p.t_ramp - sum(spoiler_element.DUR) - sum(balancing_element.DUR) - p.t_ramp - p.t_ADC - p.t_ramp - left_delay - p.t_ramp - p.RF_duration - p.t_ramp - p.t_ramp - p.RF_duration/2, digits = 9)
    right_delay_element = Sequence([Grad(0e-3, right_delay, 0); Grad(0e-3, right_delay, 0); Grad(0e-3, right_delay, 0);;])     

    seq = Sequence()
    seq = seq + slice_selection_element + left_delay_element + readout_element + balancing_element +  spoiler_element + right_delay_element
    
    # NOTE rounds are needed to avoid very small negative amounts
    p.t_c = round(p.RF_duration / 2 + 3 * p.t_ramp + p.RF_duration, digits = 9)
    p.t_f = round(sum(balancing_element.DUR) + sum(spoiler_element.DUR) + p.RF_duration/2 + p.t_ramp, digits = 9)

    seq.DEF["parameters"] = p

    return seq
end


function create_2D_seq(seqParams::SeqParams, seq_type::SPGR, sys::Scanner)
    return create_GE_2D(seqParams, sys)
end