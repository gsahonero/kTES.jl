export create_SE_1D, create_SE_2D, create_2D_seq

"""
    create_SE_2D(sequence_params::SeqParams, sys::Scanner) -> Sequence

Construct a complete 2D Spin Echo (SE) pulse sequence with 90° excitation and 180° refocusing pulses.

# Arguments
- `sequence_params::SeqParams`: Sequence timing and imaging geometry configuration.
- `sys::Scanner`: Hardware constraints.
"""
function create_SE_2D(sequence_params::SeqParams, sys::Scanner)

    seq = Sequence()
    seq.DEF["parameters"] = sequence_params
    if sequence_params.emulated_noise_sequence
        @warn "An emulated noise sequence will replace the provided original sequence. Reduced noise computation must be used for AcqParams struct."
        seq.DEF["parameters"].t_ADC = seq.DEF["parameters"].Nx * sys.ADC_Δt
        #acqParams.noise_technique = ReducedNoise()
    end
    seq = round_seq_parameters(seq)

    RF_spoilers = @MArray ones(ComplexF64, sequence_params.Ny)    
    seq.DEF["parameters"].RF_phases = RF_spoilers

    p90 = deepcopy(seq)
    p90.DEF["parameters"].α = 90.0
    p180 = deepcopy(seq)
    p180.DEF["parameters"].α = 180.0

    if sequence_params.force_kcenter
        @info "Forcing k-center to be zero."
    end

    for i=1:sequence_params.Ny        
        seq_i =  create_SE_1D(seq, p90, p180, sys; traj_i = i)
        seq = seq + seq_i
    end
    return check_sequence(seq, sys)
end

"""
    create_SE_1D(sequence::Sequence, p90::Sequence, p180::Sequence, sys::Scanner; traj_i=1) -> Sequence

Construct a single Spin Echo TR readout block for phase encode step `traj_i`.
"""
function create_SE_1D(sequence::Sequence, p90::Sequence, p180::Sequence, sys::Scanner; traj_i = 1)
    p = sequence.DEF["parameters"]
   
    rotative_maximum_gradient = 1/sqrt(2)*sys.Gmax
    RF_subseq_90 = create_RF_pulse(p90, p.pulse_type, sys; traj_i = traj_i)
    RF_subseq_180 = create_RF_pulse(p180, p.pulse_type, sys; traj_i = traj_i)
    p90 = p90.DEF["parameters"]
    p180 = p180.DEF["parameters"]
    slice_selection_rephase = Sequence([Grad(0e-3, p90.RF_duration, p90.t_ramp); Grad(0e-3, p90.RF_duration, p90.t_ramp); Grad(!p90.disable_slice_selection * RF_subseq_90.GR[3].A, p90.RF_duration, p90.t_ramp);;], [RF_subseq_90.RF[1];;])

    G_slice_negative_amplitude = - RF_subseq_90.GR[3].A *0.5
    if p90.force_kcenter & iseven(p90.Ny)
        G_phase_negative_amplitude = ((p90.Ny - 1) / (2 * γ * p90.FOVy)) * (1 / (p90.RF_duration + p90.t_ramp))
    else
        G_phase_negative_amplitude = (p90.Ny / (2 * γ * p90.FOVy)) * (1 / (p90.RF_duration + p90.t_ramp))
    end

    if p90.force_kcenter & iseven(p90.Nx)
        G_reph_positive_amplitude = 2 * (p90.Nx - 1) / (2 * p90.FOVx * γ * p90.t_ADC)
    else
        G_reph_positive_amplitude = 2 * p90.Nx/(2 * p90.FOVx * γ * p90.t_ADC)
    end

    if p90.force_kcenter & iseven(p90.Nx)
        G_deph_negative_amplitude = 1 / (p90.RF_duration + p90.t_ramp) * (G_reph_positive_amplitude / (p90.Nx - 1) * (p90.Nx) * p90.t_ADC / 2 + G_reph_positive_amplitude * p90.t_ramp / 2)
    else
        G_deph_negative_amplitude = 1 / (p90.RF_duration + p90.t_ramp) * (G_reph_positive_amplitude * p90.t_ADC / 2 + G_reph_positive_amplitude * p90.t_ramp / 2)
    end

    G_0 = (γ * (p90.RF_duration * G_phase_negative_amplitude + p90.t_ramp * G_phase_negative_amplitude) - 1 / p90.FOVy) * 1 / (γ * (p90.RF_duration + p90.t_ramp))
    ΔG_y = G_phase_negative_amplitude - G_0

    if p.force_kcenter & iseven(p.Ny)
        selected_G_phase_negative_amplitude = (p90.Ny/2 - (traj_i - 1)) * ΔG_y
    else
        selected_G_phase_negative_amplitude = ((p90.Ny - 1) / 2 - (traj_i - 1)) * ΔG_y
    end

    slice_selection_dephase = Sequence([Grad(G_deph_negative_amplitude, p90.RF_duration, p90.t_ramp); Grad(selected_G_phase_negative_amplitude, p90.RF_duration, p90.t_ramp); Grad(!p90.disable_slice_selection * G_slice_negative_amplitude, p90.RF_duration, p90.t_ramp);;])

    slice_selection_element = slice_selection_rephase + slice_selection_dephase

    # Under the assumption that crusher gradients are equal to maximum gradient and the same as the spoiler gradient
    G_crusher_amplitude = sys.Gmax
    t_crusher_z = ((p.gradient_spoilers[3])/(p.slice_thickness * 2π * γ * G_crusher_amplitude)) - p.t_ramp

      
    
    pre180_crusher = Sequence([Grad(0, t_crusher_z, p90.t_ramp); Grad(0, t_crusher_z, p90.t_ramp); Grad(G_crusher_amplitude, t_crusher_z, p90.t_ramp);;])

    post180_crusher = Sequence([Grad(0, t_crusher_z, p90.t_ramp); Grad(0, t_crusher_z, p90.t_ramp); Grad(G_crusher_amplitude, t_crusher_z, p90.t_ramp);;])

    slice_selection_rephase180 = Sequence([Grad(0e-3, p180.RF_duration, p180.t_ramp); Grad(0e-3, p180.RF_duration, p180.t_ramp); Grad(!p180.disable_slice_selection * RF_subseq_180.GR[3].A, p180.RF_duration, p180.t_ramp);;], [RF_subseq_180.RF[1];;])

    N_ADC = round(p.t_ADC / sys.ADC_Δt, digits = 3)
    if N_ADC%floor(N_ADC)!=0
        @warn "N_ADC=$N_ADC is not integer. It will be rounded, but the sequence should be checked."
        @info "For debugging: Nx=$(p.Nx), t_ADC=$(p.t_ADC)"
        N_ADC = ceil(N_ADC)
    end

    left_delay = round(p.TE/2 - p180.RF_duration/2 - p180.t_ramp - p180.t_ramp - t_crusher_z - p180.t_ramp - p90.t_ramp - p90.RF_duration - p90.t_ramp - p90.t_ramp - p90.RF_duration/2, digits = 9)
    left_delay_element = Sequence([Grad(0e-3, left_delay, 0); Grad(0e-3, left_delay, 0); Grad(0e-3, left_delay, 0);;])  

    right_delay = round(p.TE - p.t_ADC/2 - p.t_ramp - p180.t_ramp - t_crusher_z - p180.t_ramp - p180.t_ramp - p180.RF_duration - p180.t_ramp - p180.t_ramp - t_crusher_z - p180.t_ramp - left_delay - p.t_ramp - p.RF_duration - p.t_ramp - p.t_ramp - p.RF_duration/2, digits = 9)
    right_delay_element = Sequence([Grad(0e-3, right_delay, 0); Grad(0e-3, left_delay, 0); Grad(0e-3, right_delay, 0);;])

    readout_element = Sequence([Grad(G_reph_positive_amplitude, p.t_ADC, p.t_ramp); Grad(0e-3, p.t_ADC, p.t_ramp); Grad(0e-3, p.t_ADC, p.t_ramp);;], [RF(0, p.t_ADC);;], [ADC(N_ADC, p.t_ADC, p.t_ramp, 0, angle(p.RF_phases[traj_i]))])

    spoiler_element = Sequence()
    balancing_element = Sequence()

    if sum(p.balancing_gradients)>0
        balancing_element = Sequence([Grad(-p.balancing_gradients[1]*G_deph_negative_amplitude, p.RF_duration, p.t_ramp); Grad(p.balancing_gradients[2]*selected_G_phase_negative_amplitude, p.RF_duration, p.t_ramp); Grad(-p.balancing_gradients[3]*Gz, p.RF_duration/2, p.t_ramp);;])
    else
        balancing_element = Sequence([Grad(0, p.RF_duration, p.t_ramp); Grad(selected_G_phase_negative_amplitude, p.RF_duration, p.t_ramp); Grad(0, p.RF_duration, p.t_ramp);;])
    end
    if sum(p.gradient_spoilers)>0
        gradient_existence = p.gradient_spoilers.>0
        spoiler_amplitude = sys.Gmax

        t_sp_x = ((p.gradient_spoilers[1]) / (p.FOVx / p.Nx * 2π * γ * spoiler_amplitude)) - p.t_ramp
        t_sp_y = ((p.gradient_spoilers[2])/(p.FOVy/p.Ny * 2π * γ * spoiler_amplitude)) - p.t_ramp
        t_sp_z = ((p.gradient_spoilers[3])/(p.slice_thickness * 2π * γ * spoiler_amplitude)) - p.t_ramp
        spoiler_element = Sequence([Grad(gradient_existence[1]*spoiler_amplitude, t_sp_x, p.t_ramp); Grad(gradient_existence[2]*spoiler_amplitude, t_sp_y, p.t_ramp); Grad(gradient_existence[3]*spoiler_amplitude, t_sp_z, p.t_ramp);;])
    end

    last_delay = round(p.TR - p90.RF_duration/2 - p.t_ramp - sum(spoiler_element.DUR) - sum(balancing_element.DUR) - p.t_ramp - p.t_ADC - p.t_ramp - right_delay - p180.t_ramp - t_crusher_z - p180.t_ramp - p180.t_ramp - p180.RF_duration - p180.t_ramp - p180.t_ramp - t_crusher_z - p180.t_ramp - left_delay - p90.t_ramp - p90.RF_duration - p90.t_ramp - p90.t_ramp - p90.RF_duration/2 , digits = 9)
    last_delay_element = Sequence([Grad(0e-3, last_delay, 0); Grad(0e-3, left_delay, 0); Grad(0e-3, last_delay, 0);;])
   
    
    seq = Sequence()
    seq = seq + slice_selection_element + left_delay_element  + pre180_crusher + slice_selection_rephase180 + post180_crusher + right_delay_element + readout_element + balancing_element + spoiler_element + last_delay_element

    return seq
end

"""
    create_2D_seq(seqParams::SeqParams, seq_type::SE, sys::Scanner) -> Sequence

Dispatch method generating a 2D Spin Echo sequence.
"""
function create_2D_seq(seqParams::SeqParams, seq_type::SE, sys::Scanner)
    return create_SE_2D(seqParams, sys)
end
