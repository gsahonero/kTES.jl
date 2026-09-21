export create_preparation_sequence

"""
    create_preparation_sequence(sequence::Sequence, preparation_technique::Pulses, sys::Scanner) -> Sequence

Construct a magnetization preparation module comprising `num_preparation_pulses` dummy RF pulses separated by `TR`
to drive the spin system into steady state prior to data acquisition.
"""
function create_preparation_sequence(sequence, preparation_technique::Pulses, sys::Scanner)

    @info "Hard preparation pulses will be applied to induce steady-state"
    fixed_RF_duration = hard_RF_duration(180, sys.B1)
    dur_RF = ceil(fixed_RF_duration, digits = 4)
    Δt_G = sys.RF_dead_time + dur_RF + sys.RF_ring_down_time
    pulse = Sequence([Grad(0e-3, Δt_G, 0); Grad(0e-3, Δt_G, 0); Grad(0e-3, Δt_G, 0);;], [RF((sequence.DEF["parameters"].α*π/180)/(2*π*γ*dur_RF), dur_RF, 0, sys.RF_dead_time);;])
    delay_between_pulses = sequence.DEF["parameters"].TR - dur_RF/2 - sys.RF_dead_time - sys.RF_ring_down_time
    delay = Sequence([Grad(0e-3, delay_between_pulses, 0); Grad(0e-3, delay_between_pulses, 0); Grad(0e-3, delay_between_pulses, 0);;], [RF(0, delay_between_pulses);;])

    seq = Sequence()
    for i=1:sequence.DEF["parameters"].num_preparation_pulses
        seq = seq + pulse
        seq = seq + delay
    end

    return seq
end

"""
    create_preparation_sequence(sequence::Sequence, preparation_technique::RepeatSequence, sys::Scanner) -> Sequence

Construct a preparation module by repeating the full sequence with all ADC readouts disabled.
"""
function create_preparation_sequence(sequence, preparation_technique::RepeatSequence, sys::Scanner)
    @info "The same sequence will be repeated for inducing steady-state. num_preparation_pulses is ignored."
    prep_seq = deepcopy(sequence)
    for i in 1:length(prep_seq.ADC)
        prep_seq.ADC[i] = ADC(0, 0, 0, 0, 0)
    end
    return prep_seq
end

"""
    create_preparation_sequence(sequence::Sequence, preparation_technique::Balanced, sys::Scanner) -> Sequence

Construct an alpha/2 preparation module for balanced steady-state sequences.
"""
function create_preparation_sequence(sequence, preparation_technique::Balanced, sys::Scanner)
    @info "The same sequence will be repeated for inducing steady-state. num_preparation_pulses is ignored."
    prep_seq = deepcopy(sequence)
    for i in 1:length(prep_seq.ADC)
        prep_seq.ADC[i] = ADC(0, 0, 0, 0, 0)
    end
    # TODO this should work, this is not working!
    if sum(sequence_params.balancing_gradients)>0
        if sequence_params.num_preparation_pulses>0
            @warn "Adding preparation pulses to balanced sequence."
        end
        # TODO use the sequence params structure instead of going through the whole thing
        seq += create_balanced_preparation(sequence_params.FOVx, sequence_params.FOVy, sequence_params.α/2, sequence_params.TE, sequence_params.TR/2, sequence_params.Nx, sequence_params.Ny, 1, sys, sequence_params.oversampling_factor, sequence_params.disable_slice_selection, sequence_params.max_B1, sequence_params.force_kcenter, sequence_params.gradient_spoilers, sequence_params.RF_spoiler, sequence_params.pulse_type, sequence_params.RF_duration, sequence_params.balancing_gradients, sequence_params.slice_thickness)
    end

    return prep_seq
end
