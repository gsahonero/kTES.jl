export hard_RF_duration, create_RF_pulse, create_RF_phases!

"""
    hard_RF_duration(angle::Real, amp_RF::Real) -> Float64

Calculate the duration of a rectangular (hard) RF pulse in seconds given excitation flip angle `angle` (in degrees)
and RF peak amplitude `amp_RF` (in Tesla):
```math
t = \\frac{\\alpha \\times (\\pi / 180)}{2\\pi \\gamma B_1}
```
"""
function hard_RF_duration(angle, amp_RF)
    return (angle / 180 * π) / (2 * π * γ * amp_RF)
end

"""
    create_RF_pulse(sequence::Sequence, PulseType::Soft, sys::Scanner; traj_i=1, use=nothing) -> Sequence

Construct a slice-selective sinc RF pulse sub-sequence according to parameters defined in `sequence.DEF["parameters"]`.
"""
function create_RF_pulse(sequence::Sequence, PulseType::Soft, sys::Scanner; traj_i = 1, use = nothing)
    TBP = 4 # Conventional definition
    BW_rf = TBP / sequence.DEF["parameters"].RF_duration # In Hz
    Gz = BW_rf / (γ * sequence.DEF["parameters"].slice_thickness)
    B1_rf = (sequence.DEF["parameters"].α / 180 * π) * TBP / (γ * 2 * π * sequence.DEF["parameters"].RF_duration)
    RF_seq = PulseDesigner.RF_sinc(sequence.DEF["parameters"].RF_phases[traj_i] * B1_rf, sequence.DEF["parameters"].RF_duration, sys; G = [0, 0, Gz], TBP = TBP)
    RF_seq.RF[1].delay = sequence.DEF["parameters"].t_ramp
    
    if !isnothing(use)
        RF_seq.RF[1].use = use
    end
    
    return RF_seq
end

function create_RF_pulse(RF_duration, slice_thickness, TBP, RF_phases,  PulseType::Soft, sys::Scanner; traj_i = 1, use = nothing)
    TBP = 4 # Conventional definition
    BW_rf = TBP / sequence.DEF["parameters"].RF_duration # In Hz
    Gz = BW_rf / (γ * sequence.DEF["parameters"].slice_thickness)
    B1_rf = (sequence.DEF["parameters"].α / 180 * π) * TBP / (γ * 2 * π * sequence.DEF["parameters"].RF_duration)
    RF_seq = PulseDesigner.RF_sinc(sequence.DEF["parameters"].RF_phases[traj_i] * B1_rf, sequence.DEF["parameters"].RF_duration, sys; G = [0, 0, Gz], TBP = TBP)
    RF_seq.RF[1].delay = sequence.DEF["parameters"].t_ramp
    
    if !isnothing(use)
        RF_seq.RF[1].use = use
    end
    
    return RF_seq
end

"""
    create_RF_pulse(sequence::Sequence, PulseType::Hard, sys::Scanner; traj_i=1, use=nothing) -> Sequence

Construct a non-selective rectangular (hard) RF pulse sub-sequence.
"""
function create_RF_pulse(sequence::Sequence, PulseType::Hard, sys::Scanner; traj_i = 1, use = nothing)
    B1_rf = (sequence.DEF["parameters"].α * π / 180)/(2 * π * γ * sequence.DEF["parameters"].RF_duration)
    if B1_rf>sys.B1
        @warn "Scanner's B1 exceeded. The required B1 is $B1_rf, but the scanner can only provide $sys.B1"
    end
    # Gz is usually zero when hard pulses are applied
    RF_seq = PulseDesigner.RF_hard(sequence.DEF["parameters"].RF_phases[traj_i] * B1_rf, sequence.DEF["parameters"].RF_duration, sys, G = [0, 0, 0])
    RF_seq.RF[1].delay = sequence.DEF["parameters"].t_ramp
    
    if !isnothing(use)
        RF_seq.RF[1].use = use
    end

    return RF_seq
end

"""
    create_RF_pulse(sequence::Sequence, PulseType::Pulseq, sys::Scanner; traj_i=1, use=nothing) -> Sequence

Construct an RF pulse defined via custom windowed pulseq specifications.
"""
function create_RF_pulse(sequence::Sequence, PulseType::Pulseq, sys::Scanner; traj_i = 1, use = nothing)
    # TODO needs to be worked out
    TBP = 4
    BW_rf = TBP / sequence.DEF["parameters"].RF_duration;
    alpha = 0.42
    N = round(sequence.DEF["parameters"].RF_duration/sys.RF_Δt);
    t = ((1:N) .- 0.5) * sys.RF_Δt;
    t = round.(t, digits = 8)
    tt = t .- sequence.DEF["parameters"].RF_duration*0.5;
    tt = round.(tt, digits = 8)
    window = (1.0 - alpha .+ alpha * cos.(2 * π * tt / sequence.DEF["parameters"].RF_duration));
    signal = window .* sinc.(BW_rf * tt);
    flip = sum(signal) * sys.RF_Δt * 2 * π;
    signal = signal*sequence.DEF["parameters"].α / 180 * π / flip;
    signal /= γ
    #BW_rf = 2e3; # In Hz
    Gz = BW_rf / (γ * 2 * π * sequence.DEF["parameters"].slice_thickness)
    RF_pulse = RF(sequence.DEF["parameters"].RF_phases[traj_i] * signal[1:end], sequence.DEF["parameters"].RF_duration, angle(sequence.DEF["parameters"].RF_phases[traj_i]), sequence.DEF["parameters"].t_ramp)
    RF_seq = Sequence([Grad(0e-3, sequence.DEF["parameters"].RF_duration, sequence.DEF["parameters"].t_ramp); Grad(0e-3, sequence.DEF["parameters"].RF_duration, sequence.DEF["parameters"].t_ramp); Grad(Gz, sequence.DEF["parameters"].RF_duration, sequence.DEF["parameters"].t_ramp);;], [RF_pulse;;])
    
    if !isnothing(use)
        RF_seq.RF[1].use = use
    end
    return RF_seq
end

"""
    create_RF_phases!(RF_spoilers, phase_technique::PulseqPhases, RF_spoiler_seed::Float64)

Populate `RF_spoilers` vector with incremental Pulseq-style phase increments.
"""
function create_RF_phases!(RF_spoilers, phase_technique::PulseqPhases, RF_spoiler_seed::Float64)
    rf_phase = 0.0
    rf_inc = 0.0
    for i in eachindex(RF_spoilers)
        rf_inc = mod(rf_inc + RF_spoiler_seed * 180 / π, 360.0)
        rf_phase = mod(rf_phase + rf_inc, 360) / 180 * π
        RF_spoilers[i] = exp(1im * rf_phase)
    end
end

"""
    create_RF_phases!(RF_spoilers, phase_technique::QuadraticPhases, RF_spoiler_seed::Float64)

Compute quadratic RF spoiling phase factors according to Zur et al.:
```math
\\phi_n = \\frac{1}{2} \\phi_{\\text{seed}} (n^2 + n + 2)
```
populating the array with complex phasors `exp(1im * ϕ_n)`.
"""
function create_RF_phases!(RF_spoilers, phase_technique::QuadraticPhases, RF_spoiler_seed::Float64)
    for i in eachindex(RF_spoilers)
        RF_spoilers[i] = exp(1im * 1 / 2 * RF_spoiler_seed * ((i-1)^2+ (i-1) + 2))
    end
end