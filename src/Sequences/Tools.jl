export apply_ADC_to_seq, extract_seqParams, estimate_t_ramp, check_sequence, print_seq_parameters, round_seq_parameters, create_GE_2D, combine_gradients, get_FOV, compute_maximum_OF, create_FID_sequence, area_inspection

"""
    apply_ADC_to_seq(seq::Sequence) -> Sequence

Utility for sequence debugging: applies an active ADC readout block to every block in `seq`.
"""
function apply_ADC_to_seq(seq::Sequence)
    temp_seq = deepcopy(seq)
    for (i, block) in enumerate(temp_seq)
        T = block.DUR
        N = Int(ceil(T[1]/sys.ADC_Δt, digits=0))
        temp_seq[i].ADC[1].N = N 
        temp_seq[i].ADC[1].T = T[1]
        temp_seq[i].ADC[1].delay = 0.0
    end
    return temp_seq
end

"""
    extract_seqParams(seq::Sequence, sys::Scanner) -> SeqParams

Extract and construct a [`SeqParams`](@ref) object from an existing KomaMRI `Sequence`.
"""
function extract_seqParams(seq::Sequence, sys::Scanner)
    FOVx = haskey(seq.DEF["parameters"], "FOVx") ? seq.DEF["parameters"]["FOVx"] : (@info("FOVx not found in DEF. Using default 0.35"); 0.35)
    FOVy = haskey(seq.DEF["parameters"], "FOVy") ? seq.DEF["parameters"]["FOVy"] : (@info("FOVy not found in DEF. Using default 0.35"); 0.35)
    Nx = haskey(seq.DEF["parameters"], "Nx") ? seq.DEF["parameters"]["Nx"] : (@info("Nx not found in DEF. Using default 128"); 128)
    Ny = haskey(seq.DEF["parameters"], "Ny") ? seq.DEF["parameters"]["Ny"] : (@info("Ny not found in DEF. Using default 128"); 128)
    t_ADC = haskey(seq.DEF["parameters"], "t_ADC") ? seq.DEF["parameters"]["t_ADC"] : error("t_ADC not found in DEF. It needs to be specified.")
    TE = haskey(seq.DEF["parameters"], "TE") ? seq.DEF["parameters"]["TE"] : (@info("TE not found in DEF. Using default 25e-3"); 25e-3)
    TR = haskey(seq.DEF["parameters"], "TR") ? seq.DEF["parameters"]["TR"] : (@info("TR not found in DEF. Using default 500e-3"); 500e-3)
    α = haskey(seq.DEF["parameters"], "FA") ? seq.DEF["parameters"]["FA"] : (@info("α not found in DEF. Using default 45"); 45)
    if haskey(seq.DEF["parameters"], "seq_type")
        if seq.DEF["parameters"]["seq_type"] == "SPGR"
            seq_type = SPGR()
        elseif seq.DEF["parameters"]["seq_type"] == "SE"
            seq_type = SE()
        elseif seq.DEF["parameters"]["seq_type"] == "TSE"
            seq_type = TSE()
        else
            @warn("Unknown seq_type in DEF. Using default SPGR")
            seq_type = SPGR()
        end
    else
        @info("seq_type not found in DEF. Using default SPGR")
        seq_type = SPGR()
    end
    if t_ADC/Nx > sys.RF_Δt
        emulated_noise_sequence = true
        OF = round(Int, t_ADC/(Nx * sys.ADC_Δt))
        @info("Creating an emulated noise sequence with OF = $OF")
    else
        emulated_noise_sequence = false
        OF = 1
    end
    p = SeqParams(α, TE, TR, t_ADC, FOVx, FOVy, Int(Nx), Int(Ny), 1, 100, Pulses(), false, true, [8π, 8π, 8π], 117 / 180 * π, ComplexF64.([0]), QuadraticPhases(), Soft(), 2.3e-3, [false, false, false], 3e-3, 0.0, 0.0, 0.0, true, 1)

    return p
end

"""
    estimate_t_ramp(A::Real, S::Real) -> Float64

Calculate the minimum gradient ramp time in seconds given target gradient amplitude `A` (in T/m)
and slew rate `S` (in mT/m/ms = T/m/s).
"""
function estimate_t_ramp(A, S)
    t_ramp = (A*1e3)/(S*1e3)
    return t_ramp
end

"""
    check_sequence(seq::Sequence, sys::Scanner; visual_inspection=true) -> Union{Sequence, Bool}

Verify that sequence gradient amplitudes do not exceed `sys.Gmax` and that block durations are strictly non-negative.
Returns `seq` if valid, or `false` (with diagnostic error messages) if constraints are violated.
"""
function check_sequence(seq::Sequence, sys::Scanner; visual_inspection = true)    
    local flag = false
    if sum(abs.(seq.GR[:, :].A).>sys.Gmax)>0
        flag = true
        G_error = seq.GR[:, findall(x->abs(x)>sys.Gmax, vec(seq.GR[:, :].A))[1]].A
        @error "A gradient amplitude ($G_error) exceeded sys.Gmax. Please, check the sequence."
    end
    if sum(seq.DUR.<0)>0
        flag = true
        i = findall(<(0), seq.DUR)
        print_seq_parameters(seq)
        @error "The sequence has negative timings. Please, check the sequence at block $i."        
    end
    if flag 
        if visual_inspection 
            display(plot_seq(seq))
            display(plot_seq(seq[end-100:end]))
        end
        #print_seq_parameters(seq)
        return false
    else
        return seq
    end
end

"""
    print_seq_parameters(sequence::Sequence)

Print all parameters stored in `sequence.DEF["parameters"]`.
"""
function print_seq_parameters(sequence::Sequence)
    for i in sequence.DEF["parameters"]
        println(i[1], " = ", i[2])
    end
end

"""
    round_seq_parameters(seq::Sequence) -> Sequence

Round all floating-point timing and amplitude fields in `seq.DEF["parameters"]` to 9 decimal places to eliminate floating-point precision artifacts.
"""
function round_seq_parameters(seq::Sequence)
    for param in seq.DEF["parameters"]
        if param[2] isa Real
            field_type = fieldtype(typeof(seq.DEF["parameters"]), Symbol(param[1]))
            setfield!(seq.DEF["parameters"], Symbol(param[1]), convert(field_type, round(param[2], digits = 9)))
        end
    end
    return seq
end

"""
    create_GE_2D(sequence_params::SeqParams, sys::Scanner) -> Sequence

Construct a complete 2D Gradient Echo (GE) pulse sequence including RF excitation, slice selection,
phase encoding, readout, rewinder/balancing, and spoiler gradients across all phase encoding lines.
"""
function create_GE_2D(sequence_params::SeqParams, sys::Scanner)
    seq = Sequence()

    seq.DEF["parameters"] = sequence_params
    if sequence_params.emulated_noise_sequence
        @warn "An emulated noise sequence will replace the provided original sequence. Reduced noise computation must be used for AcqParams struct."
        #seq.DEF["parameters"].t_ADC = seq.DEF["parameters"].Nx * sys.ADC_Δt
        #acqParams.noise_technique = ReducedNoise()
    end
    seq = round_seq_parameters(seq)

    RF_spoilers = @MArray zeros(ComplexF64, sequence_params.Ny)
    create_RF_phases!(RF_spoilers, sequence_params.phases_technique, sequence_params.RF_spoiler_seed)
    seq.DEF["parameters"].RF_phases = RF_spoilers

    # Just to let the user know - TODO maybe a function to emit all the info, warnings, and stuff?
    if sequence_params.force_kcenter
        @info "Forcing k-center to be zero."
    end
    

    # For each phase:
    for i=1:sequence_params.Ny        
        seq_i =  create_GE_1D(seq, sys; traj_i = i)
        seq = seq + seq_i
    end

    # Including the preparation sequence
    prep_seq = create_preparation_sequence(seq, sequence_params.preparation_technique, sys)
  
    seq = prep_seq + seq
    
    

    return check_sequence(seq, sys)
end

"""
    combine_gradients(Sequences::Vector{Sequence}) -> Sequence

Combine individual physical gradient axes (x, y, z) into a single composite sequence.
"""
function combine_gradients(Sequences::Vector{Sequence})
    seq = Sequence([Grad(Sequences[1].GR[1].A, Sequences[1].GR[1].T, Sequences[1].GR[1].rise, Sequences[1].GR[1].fall, Sequences[1].GR[1].delay); Grad(Sequences[2].GR[2].A, Sequences[2].GR[2].T, Sequences[2].GR[2].rise, Sequences[2].GR[2].fall, Sequences[2].GR[2].delay); Grad(Sequences[3].GR[3].A, Sequences[3].GR[3].T, Sequences[3].GR[3].rise, Sequences[3].GR[3].fall, Sequences[3].GR[3].delay);;])    
    return seq
end

"""
    get_FOV(sequence::Sequence) -> (FOVx, FOVy)

Calculate the effective Cartesian field of view in meters directly from sequence k-space point spacing.
"""
function get_FOV(sequence::Sequence)
    _, kspace = get_kspace(sequence)
    FOVx = 1/abs(kspace[1, 1]-kspace[2, 1])
    ky = unique(kspace[:, 2])
    FOVy = 1/abs(ky[1] - ky[2])
    return FOVx, FOVy
end

"""
    compute_maximum_OF(sequence::Sequence, sys::Scanner) -> Int

Compute the maximum permissible oversampling factor without exceeding gradient raster timing constraints.
"""
function compute_maximum_OF(sequence::Sequence, sys::Scanner)
    t_c, t_ramp = sequence.DEF["parameters"].t_c, sequence.DEF["parameters"].t_ramp

    t_a = minimum([sequence.DEF["parameters"].TR - sequence.DEF["parameters"].t_f - sequence.DEF["parameters"].TE, sequence.DEF["parameters"].TE - t_c]) - t_ramp

    OF = floor((t_a) / (sys.ADC_Δt * sequence.DEF["parameters"].Nx/2))

    return Int(OF)
end

"""
    create_FID_sequence(N_ADC::Int, sys::Scanner) -> Sequence

Construct a simple Free Induction Decay (FID) sequence comprising a hard RF pulse followed by an ADC readout block.
"""
function create_FID_sequence(N_ADC, sys)
    seq = Sequence()
    seq += PulseDesigner.RF_hard(10e-6, 1e-3, sys, G= [0.0, 0.0, 0.0])
    seq += Sequence()
    t_ramp = sys.Gmax/sys.Smax
    t_ADC = N_ADC * sys.ADC_Δt
    seq += Sequence([Grad(0e-3, t_ADC, t_ramp); Grad(0e-3, t_ADC, t_ramp); Grad(0e-3,t_ADC, t_ramp);;], [RF(0, t_ADC);;], [ADC(N_ADC, t_ADC, t_ramp, 0, 0)])
    return seq
end

"""
    area_inspection(seq::Sequence, start::Int, stop::Int, g_index::Int) -> Float64

Calculate the zeroth gradient moment (area in T/m * s) under axis `g_index` (1=x, 2=y, 3=z) between blocks `start` and `stop`.
"""
function area_inspection(seq::Sequence, start::Int, stop::Int, g_index::Int)
    A = 0.0
    for block in seq[start:stop]
        A = A + block.GR[g_index].A*block.GR[g_index].T+block.GR[g_index].rise*block.GR[g_index].A/2+block.GR[g_index].fall*block.GR[g_index].A/2
    end
    display(plot_seq(seq[start:stop]))
    return A
end