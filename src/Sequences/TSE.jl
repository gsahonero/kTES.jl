export create_TSE_2D, create_TSE_1D, create_2D_seq

"""
    create_TSE_2D(sequence_params::SeqParams, sys::Scanner) -> Sequence

Construct a 2D Turbo Spin Echo (TSE / Fast Spin Echo) sequence with a multi-echo train of refocusing pulses per TR.

# Arguments
- `sequence_params::SeqParams`: Sequence configuration containing `ETL` (echo train length), `TE`, `TR`, `Nx`, `Ny`.
- `sys::Scanner`: Hardware constraints (raster time, gradient slew rate).
"""
function create_TSE_2D(sequence_params::SeqParams, sys::Scanner)
    seq = Sequence()
    seq.DEF["parameters"] = sequence_params
    if sequence_params.emulated_noise_sequence
        @warn "An emulated noise sequence will replace the provided original sequence. Reduced noise computation must be used for AcqParams struct."
        seq.DEF["parameters"].t_ADC = seq.DEF["parameters"].Nx * sys.ADC_Δt
        #acqParams.noise_technique = ReducedNoise()
    end
    seq = round_seq_parameters(seq)

    RF_phases = @MArray ones(ComplexF64, sequence_params.Ny) 

    seq.DEF["parameters"].RF_phases = RF_phases

    p90 = deepcopy(seq)
    p90.DEF["parameters"].α = 90.0
    p180 = deepcopy(seq)
    p180.DEF["parameters"].α = 180.0
    
    if sequence_params.force_kcenter
        @info "Forcing k-center to be zero."
    end

    TRs = ceil(seq.DEF["parameters"].Ny/seq.DEF["parameters"].ETL, digits = 0)
    
    for tr_i=1:TRs
        traj_i = (tr_i - 1) * seq.DEF["parameters"].ETL + 1
        if traj_i>seq.DEF["parameters"].Ny
            @error "This should not happen. Please, check the TSE sequence creation."
        end
        seq_i =  create_TSE_1D(seq, p90, p180, sys; traj_i = Int(tr_i))
        seq = seq + seq_i    
        if tr_i == TRs 
            seq.DEF["parameters"].tBASE = seq_i.DEF["tBASE"]
            seq.DEF["parameters"].ty = seq_i.DEF["ty"]
            seq.DEF["parameters"].tx = seq_i.DEF["tx"]
            seq.DEF["parameters"].tq = seq_i.DEF["tq"]
            seq.DEF["parameters"].tz = seq_i.DEF["tz"]
            seq.DEF["parameters"].t_comp = seq_i.DEF["t_comp"]
            seq.DEF["parameters"].tw = seq_i.DEF["tw"]   
            seq.DEF["parameters"].t_LR = seq_i.DEF["t_LR"]
            seq.DEF["parameters"].Δtz = seq.DEF["Δtz"]
            seq.DEF["parameters"].δtz = seq.DEF["δtz"]
            seq.DEF["parameters"].t_crusher = seq_i.DEF["t_crusher"]
        end
    end
    return check_sequence(seq, sys)
end

"""
    create_TSE_1D(sequence::Sequence, p90::Sequence, p180::Sequence, sys::Scanner; traj_i=1) -> Sequence

Construct a single TR block containing an echo train of `ETL` readouts for Turbo Spin Echo.
"""
function create_TSE_1D(sequence, p90, p180, sys; traj_i = 1)
p = sequence.DEF["parameters"]
    p90.DEF["parameters"].RF_phases[traj_i] = exp(1im*0)
    RF_subseq_90 = create_RF_pulse(p90, p.pulse_type, sys; traj_i = traj_i, use = Excitation())
    p180.DEF["parameters"].RF_phases[traj_i] =  exp(1im*π/2)
    RF_subseq_180 = create_RF_pulse(p180, p.pulse_type, sys; traj_i = traj_i, use = Refocusing())

    p180_2 = deepcopy(p180)
    p180_2.DEF["parameters"].RF_phases[traj_i] =  exp(1im*π/2)
    RF_subseq_180_2 = create_RF_pulse(p180_2, p.pulse_type, sys; traj_i = traj_i, use = Refocusing())

    p90 = p90.DEF["parameters"]
    p180 = p180.DEF["parameters"]
    slice_selection_rephase = Sequence([Grad(0e-3, p90.RF_duration, p90.t_ramp); Grad(0e-3, p90.RF_duration, p90.t_ramp); Grad(!p90.disable_slice_selection * RF_subseq_90.GR[3].A, p90.RF_duration, p90.t_ramp);;], [RF_subseq_90.RF[1];;])

    G_slice_negative_amplitude = - RF_subseq_90.GR[3].A *0.5
    if p.force_kcenter & iseven(p.Ny)
        G_phase_negative_amplitude = ((p.Ny - 1) / (2 * γ * p.FOVy)) * (1 / (p.RF_duration + p.t_ramp))
    else
        G_phase_negative_amplitude = (p.Ny / (2 * γ * p.FOVy)) * (1 / (p.RF_duration + p.t_ramp))
    end

    if p.force_kcenter & iseven(p.Nx)
        G_reph_positive_amplitude = 2 * (p.Nx - 1) / (2 * p.FOVx * γ * p.t_ADC)
    else
        G_reph_positive_amplitude = 2 * (p.Nx-1)/(2 * p.FOVx * γ * p.t_ADC)
    end

    if p.force_kcenter & iseven(p.Nx)
        G_deph_negative_amplitude = 1 / (p.RF_duration + p.t_ramp) * (G_reph_positive_amplitude / (p.Nx - 1) * (p.Nx) * p.t_ADC / 2 + G_reph_positive_amplitude * p.t_ramp / 2)
    else
        G_deph_negative_amplitude = 1 / (p.RF_duration + p.t_ramp) * (G_reph_positive_amplitude * p.t_ADC / 2 + G_reph_positive_amplitude * p.t_ramp / 2)
    end

    G_0 = (γ * (p.RF_duration * G_phase_negative_amplitude + p.t_ramp * G_phase_negative_amplitude) - 1 / p.FOVy) * 1 / (γ * (p.RF_duration + p.t_ramp))
    ΔG_y = G_phase_negative_amplitude - G_0

    # Gx deph before refocusing pulse
    slice_selection_dephase = Sequence([Grad(G_deph_negative_amplitude, p90.RF_duration, p90.t_ramp); Grad(0.0, p90.RF_duration, p90.t_ramp); Grad(!p90.disable_slice_selection * G_slice_negative_amplitude, p90.RF_duration, p90.t_ramp);;])

    slice_selection_element = slice_selection_rephase + slice_selection_dephase

    # Under the assumption that crusher gradients are equal to maximum gradient and the same as the spoiler gradient
    G_crusher_amplitude = sys.Gmax
    t_crusher_z = ((p.gradient_spoilers[3])/(p.slice_thickness * 2π * γ * G_crusher_amplitude)) - p.t_ramp      
    
    pre180_crusher = Sequence([Grad(0, t_crusher_z, p180.t_ramp); Grad(0, t_crusher_z, p180.t_ramp); Grad(G_crusher_amplitude, t_crusher_z, p180.t_ramp);;])

    post180_crusher = Sequence([Grad(0, t_crusher_z, p180.t_ramp); Grad(0, t_crusher_z, p180.t_ramp); Grad(G_crusher_amplitude, t_crusher_z, p180.t_ramp);;])

    slice_selection_rephase180 = Sequence([Grad(0e-3, p180.RF_duration, p180.t_ramp); Grad(0e-3, p180.RF_duration, p180.t_ramp); Grad(!p180.disable_slice_selection * RF_subseq_180.GR[3].A, p180.RF_duration, p180.t_ramp);;], [RF_subseq_180.RF[1];;])

    slice_selection_rephase180_2 = Sequence([Grad(0e-3, p180_2.DEF["parameters"].RF_duration, p180_2.DEF["parameters"].t_ramp); Grad(0e-3, p180_2.DEF["parameters"].RF_duration, p180_2.DEF["parameters"].t_ramp); Grad(!p180_2.DEF["parameters"].disable_slice_selection * RF_subseq_180.GR[3].A, p180_2.DEF["parameters"].RF_duration, p180_2.DEF["parameters"].t_ramp);;], [RF_subseq_180_2.RF[1];;])

    N_ADC = round(p.t_ADC / sys.ADC_Δt, digits = 3)
    if N_ADC%floor(N_ADC)!=0
        @warn "N_ADC=$N_ADC is not integer. It will be rounded, but the sequence should be checked."
        @info "For debugging: Nx=$(p.Nx), t_ADC=$(p.t_ADC)"
        N_ADC = ceil(N_ADC)
    end

    readout_element = Sequence([Grad(G_reph_positive_amplitude, p.t_ADC, p.t_ramp); Grad(0e-3, p.t_ADC, p.t_ramp); Grad(0e-3, p.t_ADC, p.t_ramp);;], [RF(0, p.t_ADC);;], [ADC(N_ADC, p.t_ADC, p.t_ramp, 0, angle(exp(1im*π/2)))])

    spoiler_element = Sequence()
    
    if sum(p.gradient_spoilers)>0
        gradient_existence = p.gradient_spoilers.>0
        spoiler_amplitude = sys.Gmax

        t_sp_x = ((p.gradient_spoilers[1]) / (p.FOVx / p.Nx * 2π * γ * spoiler_amplitude)) - p.t_ramp
        t_sp_y = ((p.gradient_spoilers[2])/(p.FOVy/p.Ny * 2π * γ * spoiler_amplitude)) - p.t_ramp
        t_sp_z = ((p.gradient_spoilers[3])/(p.slice_thickness * 2π * γ * spoiler_amplitude)) - p.t_ramp

        spoiler_element = Sequence([Grad(gradient_existence[1]*spoiler_amplitude, gradient_existence[1]*t_sp_x, p.t_ramp); Grad(gradient_existence[2]*spoiler_amplitude, gradient_existence[2]*t_sp_y, p.t_ramp); Grad(gradient_existence[3]*spoiler_amplitude, gradient_existence[3]*t_sp_z, p.t_ramp);;])
    end
    t_compensation = p180.t_ramp + t_crusher_z + p180.t_ramp + p180.t_ramp + p180.RF_duration + p180.t_ramp + p180.t_ramp + t_crusher_z + p180.t_ramp 

    t_LR = (p.EST - (2*p.t_ramp + p.t_ADC + 3*(p180.t_ramp + p180.RF_duration + p180.t_ramp) + 2*(p.t_ramp * 2 + t_crusher_z)))/2

    tx = p.ETL*(p180.t_ramp + t_crusher_z + p180.t_ramp + p180.t_ramp + p180.RF_duration + p180.t_ramp + p180.t_ramp + t_crusher_z + p180.t_ramp + t_LR + p.t_ramp + p.RF_duration + p.t_ramp + p.t_ramp + p.t_ADC + p.t_ramp + p.t_ramp + p.RF_duration + p.t_ramp + t_LR) + t_compensation # the last term provide symmetry to appropriately compute the TEeff compensation through ty and tw
    tBASE= p90.RF_duration/2 + p90.t_ramp + p90.t_ramp + p90.RF_duration + p90.t_ramp
    ty = p.EST/2 - tBASE - p180.t_ramp - t_crusher_z - p180.t_ramp - p180.t_ramp - p180.RF_duration/2
    
    ty_delay_element = Sequence([Grad(0e-3, ty, 0); Grad(0e-3, ty, 0); Grad(0e-3, ty, 0);;])
    #t_LR = t_LR - p.t_ramp - p.RF_duration - p.t_ramp # 3 last elements resemble timing of the phase rewinder
    t_LR_delay_element = Sequence([Grad(0e-3, t_LR, 0); Grad(0e-3, t_LR, 0); Grad(0e-3, t_LR, 0);;])

    seq = Sequence()
    seq = seq + slice_selection_element + ty_delay_element
    traj_x = traj_i
    compenser = ((G_reph_positive_amplitude*(p.t_ADC + p.t_ramp)) - (G_deph_negative_amplitude*(p.t_ramp + p.RF_duration)))/(p.t_ramp + p.RF_duration)
    for i=1:p.ETL
        if mod(p.Ny, p.ETL) == 0
            traj_x = ceil(p.Ny/p.ETL, digits=0) * i - (traj_i - 1)
            traj_x = mod(traj_x - 1/2*ceil(p.Ny/p.ETL, digits=0) - 1, p.Ny) + 1
        else
            traj_x = ceil(p.Ny/p.ETL, digits = 0)*i - (traj_i - 1)
        end

        """
        if traj_i<=ceil(p.Ny/p.ETL, digits=0)/2
            traj_x = p.Ny/2-(traj_i-1) - ceil((p.Ny/2)/p.ETL)*(i-1)
        else
            traj_x = p.Ny/2+(traj_i-ceil(p.Ny/p.ETL, digits=0)/2) + ceil((p.Ny/2)/p.ETL)*(i-1)
        end
        """

        #traj_x = getTSEIndex(p.ETL, p.Ny, p.sweep_mode)[Int((traj_i-1)*p.ETL+i)]

        #println("traj_i:", traj_i, " - traj_x", traj_x)
        if traj_x <= p.Ny && traj_x > 0    
            if p.force_kcenter && iseven(p.Ny)
                selected_G_phase_negative_amplitude = (p90.Ny/2 - (traj_x - 1)) * ΔG_y
                #selected_G_phase_negative_amplitude = traj_x * ΔG_y
            else
                selected_G_phase_negative_amplitude = ((p90.Ny - 1) / 2 - (traj_x - 1)) * ΔG_y     
                #selected_G_phase_negative_amplitude = traj_x * ΔG_y
           
            end       
            
            
            Gy_encoder = Sequence([Grad(-0.00*G_deph_negative_amplitude, p.RF_duration, p.t_ramp); Grad(-selected_G_phase_negative_amplitude, p.RF_duration, p.t_ramp); Grad(0, p.RF_duration, p.t_ramp);;])
            Gy_rewinder = Sequence([Grad(-0.0*compenser, p.RF_duration, p.t_ramp); Grad(selected_G_phase_negative_amplitude, p.RF_duration, p.t_ramp); Grad(0, p.RF_duration, p.t_ramp);;])

            if iseven(traj_x)
                seq = seq + pre180_crusher + slice_selection_rephase180_2 + post180_crusher + t_LR_delay_element + Gy_encoder + readout_element + Gy_rewinder + t_LR_delay_element
            else
                seq = seq + pre180_crusher + slice_selection_rephase180 + post180_crusher + t_LR_delay_element + Gy_encoder + readout_element + Gy_rewinder + t_LR_delay_element
            end
        else
            break
        end
        #traj_x += 1
    end
    #seq = seq + pre180_crusher + post180_crusher
    tw = p.TR - sum(seq.DUR)
    tw_delay_element = Sequence([Grad(0e-3, tw, 0); Grad(0e-3, tw, 0); Grad(0e-3, tw, 0);;])

    seq = seq + tw_delay_element


    seq.DEF["tBASE"] = tBASE # RF + slice selection gradients duration + Gx dephasing duration
    seq.DEF["ty"] = ty # delay before first 180
    seq.DEF["tx"] = tx # total duration of all 180s and ADCs plus delays
    seq.DEF["tq"] = tx - t_compensation # 
    seq.DEF["tz"] =tx - t_compensation - t_LR
    seq.DEF["t_LR"] = t_LR
    seq.DEF["t_crusher"] = t_crusher_z
    seq.DEF["t_comp"] = t_compensation
    seq.DEF["tw"] = p.TR - seq.DEF["tq"] - ty - tBASE
    seq.DEF["Δtz"] = p.t_ramp + t_crusher_z + p.t_ramp + p.t_ramp + p.RF_duration + p.t_ramp + p.t_ramp + t_crusher_z + p.t_ramp + t_LR + p.t_ramp + p.RF_duration + p.t_ramp + p.t_ramp + p.t_ADC + p.t_ramp + p.t_ramp + p.RF_duration + p.t_ramp + t_LR # needed for ETL optimization - time for an echo, it includes everything from the beginning of the crusher.
    seq.DEF["δtz"] = 3 * (p.t_ramp + p.RF_duration + p.t_ramp) + t_LR + p.t_ramp + p.RF_duration + p.t_ramp + p.t_ramp + p.t_ADC/2 # needed for ETL optimization - Δtz - enough time to reach the echo time of the respective echo.
    seq.DEF["parameters"] = p

    return seq
end


"""
    getIndex(etl::Int=1, nPH::Int=1, sweepMode::Int=1) -> Vector{Int32}

Generate an array representing the order to sweep k-space phase lines along an echo train length.

Arguments
- `etl`: echo train length
- `nPH`: number of phase-encoding steps
- `sweepMode`:
    0 = sequential (then reversed)
    1 = center-out
    2 = out-to-center (center-out then reversed)
    3 = Nyquist-modulated

Returns
- `Vector{Int32}` of indices (0-based line indices).
"""
function getTSEIndex(etl::Int=1, nPH::Int=1, sweepMode::Int=1)::Vector{Int32}
    n2ETL = Int(floor(nPH / 2 / etl))  # matches Python's int() truncation
    ind = Float64[]                   # build as floats like np.linspace, cast at end

    if nPH == 1
        ind = [0.0]
    else
        if sweepMode == 0
            # Sequential for T2 contrast
            for ii in 0:(Int(floor(nPH / etl)) - 1)
                # np.linspace(ii, nPH + ii, num=etl, endpoint=false)
                # For etl=2 this yields [ii, ii + nPH/2] ? No: it yields evenly spaced points.
                # General formula: start + (0:etl-1)*(stop-start)/etl
                start = float(ii)
                stop  = float(nPH + ii)
                step  = (stop - start) / etl
                append!(ind, start .+ (0:etl-1) .* step)
            end
            reverse!(ind)

        elseif sweepMode == 1
            # Center-out for T1 contrast
            if etl == nPH
                ind = zeros(Float64, nPH)
                half = Int(floor(nPH / 2))
                # ind[0::2] = linspace(nPH/2, nPH, num=nPH/2, endpoint=false)
                for k in 0:(half-1)
                    ind[2k + 1] = half + k
                end
                # ind[1::2] = linspace(nPH/2 - 1, -1, num=nPH/2, endpoint=false)
                for k in 0:(half-1)
                    ind[2k + 2] = (half - 1) - k
                end
            else
                half = Int(floor(nPH / 2))
                for ii in 0:(n2ETL - 1)
                    # upper block
                    start = float(half + ii)
                    stop  = float(nPH + ii)
                    step  = (stop - start) / etl
                    append!(ind, start .+ (0:etl-1) .* step)

                    # lower block
                    start = float(half - ii - 1)
                    stop  = float(-ii - 1)
                    step  = (stop - start) / etl
                    append!(ind, start .+ (0:etl-1) .* step)
                end
            end

        elseif sweepMode == 2
            # Out-to-center for T2 contrast (center-out then reverse)
            if etl == nPH
                ind = zeros(Float64, nPH)
                half = Int(floor(nPH / 2))
                for k in 0:(half-1)
                    ind[2k + 1] = half + k
                end
                for k in 0:(half-1)
                    ind[2k + 2] = (half - 1) - k
                end
            else
                half = Int(floor(nPH / 2))
                for ii in 0:(n2ETL - 1)
                    start = float(half + ii)
                    stop  = float(nPH + ii)
                    step  = (stop - start) / etl
                    append!(ind, start .+ (0:etl-1) .* step)

                    start = float(half - ii - 1)
                    stop  = float(-ii - 1)
                    step  = (stop - start) / etl
                    append!(ind, start .+ (0:etl-1) .* step)
                end
            end
            reverse!(ind)

        elseif sweepMode == 3
            # Nyquist modulated to reduce ghosting artifact
            if etl == nPH
                ind = collect(0.0:1.0:(nPH-1))
            else
                for ii in 0:(n2ETL - 1)
                    # np.arange(0, nPH, 2*n2ETL) + 2*ii
                    step = 2 * n2ETL
                    forward = (0:step:(nPH-1)) .+ (2 * ii)
                    append!(ind, Float64.(forward))

                    # np.arange(nPH-1, 0, -2*n2ETL) - 2*ii
                    reversepart = (nPH-1):-step:1
                    reversepart = reversepart .- (2 * ii)
                    append!(ind, Float64.(reversepart))
                end
            end
        elseif sweepMode == 4
            if etl==nPH
                ind = collect(0.0:1.0:(nPH-1))
            else
                for ii in 0:(n2ETL - 1)
                    # np.arange(0, nPH, 2*n2ETL) + 2*ii
                    step = 2 * n2ETL
                    forward = (0:step:(nPH-1)) .+ (2 * ii)
                    append!(ind, Float64.(forward))

                    # np.arange(nPH-1, 0, -2*n2ETL) - 2*ii
                    reversepart = (nPH-1):-step:1
                    reversepart = reversepart .- (2 * ii)
                    append!(ind, Float64.(reversepart))
                end
            end
        else
            error("Unsupported sweepMode: $sweepMode")
        end
    end

    # Match Python: return np.int32(ind) (truncation toward zero)
    return Int32.(trunc.(ind) .+ 1)
end


"""
    create_2D_seq(seqParams::SeqParams, seq_type::TSE, sys::Scanner) -> Sequence

Dispatch method generating a 2D Turbo Spin Echo sequence.
"""
function create_2D_seq(seqParams::SeqParams, seq_type::TSE, sys::Scanner)
    return create_TSE_2D(seqParams, sys)
end