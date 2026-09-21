export Gradients, BalancingGradients, SpoilerGradients, UnknownGradients, SequenceType, SPGR, SE, TSE, RARE_i3M, SeqParams, parameter_type, PreparationSeq, Pulses, RepeatSequence, Balanced, UnknownPreparation, RFPulse, Soft, Hard, Pulseq, UnknownRFPulse, RFPhases, PulseqPhases, QuadraticPhases, UnknownPhase

"""
    Gradients

Abstract supertype for gradient moment balancing and spoiling schemes.
"""
abstract type Gradients end

"""
    BalancingGradients <: Gradients

Fully balanced gradient waveform scheme where total gradient zeroth moment sums to zero over each TR (e.g. bSSFP).
"""
struct BalancingGradients <: Gradients end

"""
    SpoilerGradients <: Gradients

Gradient spoiling scheme applying unbalanced dephasing crusher gradients along slice, phase, or readout axes.
"""
struct SpoilerGradients <: Gradients end

"""
    UnknownGradients <: Gradients

Fallback gradient scheme type when no specific gradient pattern is recognized.
"""
struct UnknownGradients <: Gradients end

"""
    SequenceType

Abstract supertype for MRI pulse sequence families supported by the k-TES simulation framework.
"""
abstract type SequenceType end

"""
    SPGR <: SequenceType

RF-spoiled gradient echo pulse sequence (FLASH / Fast Low Angle Shot).
"""
struct SPGR <: SequenceType end

"""
    SE <: SequenceType

Spin Echo sequence with a 90° excitation pulse followed by a single 180° refocusing pulse.
"""
struct SE <: SequenceType end

"""
    TSE <: SequenceType

Turbo Spin Echo (Fast Spin Echo) sequence utilizing a train of refocusing RF pulses per TR.
"""
struct TSE <: SequenceType end

"""
    RARE_i3M <: SequenceType

Rapid Acquisition with Relaxation Enhancement (RARE) sequence variant tailored for low-field MRI.
"""
struct RARE_i3M <: SequenceType end

"""
    SeqParams(; FOVx=0.256, FOVy=0.256, Nx=128, Ny=128, seq_type=SPGR(), extras=Dict())
    SeqParams(params::Dict)

Dynamic sequence parameter dictionary wrapper supporting property access and arbitrary custom parameters.

# Fields
- `FOVx::Float64`: Field of view along readout axis in meters.
- `FOVy::Float64`: Field of view along phase encoding axis in meters.
- `Nx::Int`: Number of frequency encoding points.
- `Ny::Int`: Number of phase encoding lines.
- `seq_type::SequenceType`: Sequence category (`SPGR()`, `SE()`, `TSE()`, `RARE_i3M()`).
- `extras::Dict`: Additional pulse parameters (`TE`, `TR`, `α`, `t_ADC`, `t_ramp`, `gradient_spoilers`, etc.).
"""
@kwdef mutable struct SeqParams
    FOVx::Float64 = 0.256
    FOVy::Float64 = 0.256
    Nx::Int = 128
    Ny::Int = 128
    seq_type::SequenceType = SPGR()    
    extras::Dict = Dict()

    function SeqParams(params::Dict)
        obj = new(params["FOVx"], params["FOVy"], params["Nx"], params["Ny"], params["seq_type"], Dict())
        for (key, val) in params
            if key != "Nx" && key != "Ny" && key != "FOVx" && key != "FOVy" && key != "seq_type"
                setproperty!(obj, Symbol(key), val)
            end
        end
        return obj
    end 
end

"""
    parameter_type(p::SeqParams, name) -> DataType

Query the underlying datatype of field or extra parameter `name` inside `p`.
"""
function parameter_type(p::SeqParams, name)
    sym = Symbol(name)

    if hasfield(typeof(p), sym)
        return fieldtype(typeof(p), sym)
    else
        val = getproperty(p, sym)
        return isnothing(val) ? Nothing : typeof(val)
    end
end

function Base.getproperty(s::SeqParams, name::Symbol) 
    if hasfield(typeof(s), name) 
        getfield(s, name) 
    else
        output = get(s.extras, String(name), nothing)
        if isnothing(output)
            @error "The field $(String(name)) was not defined."
        else
            return output
        end
    end
end

function Base.setproperty!(s::SeqParams, name::Symbol, value) 
    if hasfield(typeof(s), name) 
        setfield!(s, name, value) 
    else 
        (s.extras[String(name)] = value)
    end
end

function Base.iterate(seqParams::SeqParams)
    fields = fieldnames(SeqParams)
    values = getfield.(Ref(seqParams), fields)
    return (fields[1], values[1]), 2
end

function Base.iterate(seqParams::SeqParams, state)
    fields = fieldnames(SeqParams)
    values = getfield.(Ref(seqParams), fields)
    if state > length(fields)
        return nothing
    else
        return (fields[state], values[state]), state + 1
    end
end

"""
    PreparationSeq

Abstract supertype for magnetization preparation schemes (e.g. dummy/catalyzation pulses or inversion recovery).
"""
abstract type PreparationSeq end

"""
    Pulses <: PreparationSeq

Applies a train of dummy RF excitation pulses to drive magnetization into steady state before acquisition begins.
"""
struct Pulses <: PreparationSeq end

"""
    RepeatSequence <: PreparationSeq

Repeats the entire imaging block several times to achieve steady state.
"""
struct RepeatSequence <: PreparationSeq end

"""
    Balanced <: PreparationSeq

Applies an alpha/2 catalyzation preparation pulse for balanced steady-state sequences.
"""
struct Balanced <: PreparationSeq end

"""
    UnknownPreparation <: PreparationSeq

Unspecified or custom magnetization preparation module.
"""
struct UnknownPreparation <: PreparationSeq end

"""
    RFPulse

Abstract supertype for radiofrequency pulse envelopes.
"""
abstract type RFPulse end

"""
    Soft <: RFPulse

Selective sinc or windowed-sinc RF pulse with finite duration and bandwidth for slice selection.
"""
struct Soft <: RFPulse end

"""
    Hard <: RFPulse

Non-selective rectangular RF pulse with short duration and broad excitation profile.
"""
struct Hard <: RFPulse end

"""
    Pulseq <: RFPulse

Arbitrary custom RF pulse defined through external Pulseq definitions.
"""
struct Pulseq <: RFPulse end

"""
    UnknownRFPulse <: RFPulse

Fallback type for undefined RF pulse types.
"""
struct UnknownRFPulse <: RFPulse end

"""
    RFPhases

Abstract supertype for RF transmitter phase cycle schemes.
"""
abstract type RFPhases end

"""
    PulseqPhases <: RFPhases

RF phase cycle loaded directly from a Pulseq file.
"""
struct PulseqPhases <: RFPhases end

"""
    QuadraticPhases <: RFPhases

Quadratic RF phase spoiling scheme: \$\\phi_n = \\phi_{n-1} + n \\Delta\\phi\$ (typically \$\\Delta\\phi = 117^\\circ\$)
to effectively eliminate transverse steady-state coherence.
"""
struct QuadraticPhases <: RFPhases end

"""
    UnknownPhase <: RFPhases

Fallback type for custom or unrecognized RF phase patterns.
"""
struct UnknownPhase <: RFPhases end
