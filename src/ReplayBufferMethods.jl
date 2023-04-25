export f # dummy function

export Transition
export ReplayBuffer
export BufferPush!, BufferSample, BufferLength

using StatsBase

struct Transition
    s::Vector{Float64}
    a::Int64
    r::Float64
    s_::Vector{Float64}
    d::Bool
end

struct ReplayBuffer
    buffer::Vector{Transition}
    capacity::Int64
end

function BufferPush!(buffer::ReplayBuffer, transition::Transition)
    push!(buffer.buffer, transition)
    if length(buffer.buffer) > buffer.capacity
        popfirst!(buffer.buffer)
    end
end

function BufferSample(buffer::ReplayBuffer, batch_size::Int64)
    return StatsBase.sample(buffer.buffer, batch_size, replace=false)
end

function BufferLength(buffer::ReplayBuffer)
    return length(buffer.buffer)
end

function f(x)
    return x^2
end