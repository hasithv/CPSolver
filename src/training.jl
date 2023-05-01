export TrainParams, DefaultTrainParams
export SelectAction, PolicyUpdate!, TargetSoftUpdate!

using Flux

include("DQN.jl")
include("ReplayBufferMethods.jl")

mutable struct TrainParams
    nEpisodes::Int64
    maxSteps::Int64
    epsilon::Float64
    epsilonDecay::Float64
    epsilonMin::Float64
    gamma::Float64
    learningRate::Float64
    batchSize::Int64
    targetUpdate::Float64

    policyNet::DQN
    targetNet::DQN
    buffer::ReplayBuffer
end

DefaultTrainParams() = TrainParams(100, 500, .9, 1000, .05, .99, 1e-4, 32, .005, 
                            DQN(2, 4, Chain(Dense(4, 16, relu), Dense(16, 2))), 
                            DQN(2, 4, Chain(Dense(4, 16, relu), Dense(16, 2))), 
                            ReplayBuffer([], 10000))

function SelectAction(tparams::TrainParams, state, steps_done)
    sample = rand()
    eps_threshold = tparams.epsilonMin + (tparams.epsilon - tparams.epsilonMin) * exp(-1. * steps_done / tparams.epsilonDecay)
    if sample > eps_threshold
        return argmax(tparams.policyNet.model(state))
    else
        return rand(1:2)
    end
end

function PolicyUpdate!(tparams::TrainParams, batch::Vector{Transition}; loss = Flux.huber_loss)
    b = length(batch)
    states = Vector{Vector{Float64}}(undef, b)
    actions = Vector{Int64}(undef, b)
    rewards = Vector{Float64}(undef, b)
    nextStates = Vector{Vector{Float64}}(undef, b)
    final = Vector{Bool}(undef, b)

    for i in 1:length(batch)
        states[i] = batch[i].s
        actions[i] = batch[i].a
        rewards[i] = batch[i].r
        nextStates[i] = batch[i].s_
        final[i] = batch[i].d
    end

    stateActionValues = [tparams.policyNet.model(Flux.stack(states, dims=2))[actions[i], i] for i in 1:b]
    nextStateValues = [maximum(tparams.targetNet.model(Flux.stack(nextStates, dims=2))[:, i]) for i in 1:b] .* .!final

    expectedStateActionValues = nextStateValues * tparams.gamma .+ rewards

    opt = Flux.Optimiser(ClipValue(100), AMSGrad(tparams.learningRate))
    ps = Flux.params(tparams.policyNet.model)
    gs = gradient(() -> loss(stateActionValues, expectedStateActionValues), ps)
    Flux.Optimise.update!(opt, ps, gs)
end

function TargetSoftUpdate!(tparams::TrainParams)
    for (t, p) in zip(Flux.params(tparams.targetNet.model), Flux.params(tparams.policyNet.model))
        t = (1 - tparams.targetUpdate) .* t .+ tparams.targetUpdate .* p
    end
end

