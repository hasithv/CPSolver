using CPSolver
using Test

using Flux
using ReinforcementLearning

@testset "CPSolver.jl" begin
    # ReplayBufferMethods tests
    @test CPSolver.f(3) == 9
    @test f(3) == 9

    env = CartPoleEnv()
    reset!(env)
    s = deepcopy(env.state)
    @test s isa Vector{Float64}

    a = rand(1:2)
    env(a)
    t = Transition(s, a, 1, env.state, is_terminated(env))
    @test t isa Transition
    
    buffer = ReplayBuffer([], 10)
    @test BufferLength(buffer) == 0

    for i in 1:100
        s = deepcopy(env.state)
        a = rand(1:2)
        env(a)
        BufferPush!(buffer, Transition(s, a, 1, deepcopy(env.state), is_terminated(env)))
    end
    @test BufferLength(buffer) == 10
    @test BufferSample(buffer, 1) isa Vector{Transition}
    @test buffer.buffer[1].s != buffer.buffer[1].s_
    @test buffer.buffer[1].s_ == buffer.buffer[2].s

    @test buffer.buffer[1].s != buffer.buffer[2].s
    @test buffer.buffer[1].s_ != buffer.buffer[2].s_
    @test !(sum([t.a for t in buffer.buffer]) in (10, 20))

    # DQN tests
    model = Chain(Dense(4, 16, relu), Dense(16, 2))
    dqn = DQN(2, 4, model)
    @test dqn isa DQN
    @test dqn.model([1,1,1,1]) isa Vector{Float32}

    # training tests
    params = DefaultTrainParams()
    @test params isa TrainParams
    @test params.policyNet isa DQN
    @test params.targetNet isa DQN
    @test params.buffer isa ReplayBuffer

    reset!(env)
    @test env.state isa Vector{Float64}
    @test SelectAction(params, env.state, 0) in [1,2]

    buffer = ReplayBuffer([], 100)
    for i in 1:1000
        s = deepcopy(env.state)
        a = rand(1:2)
        env(a)
        BufferPush!(buffer, Transition(s, a, 1, deepcopy(env.state), is_terminated(env)))
    end
    batch = BufferSample(buffer, 32)
    old_params = deepcopy(params.policyNet)
    PolicyUpdate!(params, batch)
    @test params.policyNet != old_params
end
