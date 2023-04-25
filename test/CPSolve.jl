using CPSolver

using Flux
using ReinforcementLearning
using ProgressBars
using Plots

gr(show = true)

env = CartPoleEnv()
tparams = TrainParams(600, 500, .9, 1000, .05, .99, 1e-4, 128, .005, 
                        DQN(2, 4, Chain(Dense(4, 128, relu), Dense(128, 128), Dense(128, 2))), 
                        DQN(2, 4, Chain(Dense(4, 128, relu), Dense(128, 128), Dense(128, 2))), 
                        ReplayBuffer([], 10000))


stepsDone = 0
episodeSteps = []

iter = ProgressBar(1:tparams.nEpisodes)
for episode in iter
    reset!(env)
    t = 0

    s = deepcopy(env.state)
    for i in 1:tparams.maxSteps
        a = SelectAction(tparams, s, stepsDone)
        r = reward(env)
        env(a)
        BufferPush!(tparams.buffer, Transition(s, a, r, 
                    deepcopy(env.state), is_terminated(env)))
        s = deepcopy(env.state)
        global stepsDone += 1
        t += 1

        if BufferLength(tparams.buffer) > tparams.batchSize
            batch = BufferSample(tparams.buffer, tparams.batchSize)
            PolicyUpdate!(tparams, batch)
            TargetSoftUpdate!(tparams)
        end
        if is_terminated(env)
            push!(episodeSteps, t)
            display(plot(episodeSteps))
            break
        end
    end
    set_description(iter, string("t: $t"))
end

print(episodeSteps)
