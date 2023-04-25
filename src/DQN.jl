export DQN

using Flux

struct DQN
    nActions::Int64
    nObservations::Int64
    model::Chain
end
