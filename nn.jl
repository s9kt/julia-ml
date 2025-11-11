"""
Neural Network Building Blocks
Part 2: Module Base Class, Layers, and Containers
"""

using Random
using Statistics

include("tensor.jl")

# ============================================================================
# MODULE BASE CLASS (ABSTRACT)
# ============================================================================

abstract type Module end

# Core interface - must be implemented by subtypes
function forward(m::Module, input::Tensor)
    error("forward() must be implemented by subtype $(typeof(m))")
end

# Make modules callable
(m::Module)(input::Tensor) = forward(m, input)

# Collect all parameters recursively
function parameters(m::Module)
    params = Tensor[]
    for field in fieldnames(typeof(m))
        val = getfield(m, field)
        if val isa Tensor
            push!(params, val)
        elseif val isa Module
            append!(params, parameters(val))
        elseif val isa Vector && !isempty(val) && val[1] isa Module
            for submodule in val
                append!(params, parameters(submodule))
            end
        end
    end
    return params
end

function zero_grad!(m::Module)
    for param in parameters(m)
        param.grad .= 0.0
    end
end

# Training/eval mode tracking
const MODULE_STATES = IdDict{Module, Bool}()  # true = training, false = eval

function train!(m::Module)
    MODULE_STATES[m] = true
    for field in fieldnames(typeof(m))
        val = getfield(m, field)
        if val isa Module
            train!(val)
        elseif val isa Vector && !isempty(val) && val[1] isa Module
            for submodule in val
                train!(submodule)
            end
        end
    end
end

function eval!(m::Module)
    MODULE_STATES[m] = false
    for field in fieldnames(typeof(m))
        val = getfield(m, field)
        if val isa Module
            eval!(val)
        elseif val isa Vector && !isempty(val) && val[1] isa Module
            for submodule in val
                eval!(submodule)
            end
        end
    end
end

is_training(m::Module) = get(MODULE_STATES, m, true)

# ============================================================================
# LINEAR LAYER
# ============================================================================

mutable struct Linear <: Module
    weight::Tensor
    bias::Union{Tensor, Nothing}
    in_features::Int
    out_features::Int
    
    function Linear(in_features::Int, out_features::Int; use_bias::Bool=true)
        # Xavier/He initialization
        std = sqrt(2.0 / (in_features + out_features))
        weight = Tensor(randn(out_features, in_features) .* std, 
                       requires_grad=true, name="weight")
        
        bias = use_bias ? Tensor(zeros(1, out_features), requires_grad=true, name="bias") : nothing
        
        new(weight, bias, in_features, out_features)
    end
end

function forward(layer::Linear, x::Tensor)
    output = matmul(x, transpose(layer.weight))
    return layer.bias !== nothing ? output + layer.bias : output
end

# ============================================================================
# ACTIVATION LAYERS
# ============================================================================

struct ReLU <: Module end
forward(::ReLU, x::Tensor) = relu(x)

struct Sigmoid <: Module end
forward(::Sigmoid, x::Tensor) = sigmoid(x)

struct Tanh <: Module end
forward(::Tanh, x::Tensor) = tanh(x)

mutable struct Softmax <: Module
    dims::Int
    Softmax(; dims::Int=1) = new(dims)
end
forward(layer::Softmax, x::Tensor) = softmax(x, dims=layer.dims)

# ============================================================================
# CONTAINER MODULES
# ============================================================================

mutable struct Sequential <: Module
    layers::Vector{Module}
    Sequential(layers::Vector{<:Module}) = new(layers)
    Sequential(layers::Module...) = new(collect(layers))
end

function forward(seq::Sequential, x::Tensor)
    for layer in seq.layers
        x = layer(x)
    end
    return x
end

Base.push!(seq::Sequential, layer::Module) = push!(seq.layers, layer)

mutable struct ModuleList <: Module
    modules::Vector{Module}
    ModuleList(modules::Vector{<:Module}=Module[]) = new(modules)
    ModuleList(modules::Module...) = new(collect(modules))
end

Base.push!(ml::ModuleList, mod::Module) = push!(ml.modules, mod)
Base.getindex(ml::ModuleList, i::Int) = ml.modules[i]
Base.length(ml::ModuleList) = length(ml.modules)

# ============================================================================
# REGULARIZATION LAYERS
# ============================================================================

mutable struct Dropout <: Module
    p::Float64
    
    function Dropout(p::Float64=0.5)
        @assert 0.0 <= p < 1.0 "Dropout probability must be in [0, 1)"
        new(p)
    end
end

function forward(layer::Dropout, x::Tensor)
    if !is_training(layer)
        return x
    end
    
    # Inverted dropout
    mask = rand(Float64, size(x)) .> layer.p
    scale = 1.0 / (1.0 - layer.p)
    
    return x * Tensor(mask .* scale, requires_grad=false)
end

mutable struct BatchNorm1D <: Module
    num_features::Int
    momentum::Float64
    epsilon::Float64
    gamma::Tensor
    beta::Tensor
    running_mean::Vector{Float64}
    running_var::Vector{Float64}
    num_batches_tracked::Int
    
    function BatchNorm1D(num_features::Int; momentum::Float64=0.1, epsilon::Float64=1e-5)
        gamma = Tensor(ones(num_features), requires_grad=true, name="gamma")
        beta = Tensor(zeros(num_features), requires_grad=true, name="beta")
        new(num_features, momentum, epsilon, gamma, beta, 
            zeros(Float64, num_features), ones(Float64, num_features), 0)
    end
end

function forward(layer::BatchNorm1D, x::Tensor)
    if is_training(layer)
        batch_mean = mean(x, dims=1, keepdims=true)
        batch_var = mean((x - batch_mean) ^ 2, dims=1, keepdims=true)
        
        layer.running_mean .= (1 - layer.momentum) .* layer.running_mean .+ 
                              layer.momentum .* vec(batch_mean.data)
        layer.running_var .= (1 - layer.momentum) .* layer.running_var .+ 
                             layer.momentum .* vec(batch_var.data)
        layer.num_batches_tracked += 1
        
        x_normalized = (x - batch_mean) / (batch_var + layer.epsilon) ^ 0.5
    else
        running_mean_t = Tensor(reshape(layer.running_mean, 1, :), requires_grad=false)
        running_var_t = Tensor(reshape(layer.running_var, 1, :), requires_grad=false)
        x_normalized = (x - running_mean_t) / (running_var_t + layer.epsilon) ^ 0.5
    end
    
    # Apply learnable scale and shift
    gamma_reshaped = reshape(layer.gamma.data, 1, :)
    beta_reshaped = reshape(layer.beta.data, 1, :)
    
    return Tensor(gamma_reshaped .* x_normalized.data .+ beta_reshaped,
                 requires_grad=x.requires_grad)
end

# ============================================================================
# UTILITIES
# ============================================================================

function summary(m::Module)
    println("="^60)
    println("Model Summary")
    println("="^60)
    
    total_params = 0
    trainable_params = 0
    
    for (i, param) in enumerate(parameters(m))
        num_params = length(param.data)
        total_params += num_params
        trainable_params += param.requires_grad ? num_params : 0
        
        name = isempty(param.name) ? "param_$i" : param.name
        println("$name: $(size(param)) - $num_params parameters")
    end
    
    println("="^60)
    println("Total parameters: $total_params")
    println("Trainable parameters: $trainable_params")
    println("Non-trainable parameters: $(total_params - trainable_params)")
    println("="^60)
end

# Exports
export Module, Linear, ReLU, Sigmoid, Tanh, Softmax
export Sequential, ModuleList
export Dropout, BatchNorm1D
export forward, parameters, zero_grad!, train!, eval!, is_training, summary

println("✓ Part 2: Neural Network Building Blocks loaded successfully")