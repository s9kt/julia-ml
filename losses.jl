"""
Loss Functions for Machine Learning
Part 3: Regression, Classification, and Physics-Informed Losses
"""

using Statistics
include("tensor.jl")
include("nn.jl")

# ============================================================================
# REGRESSION LOSSES
# ============================================================================

"""
Mean Squared Error loss
L = mean((predictions - targets)^2)
"""
function mse_loss(predictions::Tensor, targets::Tensor)::Tensor
    diff = predictions - targets
    squared = diff ^ 2
    return mean(squared)
end

"""
Mean Absolute Error loss  
L = mean(|predictions - targets|)
"""
function mae_loss(predictions::Tensor, targets::Tensor)::Tensor
    diff = predictions - targets
    absolute = abs.(diff)
    return mean(absolute)
end

# ============================================================================
# CLASSIFICATION LOSSES
# ============================================================================

"""
Binary Cross-Entropy loss (expects probabilities)
L = -[y*log(p) + (1-y)*log(1-p)]
"""
function binary_cross_entropy(predictions::Tensor, targets::Tensor; epsilon::Float64=1e-7)::Tensor
    # Clip predictions for numerical stability
    predictions_clipped = Tensor(clamp.(predictions.data, epsilon, 1.0 - epsilon), 
                               requires_grad=predictions.requires_grad)
    
    term1 = targets * log(predictions_clipped)
    term2 = (1.0 - targets) * log(1.0 - predictions_clipped)
    
    return -mean(term1 + term2)
end

"""
Cross-Entropy loss with logits (more numerically stable)
Combines softmax + negative log likelihood
targets: integer class indices (batch_size,)
logits: (batch_size, num_classes)
"""
function cross_entropy_loss(logits::Tensor, targets::Tensor; epsilon::Float64=1e-7)::Tensor
    # Numerically stable log-softmax
    max_logits = maximum(logits.data, dims=2)
    shifted_logits = Tensor(logits.data .- max_logits, requires_grad=logits.requires_grad)
    
    exp_vals = exp.(shifted_logits)
    log_sum_exp = log.(sum(exp_vals.data, dims=2) .+ epsilon)
    log_probs = Tensor(shifted_logits.data .- log_sum_exp, requires_grad=logits.requires_grad)
    
    # Gather log probabilities for target classes
    batch_size = size(logits)[1]
    target_indices = [CartesianIndex(i, Int(targets.data[i])) for i in 1:batch_size]
    target_log_probs = Tensor(log_probs.data[target_indices], requires_grad=logits.requires_grad)
    
    return -mean(target_log_probs)
end

# ============================================================================
# PHYSICS-INFORMED LOSSES (for PIML)
# ============================================================================

"""
Base class for physics-informed losses
"""
abstract type PhysicsInformedLoss <: Module end

"""
Concrete implementation of PhysicsInformedLoss
"""
mutable struct BasicPhysicsInformedLoss <: PhysicsInformedLoss
    data_loss_fn::Function
    physics_loss_fn::Function
    lambda_data::Float64
    lambda_physics::Float64
    
    function BasicPhysicsInformedLoss(data_loss_fn::Function, physics_loss_fn::Function;
                                     lambda_data::Float64=1.0, lambda_physics::Float64=1.0)
        new(data_loss_fn, physics_loss_fn, lambda_data, lambda_physics)
    end
end

function forward(loss::BasicPhysicsInformedLoss, predictions::Tensor, targets::Tensor, inputs::Tensor)::Tensor
    # Data fitting loss
    data_loss = loss.data_loss_fn(predictions, targets)
    
    # Physics constraint loss (e.g., PDE residuals)
    physics_loss = loss.physics_loss_fn(predictions, inputs)
    
    # Combined loss
    total_loss = loss.lambda_data * data_loss + loss.lambda_physics * physics_loss
    return total_loss
end

"""
Compute PDE residual loss: ||PDE(u, ∂u/∂x, ∂²u/∂x², ...)||²
where u = model(x)
"""
function pde_residual_loss(model::Module, inputs::Tensor, pde_fn::Function)::Tensor
    # Enable gradient tracking for inputs (for computing derivatives)
    inputs_grad = Tensor(inputs.data, requires_grad=true)
    
    # Forward pass through model
    predictions = model(inputs_grad)
    
    # Compute PDE residuals using the provided function
    residuals = pde_fn(predictions, inputs_grad)
    
    return mean(residuals ^ 2)
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
Absolute value function for Tensors
"""
function Base.abs(t::Tensor)::Tensor
    output_data = abs.(t.data)
    output = Tensor(output_data, requires_grad=t.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_sign = sign.(t.data)
        
        output.backward_fn = function()
            if t.requires_grad
                # ∂|x|/∂x = sign(x)
                grad_t = output.grad .* captured_sign
                t.grad .+= grad_t
            end
        end
        
        output.parents = [t]
    end
    
    return output
end

"""
Sign function for Tensors
"""
function Base.sign(t::Tensor)::Tensor
    Tensor(sign.(t.data), requires_grad=false)
end

"""
Clamp function for Tensors
"""
function Base.clamp(t::Tensor, min_val::Real, max_val::Real)::Tensor
    output_data = clamp.(t.data, min_val, max_val)
    output = Tensor(output_data, requires_grad=t.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_mask = (t.data .>= min_val) .& (t.data .<= max_val)
        
        output.backward_fn = function()
            if t.requires_grad
                # Gradient passes through only where within bounds
                grad_t = output.grad .* captured_mask
                t.grad .+= grad_t
            end
        end
        
        output.parents = [t]
    end
    
    return output
end

# ============================================================================
# EXPORTS
# ============================================================================

export mse_loss, mae_loss
export binary_cross_entropy, cross_entropy_loss
export PhysicsInformedLoss, BasicPhysicsInformedLoss, pde_residual_loss
export abs, sign, clamp

println("✓ Part 3: Loss Functions loaded successfully")