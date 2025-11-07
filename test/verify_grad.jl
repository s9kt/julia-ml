"""
Verify Linear Regression Gradients Are Correct
Distinguishes between training issues vs autodiff bugs
"""

include("../tensor.jl")
using Statistics, Printf

# ============================================================================
# NUMERICAL GRADIENT (Ground Truth)
# ============================================================================

function numerical_gradient_linreg(X_data, y_data, w_val, b_val, epsilon=1e-5)
    """Compute gradients numerically using finite differences"""
    
    # Helper: compute loss given w, b
    function compute_loss(w, b)
        pred = X_data * w .+ b
        return mean((pred .- y_data).^2)
    end
    
    # Gradient w.r.t. w
    loss_plus = compute_loss(w_val + epsilon, b_val)
    loss_minus = compute_loss(w_val - epsilon, b_val)
    grad_w_numerical = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Gradient w.r.t. b
    loss_plus = compute_loss(w_val, b_val + epsilon)
    loss_minus = compute_loss(w_val, b_val - epsilon)
    grad_b_numerical = (loss_plus - loss_minus) / (2 * epsilon)
    
    return grad_w_numerical, grad_b_numerical
end

# ============================================================================
# ANALYTICAL GRADIENT (Our Implementation)
# ============================================================================

function analytical_gradient_linreg(X_data, y_data, w_val, b_val)
    """Compute gradients using our autodiff"""
    
    X = Tensor(X_data, requires_grad=false)
    y = Tensor(y_data, requires_grad=false)
    w = Tensor([w_val;;], name="w")  # Make it 2D for matmul
    b = Tensor([b_val;;], name="b")
    
    # Forward pass
    pred = matmul(X, transpose(w)) + reshape(b, (1, 1))
    loss = Statistics.mean((pred - y)^2)
    
    # Backward pass
    zero_grad!(loss)
    backward!(loss)
    
    return w.grad[1], b.grad[1]
end

# ============================================================================
# TEST
# ============================================================================

println("="^60)
println("GRADIENT VERIFICATION TEST")
println("="^60)

# Simple test case
X_data = reshape([1.0, 2.0, 3.0, 4.0, 5.0], (5, 1))
y_data = 3.0 .* X_data .+ 2.0  # True: y = 3x + 2

# Test at specific point
w_test = 2.5
b_test = 1.5

println("\nTest Point:")
@printf("  w = %.2f (true: 3.0)\n", w_test)
@printf("  b = %.2f (true: 2.0)\n", b_test)

# Compute numerical gradients (ground truth)
grad_w_num, grad_b_num = numerical_gradient_linreg(X_data, y_data, w_test, b_test)

println("\nNumerical Gradients (Ground Truth):")
@printf("  ∂L/∂w = %.6f\n", grad_w_num)
@printf("  ∂L/∂b = %.6f\n", grad_b_num)

# Compute analytical gradients (our implementation)
grad_w_auto, grad_b_auto = analytical_gradient_linreg(X_data, y_data, w_test, b_test)

println("\nAnalytical Gradients (Our Autodiff):")
@printf("  ∂L/∂w = %.6f\n", grad_w_auto)
@printf("  ∂L/∂b = %.6f\n", grad_b_auto)

# Compare
println("\nComparison:")
error_w = abs(grad_w_num - grad_w_auto)
error_b = abs(grad_b_num - grad_b_auto)
relative_error_w = error_w / (abs(grad_w_num) + 1e-8)
relative_error_b = error_b / (abs(grad_b_num) + 1e-8)

@printf("  Weight gradient error: %.2e (relative: %.2e)\n", error_w, relative_error_w)
@printf("  Bias gradient error:   %.2e (relative: %.2e)\n", error_b, relative_error_b)

println("\n" * "="^60)
if relative_error_w < 1e-4 && relative_error_b < 1e-4
    println("✓ GRADIENTS CORRECT - Autodiff implementation is working!")
    println("  Issue is purely training hyperparameters (LR, epochs)")
else
    println("✗ GRADIENTS INCORRECT - Bug in autodiff implementation!")
    println("  Fix the backward pass before adjusting hyperparameters")
end
println("="^60)

# ============================================================================
# MANUAL DERIVATION CHECK
# ============================================================================

println("\n" * "="^60)
println("MANUAL MATH CHECK")
println("="^60)

# For MSE loss: L = (1/n) Σ(wx + b - y)²
# ∂L/∂w = (2/n) Σ(wx + b - y) * x
# ∂L/∂b = (2/n) Σ(wx + b - y)

pred_manual = w_test .* X_data .+ b_test
residuals = pred_manual .- y_data
grad_w_manual = (2.0 / length(X_data)) * sum(residuals .* X_data)
grad_b_manual = (2.0 / length(X_data)) * sum(residuals)

println("\nManual Calculation:")
@printf("  ∂L/∂w = %.6f\n", grad_w_manual)
@printf("  ∂L/∂b = %.6f\n", grad_b_manual)

println("\nDoes manual match numerical?")
@printf("  Weight: %.2e error\n", abs(grad_w_manual - grad_w_num))
@printf("  Bias:   %.2e error\n", abs(grad_b_manual - grad_b_num))

println("\nDoes manual match autodiff?")
@printf("  Weight: %.2e error\n", abs(grad_w_manual - grad_w_auto))
@printf("  Bias:   %.2e error\n", abs(grad_b_manual - grad_b_auto))

println("="^60)