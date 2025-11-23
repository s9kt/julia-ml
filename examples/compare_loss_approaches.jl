"""
Comparison of Basic vs Proper Loss Function Approaches
This script demonstrates the fundamental differences between manual loss calculation
and proper loss function usage in the JuliaML.jl framework.
"""

include("../tensor.jl")
include("../losses.jl")
using Printf

println("="^70)
println("COMPARISON: BASIC LOSS CALCULATION vs PROPER LOSS FUNCTIONS")
println("="^70)

# Simple test case
test_w = Tensor([0.5], name="weight", requires_grad=true)
test_b = Tensor([0.3], name="bias", requires_grad=true)
test_X = Tensor([2.0], requires_grad=false)
test_y = Tensor([4.0], requires_grad=false)

println("\\nTest setup:")
@printf("  Input x: %.1f, Target y: %.1f\\n", test_X.data[1], test_y.data[1])
@printf("  Initial w: %.1f, b: %.1f\\n", test_w.data[1], test_b.data[1])
@printf("  Prediction: w*x + b = %.1f * %.1f + %.1f = %.1f\\n", 
        test_w.data[1], test_X.data[1], test_b.data[1], 
        test_w.data[1] * test_X.data[1] + test_b.data[1])

# ==========================================================================
# APPROACH 1: Basic Manual Loss Calculation
# ==========================================================================

println("\\n" * "-"^70)
println("APPROACH 1: BASIC MANUAL LOSS CALCULATION")
println("-"^70)

# Forward pass
pred_basic = test_X * test_w + test_b

# Manual loss (similar to linear_regression_basic.jl)
loss_basic_raw = Statistics.mean((pred_basic - test_y)^2)

println("\\nResults:")
@printf("  Loss type: %s\\n", typeof(loss_basic_raw))
@printf("  Loss value: %.4f\\n", loss_basic_raw)
@printf("  Has gradient tracking: %s\\n", false)

# Try to do backprop (this should fail or behave unexpectedly)
println("\\nTesting backpropagation:")
try
    # Clear gradients first
    test_w.grad .= 0.0
    test_b.grad .= 0.0
    
    # Try to call backward on Float64 (should fail)
    backward!(loss_basic_raw)
    println("  ❌ ERROR: Should not be able to backprop on Float64!")
catch e
    println("  ✓ Expected error: backward! cannot work on Float64 loss")
    println("    Error: $e")
end

# Check if gradients were computed
@printf("  w.grad after failed backprop: %s\\n", any(test_w.grad .!= 0) ? "Non-zero ❌" : "Zero ✓")
@printf("  b.grad after failed backprop: %s\\n", any(test_b.grad .!= 0) ? "Non-zero ❌" : "Zero ✓")

# ==========================================================================
# APPROACH 2: Proper Loss Function
# ==========================================================================

println("\\n" * "-"^70)
println("APPROACH 2: PROPER LOSS FUNCTION")
println("-"^70)

# Forward pass
pred_proper = test_X * test_w + test_b

# Proper loss function
loss_proper = mse_loss(pred_proper, test_y)

println("\\nResults:")
@printf("  Loss type: %s\\n", typeof(loss_proper))
@printf("  Loss value: %.4f\\n", loss_proper.data[1])
@printf("  Requires grad: %s\\n", loss_proper.requires_grad)
@printf("  Has backward function: %s\\n", loss_proper.backward_fn !== nothing)
@printf("  Number of parents: %d\\n", length(loss_proper.parents))

# Try to do backprop (this should work)
println("\\nTesting backpropagation:")
try
    # Clear gradients first
    test_w.grad .= 0.0
    test_b.grad .= 0.0
    
    # Backward should work
    backward!(loss_proper)
    println("  ✓ Backpropagation successful!")
    
    # Check if gradients were computed
    @printf("  w.grad after backprop: %.6f\\n", test_w.grad[1])
    @printf("  b.grad after backprop: %.6f\\n", test_b.grad[1])
    
    # Verify gradients are correct (manual calculation)
    # For MSE: L = (wx + b - y)^2
    # ∂L/∂w = 2*(wx + b - y)*x
    # ∂L/∂b = 2*(wx + b - y)
    error = test_w.data[1] * test_X.data[1] + test_b.data[1] - test_y.data[1]
    expected_w_grad = 2 * error * test_X.data[1]
    expected_b_grad = 2 * error
    
    @printf("  Expected w.grad: %.6f\\n", expected_w_grad)
    @printf("  Expected b.grad: %.6f\\n", expected_b_grad)
    
    w_correct = abs(test_w.grad[1] - expected_w_grad) < 1e-6
    b_correct = abs(test_b.grad[1] - expected_b_grad) < 1e-6
    
    println("  Gradient accuracy: w %s, b %s", 
           w_correct ? "✓" : "✗", b_correct ? "✓" : "✗")
    
catch e
    println("  ❌ Unexpected error: $e")
end

# ==========================================================================
# SIDE-BY-SIDE COMPARISON
# ==========================================================================

println("\\n" * "-"^70)
println("KEY DIFFERENCES SUMMARY")
println("-"^70)

println("\\n1. TENSOR PROPERTIES:")
println("   Basic:   Float64 (no computational graph)")
println("   Proper:  Tensor (full computational graph)")

println("\\n2. GRADIENT TRACKING:")
println("   Basic:   ❌ No automatic gradients")
println("   Proper:  ✅ Full automatic differentiation")

println("\\n3. BACKPROPAGATION:")
println("   Basic:   ❌ Cannot call backward! on Float64")
println("   Proper:  ✅ backpropagation works perfectly")

println("\\n4. MATHEMATICAL CORRECTNESS:")
println("   Basic:   ❌ Requires manual gradient calculation")
println("   Proper:  ✅ Automatic chain rule application")

println("\\n5. REUSABILITY:")
println("   Basic:   ❌ Hardcoded, not reusable")
println("   Proper:  ✅ Function can be reused anywhere")

println("\\n6. DEBUGGABILITY:")
println("   Basic:   ❌ Hard to debug gradient issues")
println("   Proper:  ✅ Can inspect computation graph")

# ==========================================================================
# VERIFICATION WITH ACTUAL TRAINING
# ==========================================================================

println("\\n" * "-"^70)
println("VERIFICATION: CAN WE ACTUALLY TRAIN?")
println("-"^70)

# Test if we can train a simple model with each approach
function test_training_approach(name, loss_type, training_fn)
    println("\\nTesting $name approach:")
    
    # Reset parameters
    w = Tensor([0.1], name="weight", requires_grad=true)
    b = Tensor([0.1], name="bias", requires_grad=true)
    lr = 0.01
    
    losses = Float64[]
    
    try
        for epoch in 1:10
            loss = training_fn(w, b)
            
            if loss isa Tensor
                zero_grad!(loss)
                backward!(loss)
                
                w.data .-= lr * w.grad
                b.data .-= lr * b.grad
                
                push!(losses, loss.data[1])
            else
                println("  ❌ Cannot backpropagate - no gradients available")
                return losses
            end
        end
        
        println("  ✓ Training successful! Final loss: %.6f", losses[end])
        return losses
    catch e
        println("  ❌ Training failed: $e")
        return losses
    end
end

# Define training functions for each approach
function basic_training(w, b)
    X = Tensor([2.0], requires_grad=false)
    y = Tensor([4.0], requires_grad=false)
    pred = X * w + b
    return Statistics.mean((pred - y)^2)  # Returns Float64
end

function proper_training(w, b)
    X = Tensor([2.0], requires_grad=false)
    y = Tensor([4.0], requires_grad=false)
    pred = X * w + b
    return mse_loss(pred, y)  # Returns Tensor
end

# Test both approaches
basic_losses = test_training_approach("Basic", "Float64", basic_training)
proper_losses = test_training_approach("Proper", "Tensor", proper_training)

# Show training progress
if length(proper_losses) > length(basic_losses)
    println("\\nTraining comparison:")
    println("  Basic approach:   losses = [$(join(["$(round(l, digits=6))" for l in basic_losses], ", "))]")
    println("  Proper approach:  losses = [$(join(["$(round(l, digits=6))" for l in proper_losses], ", "))]")
end

println("\\n" * "="^70)
println("CONCLUSION")
println("="^70)
println("\\nThe basic approach with manual loss calculation breaks the")
println("automatic differentiation system, while proper loss functions")
println("maintain the tensor computational graph and enable correct")
println("backpropagation. This is why loss functions are essential for")
println("deep learning frameworks.")
println("="^70)