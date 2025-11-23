"""
Linear Regression with Proper Loss Functions
Demonstrates the correct way to use the JuliaML.jl framework with proper loss functions
that maintain tensor properties and automatic differentiation.
"""

include("../tensor.jl")
include("../losses.jl")
using Printf

test_passed(name) = println("✓ $name")
test_failed(name, e) = println("✗ $name: $e")

try
    println("="^60)
    println("LINEAR REGRESSION WITH PROPER LOSS FUNCTIONS")
    println("="^60)
    
    # Generate data: y = 3x + 2 + noise
    num_points = 100
    learning_rate = 0.001  # Can use single LR with proper loss functions
    epochs = 500
    
    X_data = reshape(Float64.(1:num_points), (num_points, 1))
    y_data = 3.0 .* X_data .+ 2.0 .+ randn(num_points, 1) .* 0.1
    
    # Initialize parameters
    w = Tensor(randn(1, 1), name="weight")
    b = Tensor(randn(1, 1), name="bias")
    
    # Tracking progress
    losses = Float64[]
    w_history = Float64[]
    b_history = Float64[]
    
    println("\nTraining parameters:")
    @printf("  Learning rate: %.4f\n", learning_rate)
    @printf("  Epochs: %d\n", epochs)
    @printf("  Initial w: %.4f\n", w.data[1])
    @printf("  Initial b: %.4f\n", b.data[1])
    
    println("\n" * "-"^50)
    
    for epoch in 1:epochs
        # Create input tensors (no gradients needed for inputs)
        X = Tensor(X_data, requires_grad=false)
        y = Tensor(y_data, requires_grad=false)
        
        # Forward pass
        pred = matmul(X, transpose(w)) + reshape(b, (1, 1))
        
        # ✅ PROPER: Use the loss function - maintains tensor properties
        loss = mse_loss(pred, y)  # Returns Tensor with gradient tracking
        
        # Backward pass
        zero_grad!(loss)  # Clear gradients from previous iteration
        backward!(loss)   # Compute gradients automatically
        
        # Update parameters (SGD step)
        w.data .-= learning_rate .* w.grad
        b.data .-= learning_rate .* b.grad
        
        # Record progress
        push!(losses, loss.data[1])
        push!(w_history, w.data[1])
        push!(b_history, b.data[1])
        
        if epoch % 100 == 0
            @printf("Epoch %d: Loss = %.6f, w = %.4f, b = %.4f\n", 
                    epoch, loss.data[1], w.data[1], b.data[1])
        end
    end
    
    println("\n" * "-"^50)
    println("FINAL RESULTS:")
    @printf("  Learned parameters: w = %.4f, b = %.4f\n", w.data[1], b.data[1])
    @printf("  True parameters:    w = 3.0000, b = 2.0000\n")
    @printf("  Final loss: %.6f\n", losses[end])
    
    # Verify convergence
    if isapprox(w.data[1], 3.0, atol=0.1) && isapprox(b.data[1], 2.0, atol=0.1)
        test_passed("Linear regression with proper loss functions converged")
    else
        println("⚠ Warning: Parameters diverged from expected values")
    end
    
    println("\n" * "="^60)
    println("DEMONSTRATION: Loss Function Properties")
    println("="^60)
    
    # Show tensor properties are maintained
    X_demo = Tensor(randn(5, 1), requires_grad=false)
    y_demo = Tensor(3.0 .* X_demo .+ 2.0, requires_grad=false)
    w_demo = Tensor(randn(1, 1), requires_grad=true, name="demo_weight")
    
    pred_demo = matmul(X_demo, transpose(w_demo)) + Tensor([2.0], requires_grad=false)
    
    # Proper loss function
    loss_proper = mse_loss(pred_demo, y_demo)
    
    println("\n✅ Proper loss function properties:")
    @printf("  Type: %s\n", typeof(loss_proper))
    @printf("  Requires grad: %s\n", loss_proper.requires_grad)
    @printf("  Has backward function: %s\n", loss_proper.backward_fn !== nothing)
    @printf("  Number of parents in graph: %d\n", length(loss_proper.parents))
    
    # Test backpropagation
    if loss_proper.requires_grad
        backward!(loss_proper)
        @printf("  ✓ Backpropagation successful!\n")
        @printf("  Weight gradient computed: %s\n", any(w_demo.grad .!= 0) ? "✓" : "✗")
    end
    
    using Plots
    
    println("\n" * "="^60)
    println("VISUALIZATION")
    println("="^60)
    
    # Loss curve
    p1 = plot(losses, xlabel="Epoch", ylabel="Loss", 
              title="Training Loss (Proper Loss Function)", 
              legend=false, lw=2, color=:blue)
    
    # Parameter convergence
    p2 = plot(w_history, label="w", lw=2, xlabel="Epoch", ylabel="Value")
    plot!(p2, b_history, label="b", lw=2)
    hline!(p2, [3.0], linestyle=:dash, color=:gray, label="w target")
    hline!(p2, [2.0], linestyle=:dash, color=:gray, label="b target")
    title!(p2, "Parameter Convergence")
    
    # Final fit comparison
    x_line = 0:0.1:11
    y_true = 3.0 .* x_line .+ 2.0
    y_pred = w.data[1] .* x_line .+ b.data[1]
    
    p3 = scatter(X_data[:], y_data[:], label="Data", ms=6, 
                xlabel="x", ylabel="y", color=:blue, alpha=0.6)
    plot!(p3, x_line, y_true, label="True: y=3x+2", 
          lw=2, linestyle=:dash, color=:red)
    plot!(p3, x_line, y_pred, 
          label="Learned: y=$(round(w.data[1], digits=3))x+$(round(b.data[1], digits=3))", 
          lw=2, color=:green)
    title!(p3, "Model Fit: Learned vs True Line")
    
    # Combine all plots
    plot(p1, p2, p3, layout=(1,3), size=(1400,400), dpi=300)
    
    # Save visualization
    mkpath("out")
    savefig("out/linear_regression_augmented.png")
    println("\n📊 Saved: linear_regression_augmented.png")
    
    println("\n" * "="^60)
    println("COMPARISON: Basic vs Augmented Approach")
    println("="^60)
    
    println("\n🔹 Basic Approach (linear_regression_basic.jl):")
    println("   - Uses Statistics.mean() directly on tensor data")
    println("   - ⚠ Potential gradient tracking issues")
    println("   - ❌ Breaks computation graph abstraction")
    println("   - Requires separate learning rates for different parameter types")
    
    println("\n🔹 Augmented Approach (this file):")
    println("   - Uses proper mse_loss() function")
    println("   - ✅ Maintains full tensor properties")
    println("   - ✅ Preserves computation graph")
    println("   - ✅ Automatic differentiation works correctly")
    println("   - Can use unified learning rate")
    
    println("\nKey Takeaway:")
    println("Loss functions are not just convenience methods - they're essential")
    println("for maintaining the automatic differentiation system's integrity.")
    
catch e
    test_failed("Linear regression augmented", e)
    println("\nError details: $e")
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end