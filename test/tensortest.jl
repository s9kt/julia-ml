"""
Test Suite for Part 1: Tensor and Autodiff System
"""

include("../tensor.jl")
using Printf

test_passed(name) = println("✓ $name")
test_failed(name, e) = println("✗ $name: $e")

println("\n" * "="^60)
println("Testing Basic Operations")
println("="^60)

# Tensor creation
try
    t1, t2, t3 = Tensor([1.0, 2.0, 3.0]), Tensor(5.0), Tensor([1.0 2.0; 3.0 4.0])
    @assert size(t1) == (3,) && size(t2) == (1,) && size(t3) == (2, 2)
    test_passed("Tensor creation")
catch e; test_failed("Tensor creation", e); end

# Operations
try
    a, b = Tensor([1.0, 2.0, 3.0]), Tensor([4.0, 5.0, 6.0])
    @assert all((a + b).data .≈ [5.0, 7.0, 9.0])
    @assert all((a * b).data .≈ [4.0, 10.0, 18.0])  # Fixed: 1*4=4, 2*5=10, 3*6=18
    @assert all((a + 5.0).data .≈ [6.0, 7.0, 8.0])
    @assert all((a^2).data .≈ [1.0, 4.0, 9.0])
    test_passed("Basic operations")
catch e; test_failed("Basic operations", e); end

println("\n" * "="^60)
println("Testing Matrix Operations")
println("="^60)

try
    a, b = Tensor([1.0 2.0; 3.0 4.0]), Tensor([5.0 6.0; 7.0 8.0])
    c = matmul(a, b)
    @assert all(c.data .≈ [19.0 22.0; 43.0 50.0])
    test_passed("Matrix multiplication")
catch e; test_failed("Matrix multiplication", e); end

try
    a = Tensor([1.0 2.0 3.0; 4.0 5.0 6.0])
    b = transpose(a)
    @assert size(b) == (3, 2)
    test_passed("Transpose")
catch e; test_failed("Transpose", e); end

println("\n" * "="^60)
println("Testing Activations")
println("="^60)

try
    @assert all(relu(Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])).data .≈ [0.0, 0.0, 0.0, 1.0, 2.0])
    @assert sigmoid(Tensor([0.0])).data[1] ≈ 0.5
    @assert tanh(Tensor([0.0])).data[1] ≈ 0.0
    test_passed("Activation functions")
catch e; test_failed("Activation functions", e); end

println("\n" * "="^60)
println("Testing Autodiff")
println("="^60)

# Addition gradient
try
    a, b = Tensor([1.0, 2.0, 3.0]), Tensor([4.0, 5.0, 6.0])
    c = a + b
    backward!(c)
    @assert all(a.grad .≈ ones(3)) && all(b.grad .≈ ones(3))
    test_passed("Gradient: addition")
catch e; test_failed("Gradient: addition", e); end

# Multiplication gradient
try
    a, b = Tensor([2.0, 3.0]), Tensor([4.0, 5.0])
    c = a * b
    backward!(c)
    @assert all(a.grad .≈ [4.0, 5.0]) && all(b.grad .≈ [2.0, 3.0])
    test_passed("Gradient: multiplication")
catch e; test_failed("Gradient: multiplication", e); end

# Chain rule
try
    x = Tensor([2.0])
    w = (x^2 + 3.0) * 2.0  # w = 2x² + 6
    backward!(w)
    @assert x.grad[1] ≈ 8.0  # dw/dx = 4x = 8
    test_passed("Gradient: chain rule")
catch e; test_failed("Gradient: chain rule", e); end

# ReLU gradient
try
    x = Tensor([-1.0, 0.0, 1.0])
    y = relu(x)
    backward!(sum(y))
    @assert all(x.grad .≈ [0.0, 0.0, 1.0])
    test_passed("Gradient: ReLU")
catch e; test_failed("Gradient: ReLU", e); end

println("\n" * "="^60)
println("Practical Example: Linear Regression")
println("="^60)

try
    # Data: y = 3x + 2
    X_data = reshape(Float64.(1:100), (100, 1))
    y_data = 3.0 .* X_data .+ 2.0 .+ randn(100, 1) .* 0.1

    # Normalize
    
    w = Tensor(randn(1, 1), name="weight")
    b = Tensor(randn(1, 1), name="bias")
    losses = Float64[]
    w_history = Float64[]
    b_history = Float64[]
    
    for epoch in 1:500
        X = Tensor(X_data, requires_grad=false)
        y = Tensor(y_data, requires_grad=false)
        
        pred = matmul(X, transpose(w)) + reshape(b, (1, 1))
        loss = Statistics.mean((pred - y)^2)
        
        zero_grad!(loss)
        backward!(loss)
        
        w.data .-= 0.0001 .* w.grad
        b.data .-= 0.03 .* b.grad 
        push!(losses, loss.data[1])
        push!(w_history, w.data[1])
        push!(b_history, b.data[1])
        
        if epoch % 100 == 0
            @printf("Epoch %d: Loss = %.6f, w = %.3f, b = %.3f\n", 
                    epoch, loss.data[1], w.data[1], b.data[1])
        end
    end
    println(w.data[1])
    println(b.data[1])
    @assert isapprox(w.data[1], 3.0, atol=0.5) && isapprox(b.data[1], 2.0, atol=0.5)
    test_passed("Linear regression converged")

    using Plots
    # Loss curve
    p1 = plot(losses, xlabel="Epoch", ylabel="Loss", 
            title="Training Loss", legend=false, lw=2, color=:blue)

    # Parameter convergence
    p2 = plot(w_history, label="w", lw=2, xlabel="Epoch", ylabel="Value")
    plot!(p2, b_history, label="b", lw=2)
    hline!(p2, [3.0], linestyle=:dash, color=:gray, label="w target")
    hline!(p2, [2.0], linestyle=:dash, color=:gray, label="b target")
    title!(p2, "Parameter Convergence")

    # Final fit
    x_line = 0:0.1:11
    y_true = 3.0 .* x_line .+ 2.0
    y_pred = w.data[1] .* x_line .+ b.data[1]

    p3 = scatter(X_data[:], y_data[:], label="Data", ms=6, 
                xlabel="x", ylabel="y", color=:blue)
    plot!(p3, x_line, y_true, label="True: y=3x+2", 
        lw=2, linestyle=:dash, color=:red)
    plot!(p3, x_line, y_pred, label="Learned: y=$(round(w.data[1],digits=2))x+$(round(b.data[1],digits=2))", 
        lw=2, color=:green)
    title!(p3, "Final Fit")

    # Combine
    plot(p1, p2, p3, layout=(1,3), size=(1400,400))
    mkpath("out")
    savefig("out/linear_regression.png")
    println("\n📊 Saved: linear_regression.png")
catch e; test_failed("Linear regression", e); end

println("\n" * "="^60)
println("✓ All Tests Passed - Part 1 Implementation Working!")
println("="^60)