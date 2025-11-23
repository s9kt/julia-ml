include("../tensor.jl")
using Printf

test_passed(name) = println("✓ $name")
test_failed(name, e) = println("✗ $name: $e")

try
    # Data is supposed to fit along the line y = 3x + 2

    num_points = 100 # These are the amount of data points that we want to generate to try to fit our model to
    weight_learning_rate = 0.0001 # When we find the gradient of our weights, we may not want to have it change the weights too much, we reduce its magnitude, this is called the learning bias_learning_rate
    bias_learning_rate = 0.03 # Similar situation as above, but for the bias
    epochs = 500 # An epoch is a forward pass and a back pass. The amount of epochs determines how many times the function will attempt to fit the data

    X_data = reshape(Float64.(1:num_points), (num_points, 1))
    y_data = 3.0 .* X_data .+ 2.0 .+ randn(num_points, 1) .* 0.1
    
    w = Tensor(randn(1, 1), name="weight")
    b = Tensor(randn(1, 1), name="bias")
    losses = Float64[]
    w_history = Float64[]
    b_history = Float64[]
    
    for epoch in 1:epochs
        X = Tensor(X_data, requires_grad=false)
        y = Tensor(y_data, requires_grad=false)
        
        pred = matmul(X, transpose(w)) + reshape(b, (1, 1))
        loss = Statistics.mean((pred - y)^2)
        
        zero_grad!(loss)
        backward!(loss)
        
        w.data .-= weight_learning_rate .* w.grad
        b.data .-= bias_learning_rate .* b.grad 
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
    savefig("out/linear_regression_basic.png")
    println("\n📊 Saved: linear_regression_basic.png")
catch e; test_failed("Linear regression", e); end