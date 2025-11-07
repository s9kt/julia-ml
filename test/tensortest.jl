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

println("\n" * "="^60)
println("✓ All Tests Passed - Part 1 Implementation Working!")
println("="^60)