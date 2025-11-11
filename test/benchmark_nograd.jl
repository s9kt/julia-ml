"""
Benchmark for @no_grad Macro Performance
Tests the overhead of gradient tracking and benefits of disabling it
"""

using BenchmarkTools
using Statistics
using Printf

# Load the autodiff system
include("../tensor.jl")

println("\n" * "="^70)
println("@no_grad Performance Benchmark")
println("="^70)

# ============================================================================
# BENCHMARK 1: Simple Element-wise Operations
# ============================================================================

function benchmark_elementwise()
    println("\n📊 Benchmark 1: Element-wise Operations (1000x1000 matrices)")
    println("-"^70)
    
    # Create test tensors
    a = Tensor(randn(1000, 1000), name="a")
    b = Tensor(randn(1000, 1000), name="b")
    
    # With gradients
    println("\n  With gradient tracking:")
    time_with_grad = @belapsed begin
        c = $a + $b
        d = c * $a
        e = relu(d)
        f = sum(e)
    end
    
    # Without gradients
    println("  Without gradient tracking (@no_grad):")
    time_no_grad = @belapsed begin
        @no_grad begin
            c = $a + $b
            d = c * $a
            e = relu(d)
            f = sum(e)
        end
    end
    
    speedup = time_with_grad / time_no_grad
    @printf("  ⚡ Speedup: %.2fx faster\n", speedup)
    @printf("  ⏱️  With grad: %.3f ms\n", time_with_grad * 1000)
    @printf("  ⏱️  No grad:   %.3f ms\n", time_no_grad * 1000)
    
    return (with_grad=time_with_grad, no_grad=time_no_grad, speedup=speedup)
end

# ============================================================================
# BENCHMARK 2: Matrix Multiplication Chain
# ============================================================================

function benchmark_matmul()
    println("\n📊 Benchmark 2: Matrix Multiplication Chain")
    println("-"^70)
    
    # Create weight matrices for a simple neural network
    W1 = Tensor(randn(512, 784), name="W1")
    W2 = Tensor(randn(256, 512), name="W2")
    W3 = Tensor(randn(10, 256), name="W3")
    x = Tensor(randn(784, 64), name="x")  # 64 samples
    
    # With gradients
    println("\n  With gradient tracking:")
    time_with_grad = @belapsed begin
        h1 = matmul($W1, $x)
        h1 = relu(h1)
        h2 = matmul($W2, h1)
        h2 = relu(h2)
        out = matmul($W3, h2)
    end
    
    # Without gradients
    println("  Without gradient tracking (@no_grad):")
    time_no_grad = @belapsed begin
        @no_grad begin
            h1 = matmul($W1, $x)
            h1 = relu(h1)
            h2 = matmul($W2, h1)
            h2 = relu(h2)
            out = matmul($W3, h2)
        end
    end
    
    speedup = time_with_grad / time_no_grad
    @printf("  ⚡ Speedup: %.2fx faster\n", speedup)
    @printf("  ⏱️  With grad: %.3f ms\n", time_with_grad * 1000)
    @printf("  ⏱️  No grad:   %.3f ms\n", time_no_grad * 1000)
    
    return (with_grad=time_with_grad, no_grad=time_no_grad, speedup=speedup)
end

# ============================================================================
# BENCHMARK 3: Deep Computation Graph
# ============================================================================

function benchmark_deep_graph()
    println("\n📊 Benchmark 3: Deep Computation Graph (50 layers)")
    println("-"^70)
    
    # With gradients
    println("\n  With gradient tracking:")
    time_with_grad = @belapsed begin
        x = Tensor(randn(100, 100), name="x")
        for i in 1:50
            x = relu(x + Tensor(0.01 * randn(100, 100)))
        end
        result = sum(x)
    end
    
    # Without gradients
    println("  Without gradient tracking (@no_grad):")
    time_no_grad = @belapsed begin
        @no_grad begin
            x = Tensor(randn(100, 100), name="x")
            for i in 1:50
                x = relu(x + Tensor(0.01 * randn(100, 100)))
            end
            result = sum(x)
        end
    end
    
    speedup = time_with_grad / time_no_grad
    @printf("  ⚡ Speedup: %.2fx faster\n", speedup)
    @printf("  ⏱️  With grad: %.3f ms\n", time_with_grad * 1000)
    @printf("  ⏱️  No grad:   %.3f ms\n", time_no_grad * 1000)
    
    return (with_grad=time_with_grad, no_grad=time_no_grad, speedup=speedup)
end

# ============================================================================
# BENCHMARK 4: Broadcasting Operations
# ============================================================================

function benchmark_broadcasting()
    println("\n📊 Benchmark 4: Broadcasting Operations")
    println("-"^70)
    
    a = Tensor(randn(1000, 1), name="a")
    b = Tensor(randn(1, 1000), name="b")
    
    # With gradients
    println("\n  With gradient tracking:")
    time_with_grad = @belapsed begin
        c = $a + $b  # Broadcasts to (1000, 1000)
        d = c * c
        e = sigmoid(d)
        f = sum(e)
    end
    
    # Without gradients
    println("  Without gradient tracking (@no_grad):")
    time_no_grad = @belapsed begin
        @no_grad begin
            c = $a + $b
            d = c * c
            e = sigmoid(d)
            f = sum(e)
        end
    end
    
    speedup = time_with_grad / time_no_grad
    @printf("  ⚡ Speedup: %.2fx faster\n", speedup)
    @printf("  ⏱️  With grad: %.3f ms\n", time_with_grad * 1000)
    @printf("  ⏱️  No grad:   %.3f ms\n", time_no_grad * 1000)
    
    return (with_grad=time_with_grad, no_grad=time_no_grad, speedup=speedup)
end

# ============================================================================
# BENCHMARK 5: Activation Function Heavy
# ============================================================================

function benchmark_activations()
    println("\n📊 Benchmark 5: Multiple Activation Functions")
    println("-"^70)
    
    x = Tensor(randn(500, 500), name="x")
    
    # With gradients
    println("\n  With gradient tracking:")
    time_with_grad = @belapsed begin
        a = relu($x)
        b = sigmoid($x)
        c = tanh($x)
        d = a + b + c
        e = softmax(d, dims=1)
        f = sum(e)
    end
    
    # Without gradients
    println("  Without gradient tracking (@no_grad):")
    time_no_grad = @belapsed begin
        @no_grad begin
            a = relu($x)
            b = sigmoid($x)
            c = tanh($x)
            d = a + b + c
            e = softmax(d, dims=1)
            f = sum(e)
        end
    end
    
    speedup = time_with_grad / time_no_grad
    @printf("  ⚡ Speedup: %.2fx faster\n", speedup)
    @printf("  ⏱️  With grad: %.3f ms\n", time_with_grad * 1000)
    @printf("  ⏱️  No grad:   %.3f ms\n", time_no_grad * 1000)
    
    return (with_grad=time_with_grad, no_grad=time_no_grad, speedup=speedup)
end

# ============================================================================
# BENCHMARK 6: Memory Allocation Test
# ============================================================================

function benchmark_memory()
    println("\n📊 Benchmark 6: Memory Allocation Analysis")
    println("-"^70)
    
    x = Tensor(randn(1000, 1000), name="x")
    y = Tensor(randn(1000, 1000), name="y")
    
    # With gradients
    println("\n  With gradient tracking:")
    alloc_with_grad = @allocated begin
        for i in 1:10
            z = x + y
            z = z * x
            z = relu(z)
        end
    end
    
    # Without gradients
    println("  Without gradient tracking (@no_grad):")
    alloc_no_grad = @allocated begin
        @no_grad begin
            for i in 1:10
                z = x + y
                z = z * x
                z = relu(z)
            end
        end
    end
    
    reduction = (alloc_with_grad - alloc_no_grad) / alloc_with_grad * 100
    @printf("  📉 Memory reduction: %.1f%%\n", reduction)
    @printf("  💾 With grad: %.2f MB\n", alloc_with_grad / 1024^2)
    @printf("  💾 No grad:   %.2f MB\n", alloc_no_grad / 1024^2)
    
    return (with_grad=alloc_with_grad, no_grad=alloc_no_grad, reduction=reduction)
end

# ============================================================================
# BENCHMARK 7: Inference Simulation
# ============================================================================

function benchmark_inference()
    println("\n📊 Benchmark 7: Simulated Inference (100 batches)")
    println("-"^70)
    
    # Simple 3-layer network
    W1 = Tensor(randn(128, 784), name="W1")
    W2 = Tensor(randn(64, 128), name="W2")
    W3 = Tensor(randn(10, 64), name="W3")
    
    # With gradients (wrong for inference!)
    println("\n  With gradient tracking (INCORRECT for inference):")
    time_with_grad = @belapsed begin
        for batch in 1:100
            x = Tensor(randn(784, 32))
            h1 = relu(matmul($W1, x))
            h2 = relu(matmul($W2, h1))
            out = matmul($W3, h2)
        end
    end
    
    # Without gradients (correct for inference)
    println("  Without gradient tracking (@no_grad - CORRECT):")
    time_no_grad = @belapsed begin
        @no_grad begin
            for batch in 1:100
                x = Tensor(randn(784, 32))
                h1 = relu(matmul($W1, x))
                h2 = relu(matmul($W2, h1))
                out = matmul($W3, h2)
            end
        end
    end
    
    speedup = time_with_grad / time_no_grad
    @printf("  ⚡ Speedup: %.2fx faster\n", speedup)
    @printf("  ⏱️  With grad: %.3f ms\n", time_with_grad * 1000)
    @printf("  ⏱️  No grad:   %.3f ms\n", time_no_grad * 1000)
    
    return (with_grad=time_with_grad, no_grad=time_no_grad, speedup=speedup)
end

# ============================================================================
# RUN ALL BENCHMARKS
# ============================================================================

function run_all_benchmarks()
    println("\n🚀 Running all benchmarks...")
    println("This may take a few minutes...\n")
    
    results = Dict()
    
    results[:elementwise] = benchmark_elementwise()
    results[:matmul] = benchmark_matmul()
    results[:deep_graph] = benchmark_deep_graph()
    results[:broadcasting] = benchmark_broadcasting()
    results[:activations] = benchmark_activations()
    results[:memory] = benchmark_memory()
    results[:inference] = benchmark_inference()
    
    # Summary
    println("\n" * "="^70)
    println("📈 SUMMARY")
    println("="^70)
    
    speedups = [
        results[:elementwise].speedup,
        results[:matmul].speedup,
        results[:deep_graph].speedup,
        results[:broadcasting].speedup,
        results[:activations].speedup,
        results[:inference].speedup
    ]
    
    avg_speedup = mean(speedups)
    max_speedup = maximum(speedups)
    
    @printf("\n  Average speedup: %.2fx\n", avg_speedup)
    @printf("  Maximum speedup: %.2fx\n", max_speedup)
    @printf("  Memory reduction: %.1f%%\n", results[:memory].reduction)
    
    println("\n" * "="^70)
    
    return results
end

# Run benchmarks
results = run_all_benchmarks()

println("\n✅ Benchmark complete!")