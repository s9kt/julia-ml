"""
Core Tensor and Automatic Differentiation Implementation
Part 1: Tensor Structure, Operations, and Backward Pass Engine
"""

using LinearAlgebra
using Statistics

# ============================================================================
# TENSOR STRUCTURE
# ============================================================================

mutable struct Tensor
    data::Array{Float64}
    grad::Array{Float64}
    requires_grad::Bool
    parents::Vector{Tensor}
    backward_fn::Union{Function, Nothing}
    name::String
    is_leaf::Bool
    
    # Inner constructor - accepts any AbstractArray and converts to concrete Array
    function Tensor(data::AbstractArray{<:Real}; requires_grad::Bool=true, name::String="")
        # Convert to concrete Array{Float64}
        concrete_data = Array{Float64}(data)
        grad = zeros(Float64, size(concrete_data))
        new(concrete_data, grad, requires_grad, Tensor[], nothing, name, true)
    end
    
    # Constructor for scalars
    function Tensor(data::Real; requires_grad::Bool=true, name::String="")
        arr = fill(Float64(data), (1,))
        grad = zeros(Float64, 1)
        new(arr, grad, requires_grad, Tensor[], nothing, name, true)
    end
end

# Utility functions
Base.size(t::Tensor) = size(t.data)
Base.ndims(t::Tensor) = ndims(t.data)
Base.length(t::Tensor) = length(t.data)
is_scalar(t::Tensor) = length(t.data) == 1

# ============================================================================
# SHAPE UTILITIES (CRITICAL FOR BROADCASTING)
# ============================================================================

"""
Pad shape with 1s on the left to reach target_ndim dimensions
"""
function pad_left(shape::Tuple, target_ndim::Int)
    if length(shape) >= target_ndim
        return shape
    end
    return tuple(ones(Int, target_ndim - length(shape))..., shape...)
end

"""
Implements NumPy/Julia broadcasting rules
Returns the result shape when broadcasting two shapes together
"""
function broadcast_shapes(shape_a::Tuple, shape_b::Tuple)
    max_ndim = max(length(shape_a), length(shape_b))
    
    # Pad shapes with 1s on the left
    padded_a = pad_left(shape_a, max_ndim)
    padded_b = pad_left(shape_b, max_ndim)
    
    result_shape = Int[]
    
    for i in 1:max_ndim
        if padded_a[i] == padded_b[i]
            push!(result_shape, padded_a[i])
        elseif padded_a[i] == 1
            push!(result_shape, padded_b[i])
        elseif padded_b[i] == 1
            push!(result_shape, padded_a[i])
        else
            error("Shapes $shape_a and $shape_b are not compatible for broadcasting")
        end
    end
    
    return tuple(result_shape...)
end

"""
Reduces gradient to match original tensor shape
Critical for correct backprop through broadcasting
"""
function sum_to_shape(grad::Array{Float64}, target_shape::Tuple)
    result = grad
    
    # Handle extra leading dimensions
    while length(size(result)) > length(target_shape)
        result = Base.sum(result, dims=1)
        result = dropdims(result, dims=1)
    end
    
    # Handle broadcast dimensions
    for i in 1:length(target_shape)
        if target_shape[i] == 1 && size(result, i) > 1
            result = Base.sum(result, dims=i)
        end
    end
    
    return result
end

# ============================================================================
# ELEMENT-WISE OPERATIONS
# ============================================================================

"""
Element-wise addition with broadcasting support
"""
function Base.:+(a::Tensor, b::Tensor)
    # Forward pass
    output_data = a.data .+ b.data
    output = Tensor(output_data, requires_grad=(a.requires_grad || b.requires_grad))
    output.is_leaf = false
    
    # Build backward function only if gradients enabled
    if output.requires_grad && GRAD_ENABLED[]
        captured_a_shape = size(a)
        captured_b_shape = size(b)
        
        output.backward_fn = function()
            if a.requires_grad
                grad_a = sum_to_shape(output.grad, captured_a_shape)
                a.grad .+= grad_a
            end
            
            if b.requires_grad
                grad_b = sum_to_shape(output.grad, captured_b_shape)
                b.grad .+= grad_b
            end
        end
        
        output.parents = [a, b]
    end
    
    return output
end

# Scalar addition
Base.:+(a::Tensor, b::Real) = a + Tensor(b, requires_grad=false)
Base.:+(a::Real, b::Tensor) = Tensor(a, requires_grad=false) + b

"""
Element-wise multiplication with broadcasting support
"""
function Base.:*(a::Tensor, b::Tensor)
    # Forward pass
    output_data = a.data .* b.data
    output = Tensor(output_data, requires_grad=(a.requires_grad || b.requires_grad))
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        # CRITICAL: Capture forward pass values
        captured_a_data = copy(a.data)
        captured_b_data = copy(b.data)
        captured_a_shape = size(a)
        captured_b_shape = size(b)
        
        output.backward_fn = function()
            if a.requires_grad
                grad_a = output.grad .* captured_b_data
                grad_a = sum_to_shape(grad_a, captured_a_shape)
                a.grad .+= grad_a
            end
            
            if b.requires_grad
                grad_b = output.grad .* captured_a_data
                grad_b = sum_to_shape(grad_b, captured_b_shape)
                b.grad .+= grad_b
            end
        end
        
        output.parents = [a, b]
    end
    
    return output
end

# Scalar multiplication
Base.:*(a::Tensor, b::Real) = a * Tensor(b, requires_grad=false)
Base.:*(a::Real, b::Tensor) = Tensor(a, requires_grad=false) * b

"""
Unary negation
"""
function Base.:-(a::Tensor)
    output_data = -a.data
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        output.backward_fn = function()
            if a.requires_grad
                a.grad .+= -output.grad
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

"""
Element-wise subtraction (implemented as a + (-b))
"""
Base.:-(a::Tensor, b::Tensor) = a + (-b)
Base.:-(a::Tensor, b::Real) = a + (-b)
Base.:-(a::Real, b::Tensor) = a + (-b)

"""
Element-wise division
"""
function Base.:/(a::Tensor, b::Tensor)
    output_data = a.data ./ b.data
    output = Tensor(output_data, requires_grad=(a.requires_grad || b.requires_grad))
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_a_data = copy(a.data)
        captured_b_data = copy(b.data)
        captured_a_shape = size(a)
        captured_b_shape = size(b)
        
        output.backward_fn = function()
            if a.requires_grad
                # ∂(a/b)/∂a = 1/b
                grad_a = output.grad ./ captured_b_data
                grad_a = sum_to_shape(grad_a, captured_a_shape)
                a.grad .+= grad_a
            end
            
            if b.requires_grad
                # ∂(a/b)/∂b = -a/b²
                grad_b = -output.grad .* captured_a_data ./ (captured_b_data .^ 2)
                grad_b = sum_to_shape(grad_b, captured_b_shape)
                b.grad .+= grad_b
            end
        end
        
        output.parents = [a, b]
    end
    
    return output
end

Base.:/(a::Tensor, b::Real) = a / Tensor(b, requires_grad=false)
Base.:/(a::Real, b::Tensor) = Tensor(a, requires_grad=false) / b

"""
Element-wise power
"""
function Base.:^(a::Tensor, exponent::Real)
    output_data = a.data .^ exponent
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_a_data = copy(a.data)
        captured_exponent = Float64(exponent)
        
        output.backward_fn = function()
            if a.requires_grad
                # ∂(a^n)/∂a = n * a^(n-1)
                grad_a = output.grad .* captured_exponent .* (captured_a_data .^ (captured_exponent - 1))
                a.grad .+= grad_a
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

# ============================================================================
# MATRIX OPERATIONS
# ============================================================================

"""
Matrix multiplication: C = A @ B
"""
function matmul(a::Tensor, b::Tensor)
    # Check dimensions
    if size(a)[end] != size(b)[end-1]
        error("Inner dimensions must match for matmul: got $(size(a)) and $(size(b))")
    end
    
    output_data = a.data * b.data
    output = Tensor(output_data, requires_grad=(a.requires_grad || b.requires_grad))
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_a_data = copy(a.data)
        captured_b_data = copy(b.data)
        
        output.backward_fn = function()
            # ∂L/∂A = ∂L/∂C @ B^T
            # ∂L/∂B = A^T @ ∂L/∂C
            
            if a.requires_grad
                grad_a = output.grad * transpose(captured_b_data)
                a.grad .+= grad_a
            end
            
            if b.requires_grad
                grad_b = transpose(captured_a_data) * output.grad
                b.grad .+= grad_b
            end
        end
        
        output.parents = [a, b]
    end
    
    return output
end

"""
Transpose tensor along two axes
"""
function Base.transpose(a::Tensor)
    if ndims(a) != 2
        error("transpose requires 2D tensor, got $(ndims(a))D")
    end
    
    # Convert to concrete array (transpose returns lazy wrapper)
    output_data = Matrix(transpose(a.data))
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        output.backward_fn = function()
            if a.requires_grad
                # Gradient flows through by transposing back
                grad_a = transpose(output.grad)
                a.grad .+= grad_a
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

"""
Reshape tensor to new shape
"""
function Base.reshape(a::Tensor, new_shape::Tuple)
    if prod(new_shape) != prod(size(a))
        error("Total elements must match: $(prod(size(a))) vs $(prod(new_shape))")
    end
    
    output_data = reshape(a.data, new_shape)
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_original_shape = size(a)
        
        output.backward_fn = function()
            if a.requires_grad
                # Gradient flows back by reshaping to original shape
                grad_a = reshape(output.grad, captured_original_shape)
                a.grad .+= grad_a
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

# ============================================================================
# REDUCTION OPERATIONS
# ============================================================================

"""
Sum tensor along specified dimension or all elements
"""
function Base.sum(a::Tensor; dims=nothing, keepdims=false)
    if dims === nothing
        output_data = [Base.sum(a.data)]
    else
        output_data = Base.sum(a.data; dims=dims)
        if !keepdims
            output_data = dropdims(output_data, dims=dims)
        end
    end
    
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_a_shape = size(a)
        captured_dims = dims
        captured_keepdims = keepdims
        
        output.backward_fn = function()
            if a.requires_grad
                # Gradient broadcasts back to original shape
                grad_a = output.grad
                
                if !captured_keepdims && captured_dims !== nothing
                    # Add back the reduced dimension as size 1
                    new_shape = collect(size(grad_a))
                    insert!(new_shape, captured_dims, 1)
                    grad_a = reshape(grad_a, tuple(new_shape...))
                end
                
                # Broadcast to original shape
                grad_broadcasted = zeros(Float64, captured_a_shape)
                grad_broadcasted .+= grad_a
                a.grad .+= grad_broadcasted
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

"""
Mean of tensor along specified dimension
"""
function Statistics.mean(a::Tensor; dims=nothing, keepdims=false)
    sum_result = sum(a, dims=dims, keepdims=keepdims)
    
    if dims === nothing
        count = length(a.data)
    else
        count = size(a.data, dims)
    end
    
    return sum_result / count
end

# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

"""
ReLU activation: max(0, x)
"""
function relu(a::Tensor)
    output_data = max.(0.0, a.data)
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_mask = a.data .> 0.0
        
        output.backward_fn = function()
            if a.requires_grad
                # Gradient passes through where input > 0
                grad_a = output.grad .* captured_mask
                a.grad .+= grad_a
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

"""
Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
"""
function sigmoid(a::Tensor)
    output_data = 1.0 ./ (1.0 .+ exp.(-a.data))
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_output_data = copy(output_data)
        
        output.backward_fn = function()
            if a.requires_grad
                # ∂σ/∂x = σ(x) * (1 - σ(x))
                grad_a = output.grad .* captured_output_data .* (1.0 .- captured_output_data)
                a.grad .+= grad_a
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

"""
Tanh activation
"""
function Base.tanh(a::Tensor)
    output_data = tanh.(a.data)
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_output_data = copy(output_data)
        
        output.backward_fn = function()
            if a.requires_grad
                # ∂tanh/∂x = 1 - tanh²(x)
                grad_a = output.grad .* (1.0 .- captured_output_data .^ 2)
                a.grad .+= grad_a
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

"""
Softmax activation along specified dimension
"""
function softmax(a::Tensor; dims::Int=1)
    # Numerically stable softmax
    max_vals = maximum(a.data, dims=dims)
    exp_vals = exp.(a.data .- max_vals)
    sum_exp = sum(exp_vals, dims=dims)
    output_data = exp_vals ./ sum_exp
    
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_output_data = copy(output_data)
        captured_dims = dims
        
        output.backward_fn = function()
            if a.requires_grad
                # Jacobian: softmax_i * (δ_ij - softmax_j)
                sum_term = sum(output.grad .* captured_output_data, dims=captured_dims)
                grad_a = captured_output_data .* (output.grad .- sum_term)
                a.grad .+= grad_a
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

"""
Exponential function
"""
function Base.exp(a::Tensor)
    output_data = exp.(a.data)
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_output_data = copy(output_data)
        
        output.backward_fn = function()
            if a.requires_grad
                # ∂e^x/∂x = e^x
                grad_a = output.grad .* captured_output_data
                a.grad .+= grad_a
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

"""
Natural logarithm
"""
function Base.log(a::Tensor)
    if any(a.data .<= 0)
        error("Log requires positive values")
    end
    
    output_data = log.(a.data)
    output = Tensor(output_data, requires_grad=a.requires_grad)
    output.is_leaf = false
    
    if output.requires_grad && GRAD_ENABLED[]
        captured_a_data = copy(a.data)
        
        output.backward_fn = function()
            if a.requires_grad
                # ∂ln(x)/∂x = 1/x
                grad_a = output.grad ./ captured_a_data
                a.grad .+= grad_a
            end
        end
        
        output.parents = [a]
    end
    
    return output
end

# ============================================================================
# BACKWARD PASS ENGINE
# ============================================================================

"""
Compute all gradients via backpropagation
"""
function backward!(root::Tensor)
    if !root.requires_grad
        error("Root tensor must require gradients")
    end
    
    # Step 1: Build topological ordering
    topo_order = Tensor[]
    visited = Set{UInt64}()
    
    function build_topo(node::Tensor)
        node_id = objectid(node)
        if node_id ∉ visited && node.requires_grad
            push!(visited, node_id)
            for parent in node.parents
                build_topo(parent)
            end
            push!(topo_order, node)
        end
    end
    
    build_topo(root)
    
    # Step 2: Initialize root gradient
    if is_scalar(root)
        root.grad[1] = 1.0
    else
        root.grad .= ones(Float64, size(root.data))
    end
    
    # Step 3: Propagate gradients backward
    for node in reverse(topo_order)
        if node.backward_fn !== nothing
            node.backward_fn()
        end
    end
end

"""
Reset all gradients in computation graph
"""
function zero_grad!(root::Tensor)
    visited = Set{UInt64}()
    
    function zero_recursive(node::Tensor)
        node_id = objectid(node)
        if node_id ∉ visited
            push!(visited, node_id)
            node.grad .= 0.0
            for parent in node.parents
                zero_recursive(parent)
            end
        end
    end
    
    zero_recursive(root)
end

"""
Context for disabling gradient tracking
Usage: @no_grad begin ... end
"""
const GRAD_ENABLED = Ref(true)

macro no_grad(expr)
    quote
        old_state = GRAD_ENABLED[]
        GRAD_ENABLED[] = false
        try
            $(esc(expr))
        finally
            GRAD_ENABLED[] = old_state
        end
    end
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
Display tensor information
"""
function Base.show(io::IO, t::Tensor)
    println(io, "Tensor(")
    println(io, "  data: ", t.data)
    if t.requires_grad && any(t.grad .!= 0)
        println(io, "  grad: ", t.grad)
        end
    println(io, "  requires_grad: ", t.requires_grad)
    if !isempty(t.name)
        println(io, "  name: ", t.name)
    end
    print(io, ")")
end

# Export main functions
export Tensor, backward!, zero_grad!, @no_grad
export matmul, relu, sigmoid, tanh, softmax
export sum_to_shape, broadcast_shapes

println("✓ Part 1: Tensor and Autodiff System loaded successfully")