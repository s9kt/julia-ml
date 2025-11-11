using Random
using LinearAlgebra
using Statistics 
# Assuming includes like:
include("../tensor.jl")
include("../nn.jl")

# ============================================================================
# 1. MODEL DEFINITION
# ============================================================================

# Define a simple, three-layer fully connected neural network
# Input size: 10 features
# Hidden size: 20 features
# Output size: 3 (e.g., for a 3-class classification problem)
model = Sequential(
    # Layer 1: Linear transformation (10 -> 20)
    Linear(10, 20, use_bias=true),
    
    # Non-linearity
    ReLU(),
    
    # Regularization (Dropout probability = 0.5)
    Dropout(0.5),

    # Layer 2: Linear transformation (20 -> 3)
    Linear(20, 3, use_bias=true),
    
    # Output activation
    Softmax(dims=2) # Softmax over the feature dimension (columns)
)

println("--- Model Initialized ---")
# Print a summary of the model structure and parameter count
summary(model)


# ============================================================================
# 2. DATA PREPARATION
# ============================================================================

# Create a batch of input data (Batch size = 5, Features = 10)
X = Tensor(randn(5, 10), requires_grad=false, name="Input_Data")
println("\nInput Tensor X: $(size(X))")

# ============================================================================
# 3. FORWARD PASS
# ============================================================================

# Run the data through the network
Y = model(X)

println("\n--- Forward Pass ---")
println("Output Tensor Y: $(size(Y))")
println("First row of output (probabilities): $(Y.data[1, :])")


# ============================================================================
# 4. MODE SWITCHING DEMO (Dropout/BatchNorm dependent)
# ============================================================================

println("\n--- Mode Switching Demo ---")

# The model is in 'train' mode by default (Dropout is active)
println("Is model training? $(is_training(model))") 

# Switch to evaluation mode (Dropout becomes inactive)
eval!(model)
println("Switching to eval mode...")
println("Is model training now? $(is_training(model))") 

# Switch back to training mode
train!(model)
println("Switching back to train mode...")
println("Is model training now? $(is_training(model))") 


# ============================================================================
# 5. BACKWARD PASS (TRAINING STEP SIMULATION)
# ============================================================================

# 5a. Clear previous gradients (essential for training loops)
zero_grad!(model)
println("\n--- Gradients Cleared ---")
println("Weight gradient before backprop (L1): $(model.layers[1].weight.grad[1])")


# 5b. Define a simple loss function (Sum of squares for demonstration)
# We need a scalar loss to start the backpropagation
loss = sum(Y ^ 2)

println("\n--- Backward Pass ---")
println("Scalar Loss: $(loss.data[1])")

# 5c. Perform backpropagation
# This populates the '.grad' attribute for all Tensors that require gradients
backward!(loss)

println("Backpropagation complete.")

# 5d. Inspect gradients on parameters
l1_weight = model.layers[1].weight
l2_weight = model.layers[4].weight

println("\n--- Gradient Inspection ---")
println("L1 Weight gradient (first element): $(l1_weight.grad[1])")
println("L2 Bias gradient (first element): $(l2_weight.grad[1])")
println("L2 Bias gradient (second element): $(model.layers[4].bias.grad[2])")

# Note: The output gradients are non-zero, indicating that the automatic 
# differentiation system successfully calculated the derivatives 
# from the scalar loss back to the initial learnable parameters.