# Pranav's JuliaML.jl

This is my attempt at understanding how AI and ML works by building it from the ground up.

In its current state, this is simply a side project. You should not attempt to develop with this or run it in production because it sucks. Use [*Flux.jl*](https://github.com/FluxML/Flux.jl) or [*MLJ.jl*](https://github.com/JuliaAI/MLJ.jl) if you actually need AI/ML in Julia.

My main goal for this project is to have all the foundations of AI covered. However, if I feel like it, I may also try to develop functionality for [Physics Informed Machine Learning](https://blogs.mathworks.com/deep-learning/2025/06/23/what-is-physics-informed-machine-learning/).

I did also use AI to write code throughout the project. Interestingly the little imperfections and screw-ups it does make for a great learning experience. Debugging the errors is a great way to get a handle on how Julia works as a language and all its little quirks.

## Installation
### Install Julia (if needed)
Go to the [Julia Install documentation](https://julialang.org/install/) and follow their instructions. Verify your installation by running `julia -v`

### Install Git (if needed)
Follow the [Git documentation](https://git-scm.com/install/). Verify your installation with `git -v`

### Clone the repo and install deps
```
git clone https://github.com/s9kt/julia-ml
cd julia-ml
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Test and mess around
`julia examples/linear_regression.jl`