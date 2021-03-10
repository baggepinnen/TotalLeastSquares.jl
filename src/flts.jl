"""
    flts(A::AbstractArray{<:Real}, y::Vector{<:Real})

Fast least trimmed squares by an algorithm from Rousseeuw et al. - Computing LTS Regression for Large Data Sets (1999)
The basic idea is to find an outlier-free subset `H` of regressors that minimizes the sum of squared residuals `Q`, resulting in a robust version of least squares up to 50% outliers

# Args

* A::AbstractArray{<:Real}: Vector or Matrix ∈ R(n x p) containing regressors 
* y::Vector{<:Real}: Output Vector ∈ R(n)

# Keyword arguments

* h::Int: The size of the subset H, has to be between 0.5(n + p + 1)) <= h <= n and defaults to 0.5(n + p + 1)
* outliers::Real: Amount of outliers, between 0 <= outliers <= 0.5. A valid definition of h overwrites an outlier definition!
* N::Int: Number of initial p-subsets, defaults to 500 (minimum is 10). This is the main parameter regarding speed, to estimate a reasonable N, the probability of having at least one outlier free p-subset is given by: 1-(1-(1-outliers)^p)^N > 0. If verbose = true information about this gets printed.
* maxiter::Int: Maximum number of C-steps applied in the final optimization, defaults to 100
* ΔQmin::Real: Minimum change of the objective function (squared residuals) that defines convergence
* return_set::Bool: If true, the subset H and the regarding value Q are also returned in the form of (H, θ, Q)
* verbose::Bool: Print some more information

# Return
The parameters θ that minimize Q and optionally a tuple (H, θ, Q), which also contains the selected subset H and the minimal Q

# Example
Taken from the original paper
```jldoctest
julia> x = rand(Normal(0,10), 1000)
julia> y = x .+ 1 + randn(1000) 
julia> y[801:end] = rand(Normal(0,5), 200) 
julia> x[801:end] = rand(Normal(50,5), 200) 
julia> A = [x ones(1000)]
julia> flts(xb,y, return_set = true)

([656, 336, 293, 763, 281, 248, 269, 231, 678, 372  …  734, 220, 434, 658, 174, 465, 745, 371, 759, 732], [1.0003968687209952, 1.0058265326498548], 750.0607838759548)
```
"""
function flts(A::AbstractArray{<:Real}, y::AbstractVector{<:Real};
    h::Integer = 0, outliers::Real = -1, N::Integer = 500, maxiter::Integer = 100, ΔQmin::Real = 0.0001, return_set::Bool = false, verbose::Bool = false)
    # Input checking
    n = length(y)
    size(A, 1) == n || throw(DimensionMismatch("Both inputs A and y should have the same number of rows"))
    N >= 10 || throw(DomainError("N needs to be >= 10"))
    if ndims(A) == 2
        p = size(A, 2)
    else
        # make sure matrix multiplication works
        p = 1
        A = reshape(A, :, 1)
    end

    # Get the subset length h 
    if (round(0.5(n + p + 1)) <= h <= n)
        verbose && @info "h was set to: h = $h. Breakdown point is at ~$(round(100 - h / n * 100))% outliers"
    elseif (0.0 <= outliers <= 0.5)
        h = Int(round((1-outliers)*n))
        verbose && @info "h was set to: h = $h. Breakdown point is at ~$(round(100 - h / n * 100))% outliers"
    else   
        h = Int(round(0.5(n + p + 1)))
        verbose && @info "h was set to default: h = $h. Breakdown point is at ~$(round(100 - h / n * 100))% outliers"
    end
    # Information about possible success
    verbose && @info "Chance to find an outlier-free p subset is at least ~$((1-(1-(h/n)^p)^N) * 100) %"

    # Initial subset H1 using p-subsets (Option b from paper)
    hs = fill(h, N)
    f(h) = get_initial_H(A, y, p, n, h)
    # Array of Tuples (H, θ, Q)
    initials = map(f, hs)

    # Carry out two C-steps, take top 10 candidates for further optimization
    g1(x) = optimize_H(A, y, h, x, 2, 0)
    opts = map(g1, initials)
    sort!(opts, by = x -> x[3])
    candidates = opts[1:10]

    # Optimze until convergence and return the best
    g2(x) = optimize_H(A, y, h, x, maxiter, ΔQmin)
    results = map(g2, candidates)
    sort!(results, by = x -> x[3])

    # Return the winner
    winner = results[1]
    if return_set
        return winner
    else
        return winner[2]
    end
end

# Helper function that applies a certain number of C-steps to a subset H and checks for convergence
function optimize_H(A, y, h::Int, initial, maxiter::Int, ΔQmin)
    Q_old = initial[3]
    # Create object in outer scope
    opt = nothing
    # Apply C-steps until convergence / maxiter
    for i in 1:maxiter
        opt = C_step(A, y, initial[2], h)
        ΔQ = Q_old - opt[3]
        if ΔQ < ΔQmin
            break
        end
        Q_old = opt[3]
    end    
    return opt
end

# Helper function that gets an initial set H from a random p subset
function get_initial_H(A, y, p, n, h)
    inds = 1:n
    # sample a random p subset
    J = sample(inds, p, replace = false)
    # if subset does not specify a unique hyperplane add random observation
    i = 1
    while ((p+i+1) < n) && (rank(A[J, :]) < p)
        J = sample(inds, p+i, replace = false)
        i += 1
    end

    # Estimate first parameters
    θ_J = \(A[J, :], y[J])
    # Use a C-step for an initial H, θ, Q
    H_initial, θ_initial, Q_initial = C_step(A, y, θ_J, h)
    return H_initial, θ_initial, Q_initial
end

# The C-step, workinghorse that guarentees converging parameters
function C_step(A, y, θ_old, h)
    # Use the old parameters to get the residuals
    residuals = y .- A * θ_old
    # Sort the residuals and return a new subset with the smallest residuals
    # H_new = sortperm(abs.(residuals))[1:h] # should be more efficient, but apperantly is not
    H_new = sortperm(abs.(residuals))[1:h]
    # Get new parameters using oridinary least squares 
    θ_new = \(A[H_new, :], y[H_new])
    Q_new = get_Q(A, y, H_new, θ_new)
    return H_new, θ_new, Q_new
end

# get the objective function Q
function get_Q(A, y, H, θ)
    residuals = y .- A * θ
    return sum(abs2, residuals[H])
end
