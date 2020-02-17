using ProximalOperators
@inline soft_th(x, ϵ) = max(x-ϵ,zero(x)) + min(x+ϵ,zero(x))
@inline soft_th(x, ϵ, l) = max(x-ϵ,l) + min(x+ϵ,l) - l

function soft_toeplitz!(A, ϵ)
    @inbounds for i in 0:size(A,1)-2
        di = diagind(A,-i)
        m = sum(A[j] for j in di)/length(di)
        for di in di
            A[di] = soft_th(A[di], ϵ, m)
        end
    end
    @inbounds for i in 1:size(A,2)-2
        di = diagind(A,i)
        m = sum(A[j] for j in di)/length(di)
        for di in di
            A[di] = soft_th(A[di], ϵ, m)
        end
    end
end

"""
    A,E,s,sv = rpca(D::Matrix; λ=1.0 / √(maximum(size(D))), iters=1000, tol=1.0e-7, ρ=1.5, verbose=false, nonnegA=false, nonnegE=false, nukeA=true)

minimize_{A,E} ||A||_* + λ||E||₁ s.t. D = A+E

`s` is the last calculated svd of `A` and `sv` is the estimated rank.

Ref: "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices", Zhouchen Lin, Minming Chen, Leqin Wu, Yi Ma, https://people.eecs.berkeley.edu/~yima/psfile/Lin09-MP.pdf
Significant inspiration taken from an early implementation by Ryuichi Yamamoto in RobustPCA.jl

#Arguments:
- `D`: Design matrix
- `λ`: Sparsity regularization
- `iters`: Maximum number of iterations
- `tol`: Tolerance
- `ρ`: Algorithm tuning param
- `verbose`: Print status
- `nonnegA`: Hard thresholding on A
- `nonnegE`: Hard thresholding on E
- `nukeA`: Activate the nuclear penalty on `A`, if `false`, then `A` is not assumed to be low rank.
- `toeplitz`: Indicating whether or not `D` (and thus `A` and `E`) are Toeplitz matrices (constant diagonals). If this fact is known, the expected performance of this alogorithm goes up. If the matrix `D` is Hankel (constant antidiagonals) you may reverse the second dimension, i.e., `Dᵣ = D[:,end:-1:1]`. `toeplitz=true` should likely be paired with `nukeA=false`.
"""
function rpca(D::AbstractMatrix{T};
                          λ              = T(1.0/sqrt(maximum(size(D)))),
                          iters::Int     = 1000,
                          tol            = sqrt(eps(T)),
                          ρ              = T(1.5),
                          verbose::Bool  = false,
                          nonnegA::Bool  = false,
                          nonnegE::Bool  = false,
                          toeplitz::Bool = false,
                          proxE          = NormL1(λ),
                          nukeA          = true) where T

    M, N      = size(D)
    d         = min(M,N)
    A, E      = zeros(T, M, N), zeros(T, M, N)
    Z         = similar(D)
    Y         = copy(D)
    norm²     = svdvals(Y)[1] # can be tuned
    norm∞     = norm(Y, Inf) / λ
    dual_norm = max(norm², norm∞)
    d_norm    = opnorm(D)
    Y       ./= dual_norm
    μ         = T(1.25) / norm²
    μ̄         = μ  * T(1.0e+7)
    sv        = 10
    local s, svp
    for k = 1:iters
        prox!(E, proxE, D .- A .+ (1/μ) .* Y, 1/μ)
        # E .= soft_th.(D .- A .+ (1/μ) .* Y, λ/μ)
        if toeplitz
            soft_toeplitz!(E, λ/μ)
        end
        if nonnegE
            E .= max.(E, 0)
        end
        s = svd(Z .= D .- E .+ (1/μ) .* Y) # Z assignment just for storage
        U,S,V = s
        svp = sum(>=(1/μ), S)
        if svp < sv
            sv = svp # min(svp + 1, N) # the paper says to use these formulas but sv=svp works way better
        else
            sv = svp # min(svp + round(Int, T(0.05) * d), d)
        end

        if nukeA
            A .= U[:,1:sv] * Diagonal(S[1:sv] .- 1/μ) * V[:,1:sv]'
        else
            A .= U[:,1:sv] * Diagonal(S[1:sv]) * V[:,1:sv]'
        end
        if toeplitz
            soft_toeplitz!(A, λ/μ)
        end
        if nonnegA
            A .= max.(A, 0)
        end

        @. Z = D - A - E # Z are the reconstruction errors
        @. Y = Y + μ * Z
        μ = min(μ*ρ, μ̄)

        cost = opnorm(Z) / d_norm
        verbose && println("$(k) cost: $(round(cost, sigdigits=4))")

        if cost < tol
            verbose && println("converged")
            break
        end
        k == iters && @warn "Maximum number of iterations reached, cost: $cost, tol: $tol"
    end

    A, E, s, sv
end
