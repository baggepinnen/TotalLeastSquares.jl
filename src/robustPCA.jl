@inline soft_th(x, ϵ) = max(x-ϵ,zero(x)) + min(x+ϵ,zero(x))
@inline soft_th(x, ϵ, l) = max(x-ϵ,l) + min(x+ϵ,l) - l
@inline function soft_th(x::Complex, ϵ)
    m,a = abs(x), angle(x)
    m = max(m-ϵ,zero(m)) + min(m+ϵ,zero(m))
    m*cis(a)
end

function soft_hankel!(A, ϵ)
    K,L = size(A)
    N = K+L-1
    for k = 1:N
        ri = min(K,k):-1:max(k-L,1)
        ci = max(1,k-K+1):L
        m = mean(A[r,c] for (r,c) in zip(ri,ci))
        for (r,c) in zip(ri,ci)
            A[r,c] = soft_th(A[r,c], ϵ, m)
        end
    end
    A
end

"""
    unhankel(A)

The inverse of [`hankel`](@ref). Create a 1-D signal by antidiagonal averaging
"""
function unhankel(A)
    K,L = size(A)
    N = L+(K-1)
    y = similar(A, N)
    for k = 1:N
        ri = min(K,k):-1:max(k-L,1)
        ci = max(1,k-K+1):L
        m = mean(A[r,c] for (r,c) in zip(ri,ci))
        y[k] = m
    end
    y
end


"""
    unhankel(A, lag, N, D=1)

The inverse of [`hankel`](@ref). Create a 1-D signal by antidiagonal averaging

#Arguments:
- `A`: A Hankel matrix
- `lag`: if lag was used to create `A`, you must provide it to `unhankel`
- `N`: length of the original signal
- `D`: dimension of the original signal
"""
function unhankel(A,lag,N,D=1)
    lag == 1 && D == 1 && (return unhankel(A))
    K      = size(A,1)
    L      = size(A,2)÷D
    y      = zeros(eltype(A), N, D)
    counts = zeros(Int, N, D)
    indmat = CartesianIndex.(1:N, (1:D)')
    inds   = hankel(indmat, L, lag)
    for (Aind,yind) in enumerate(inds)
        y[yind] += A[Aind]
        counts[yind] += 1
    end
    y ./= max.(counts, 1)
    D == 1 && (return vec(y))
    y
end

"""
    X = hankel(x,L,lag=1)

Form a hankel "trajectory matrix" `X` of size KxL, K = N-L+1
x can be a vector or a matrix.
"""
function hankel(x,L,lag=1)
    N = size(x,1)
    D = size(x,2)
    @assert L <= N/2 "L has to be less than N/2 = $(N/2)"
    @assert lag <= L "lag must be <= L"
    K = (N-L)÷lag+1
    X = similar(x, K, L*D)
    for d in 1:D
        inds = 1:L
        colinds = d:D:L*D
        for k = 1:K
            X[k,colinds] = x[inds,d]
            inds = inds .+ lag
        end
    end
    X
end

function ishankel(A)
    K,L = size(A)
    N = K+L-1
    for k = 1:N
        ri = min(K,k):-1:max(k-L,1)
        ci = max(1,k-K+1):L
        val = A[ri[1],ci[1]]
        for (r,c) in zip(ri,ci)
            A[r,c] != val && return false
        end
    end
    true
end


"""
    yf = lowrankfilter(y, n=min(length(y) ÷ 20, 2000); tol=1e-3, kwargs...)

Filter time series `y` by forming a lag-embedding T (a Toeplitz matrix) and using [`rpca`](@ref) to recover a low-rank matrix from which the a filtered signal `yf` can be extracted. The size of the embedding `n` determines the complexity, higher `n` generally gives better filtering at the cost of roughly cubic complexity.

#Arguments:
- `y`: A signal to be filtered, assumed corrupted with sparse noise
- `n`: Embedding size
- `kwargs`: See [`rpca`](@ref) for keyword arguments.
"""
function lowrankfilter(y, n=min(size(y,1)÷20,2000); sv=0, lag=1, tol=1e-3, kwargs...)
    H = hankel(y, n, lag)
    if sv <= 0
        A,E = rpca(H; tol=tol, kwargs...)
    else
        s = svd(H)
        A = s.U[:,1:sv] * Diagonal(s.S[1:sv]) * s.Vt[1:sv,:]
    end
    unhankel(A, lag, size(y,1), size(y,2))
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
- `hankel`: Indicating whether or not `D` (and thus `A` and `E`) are Hankel matrices (constant anti diagonals). If this fact is known, the expected performance of this alogorithm goes up. If the matrix `D` is Toeplitz (constant diagonals) you may reverse the second dimension, i.e., `Dᵣ = D[:,end:-1:1]`. `hankel=true` should likely be paired with `nukeA=false`.

To speed up convergence you may either increase the tolerance or increase `ρ`. Increasing `tol` is often the best solution.
"""
function rpca(D::AbstractMatrix{T};
                          λ              = real(T)(1.0/sqrt(maximum(size(D)))),
                          iters::Int     = 1000,
                          tol            = sqrt(eps(real(T))),
                          ρ              = real(T)(1.5),
                          verbose::Bool  = false,
                          nonnegA::Bool  = false,
                          nonnegE::Bool  = false,
                          hankel::Bool   = false,
                          # proxE          = NormL1(λ),
                          nukeA          = true) where T
    RT        = real(T)
    M, N      = size(D)
    d         = min(M,N)
    A, E      = zeros(T, M, N), zeros(T, M, N)
    Z         = similar(D)
    Y         = copy(D)
    norm²     = opnorm(Y)::RT # can be tuned
    norm∞     = norm(Y, Inf) / λ
    dual_norm = max(norm², norm∞)
    d_norm    = norm²
    Y       ./= dual_norm
    μ         = RT(1.25) / norm²
    μ̄         = μ  * RT(1.0e+7)
    sv        = 10
    local s, svp
    for k = 1:iters
        # prox!(E, proxE, D .- A .+ (1/μ) .* Y, 1/μ)
        E .= soft_th.(D .- A .+ (1/μ) .* Y, λ/μ)
        if hankel
            soft_hankel!(E, λ/μ)
        end
        if nonnegE
            E .= max.(E, 0)
        end
        s = svd!(Z .= D .- E .+ (1/μ) .* Y) # Z assignment just for storage
        svp = sum(x-> x >= (1/μ), s.S)::Int
        if svp < sv
            sv = svp # min(svp + 1, N) # the paper says to use these formulas but sv=svp works way better
        else
            sv = svp # min(svp + round(Int, T(0.05) * d), d)
        end

        @views if nukeA
            # A .= s.U[:,1:sv] * Diagonal(s.S[1:sv] .- 1/μ) * s.Vt[1:sv,:]
            mul!(Z[:,1:sv], s.U[:,1:sv], Diagonal(s.S[1:sv] .- 1/μ))
            mul!(A, Z[:,1:sv], s.Vt[1:sv,:])
        else
            # A .= s.U[:,1:sv] * Diagonal(s.S[1:sv]) * s.Vt[1:sv,:]
            mul!(Z[:,1:sv], s.U[:,1:sv], Diagonal(s.S[1:sv]))
            mul!(A, Z[:,1:sv], s.Vt[1:sv,:])
        end
        if hankel
            soft_hankel!(A, λ/μ)
        end
        if nonnegA
            A .= max.(A, 0)
        end

        @. Z = D - A - E # Z are the reconstruction errors
        @. Y = Y + μ * Z # Y can not be moved below as it depends on μ which is changed below
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

"""
    Q = rpca_ga(X::AbstractMatrix{T}, r=minimum(size(X)), U=similar(X); μ = μ!(s,w,U), verbose=false, kwargs...) where T

"Grassmann Averages for Scalable Robust PCA", Hauberg et al.
http://www2.compute.dtu.dk/~sohau/papers/cvpr2014a/Hauberg_CVPR_2014.pdf

#Arguments:
- `X`: Data matrix
- `r`: Rank (number of components to estimate
- `U`: Optional pre-allocated buffer
- `verbose`: print status
- `kwargs`: such as `tol=1e-7`, `iters=1000`
- `μ = μ!(s,w,U)` is a function that calculates the spherical average of a all columns in the matrix `U`, weighted by `w` and stores the result in `s`. The default is the standard weighted average. To get a robust estimate, consider using a robust average, such as `entrywise_trimmed_mean` or `entrywise_median` etc.
"""
function rpca_ga(X::AbstractMatrix{T}, r=minimum(size(X)), U = similar(X); verbose = false, kwargs...) where T
    d,N    = size(X)
    X      = copy(X)
    Xs1    = similar(X,1,N)
    Xs2    = similar(X)
    Q      = zeros(T, d, r)
    w      = zeros(T,N)
    Xnorms = zeros(T,N)
    for i = 1:r
        @inbounds @views for n = 1:N
            Xnorms[n] = sqrt(sum(abs2,X[:,n]))
            U[:,n] .=  X[:,n] ./ Xnorms[n]
        end
        q = rpca_ga_1(Xnorms, U, w; verbose = verbose, kwargs...)
        Q[:,i] .= q
        @static if VERSION >= v"1.3"
            mul!(Xs1, q', X)
            mul!(X, q,Xs1, -1, 1)
        else
            X .-= q*(q'X)
        end
    end
    Q
end

"""
Find the first principal component. This is an internal function used by `rpca_ga`.
"""
function rpca_ga_1(Xnorms,U::AbstractMatrix{T},w; tol=1e-7, iters=1000, verbose=false, μ=μ!) where {T}
    d,N = size(U)

    q = randn(d)
    q ./= norm(q)
    qold = copy(q)

    for i = 1:iters
        @inbounds @views for n = eachindex(w,Xnorms)
            w[n] = sign(U[:,n]'q)*Xnorms[n]
        end
        μᵢ = μ(q,w,U)
        q .= μᵢ./norm(μᵢ)
        dq = sqrt(sum(((x,y),)->abs2(x-y), zip(q,qold)))
        verbose && @info "Change at iteration $i: $dq"
        if dq < tol
            verbose && @info "Converged after $i iterations"
            break
        end
        qold .= q
        i == iters && @warn "Reached maximum number of iterations"
    end
    q
end

function μ!(s,w,U)
    ws = zero(eltype(w))
    s .= 0
    @views @inbounds @simd for n = 1:size(U,2)
        ws += w[n]
        s .+= w[n].*U[:,n]
    end
    s ./= ws
end

"""
    entrywise_trimmed_mean(s, w, U, P=0.1)

Remove `P` percent of the data on each side before computing the weighted mean.
"""
function entrywise_trimmed_mean(s,w,U, P=0.1)
    N = size(U,2)
    range = (1+floor(Int, P*N)):floor(Int, (1-P)*N)
    s .= 0
    @views for j = 1:size(U,1)
        I = sortperm(U[j,:])[range]
        s[j] += w[I]'U[j,I] / sum(w[I])
        # s[j] += sum(U[j,I])/length(I)
    end
    s
end
#
#
# function entrywise_trimmed_mean(s,w,U, P=0.05)
#     d,N = size(U)
#     range = (1+floor(Int, P*d)):floor(Int, (1-P)*d)
#     s .= 0
#     ws = zeros(size(s))
#     @views for j = 1:size(U,2)
#         I = sortperm(U[:,j])[range]
#         s[I] .+= w[j] .* U[I,j]
#         ws[I] .+= w[j]
#     end
#     s./ws
# end

function entrywise_median(s,w,U)
    s .= 0
    for j = 1:size(U,1)
        I = sortperm((w).*U[j,:])
        s[j] = sign(w[I[end÷2]])*U[j,I[end÷2]]#, StatsBase.Weights(abs.(w)))
        # s[j] = median(U[j,:])
    end
    s
end
