module TotalLeastSquares
export tls, tls!, wtls, wls, rtls, irls, sls, rowcovariance, hankel, ishankel, unhankel, flts
export rpca, lowrankfilter, rpca_ga, entrywise_median, entrywise_trimmed_mean, μ!
using FillArrays, Printf, LinearAlgebra, SparseArrays, Statistics, StatsBase

include("flts.jl")

"""
    wls(A, y, Σ)
    wls(A, y, C::Cholesky)

Solves the weigted standard least-squares problem Ax = y + e, e ~ N(0,Σ)
# Arguments
- `A ∈ R(n,u)` Design matrix
- `y ∈ R(n)` RHS
- `Σ ∈ R(n,n)` Covariance matrix of the residuals (can be sparse or already factorized).
- `C` A Cholesky factor of the weight may be provided instead of `Σ`
"""
function wls(A, y, C::Cholesky)
    R = UpperTriangular(qr(C.L\A).R)
    R\(R'\(A'*(C\y)))
end

"""
    wls!(zA, A, y, w::Vector)

Overwrites `zA` *and* `w`.
- `zA`: storage of same size as `A`
"""
function wls!(zA,A,y,w)
    zA .= A ./ w
    w  .= y ./ w
    Symmetric(A'*zA)\(A'w)
end

wls(A,y,Σ::Union{AbstractMatrix, SparseMatrixCSC}) = wls(A,y,cholesky(Hermitian(Σ)))

"""
    tls(A,y)

Solves the total least-squares problem Ax=y using the SVD method
# Arguments
- `A` Design matrix
- `y` RHS
"""
function tls(A::AbstractArray,y::AbstractArray)
    AA  = [A y]
    s   = svd(AA) # Not in-place for Zygote compatability
    n   = size(A,2)
    V21 = s.V[1:n,n+1:end]
    V22 = s.V[n+1:end,n+1:end]
    x   = -V21/V22
end

"""
    tls!(Ay::AbstractArray, n::Integer)
    tls!(s::SVD, n::Integer)

Inplace version of `tls`. `Ay` is `[A y]` and `n` is the number of columns in `A`. Also accepts a precomputed SVD of `Ay`.
"""
tls!(Ay::AbstractArray, n::Integer)  = tls!(svd!(Ay), n)

function tls!(s::SVD, n::Integer)
    V21 = s.V[1:n,n+1:end]
    V22 = s.V[n+1:end,n+1:end]
    x   = -V21/V22
end


a ⊗ b = kron(a,b)

"""
    x = wtls(A,y,Qaa,Qay,Qyy; iters = 10)

Solves min nᵀQ⁻¹n s.t. (A+E)x = y + v
where Q = [Qaa Qay; Qay' Qyy], n = [vec(E); v]

Uses algorithm 1 from
Weighted total least squares: necessary and sufficient conditions,
fixed and random parameters, Fang 2013
https://link.springer.com/article/10.1007/s00190-013-0643-2


# Arguments
- `A` Design matrix
- `y` RHS
- `Qaa` Covariance matrix of `e = vec(E)`
- `Qay` Covariance between A and y

# Keyword Arguments
- ` iters = 10` Number of iterations
"""
function wtls(A,y,Qaa,Qay,Qyy; iters = 10)
    n,u    = size(A)
    eyef   = issparse(Qaa) ? n->SparseMatrixCSC(1I,n,n) : Eye
    Iₙ,Iᵤ  = eyef(n),eyef(u)
    x      = wls(A,y,Qyy) # Initialize with weigted LS slution
    QₐₐQₐy = [Qaa Qay]
    QΠ     = [Qaa Qay; [Qay' Qyy]]
    for i = 1:iters
        B    = [(x' ⊗ Iₙ) -Iₙ]
        BQBᵀ = factorize(Symmetric(B*QΠ*B'))
        λ    = BQBᵀ\(y-A*x)
        v    = QₐₐQₐy*B'λ
        x    = (A'*(BQBᵀ\A))\((Iᵤ ⊗ λ')*v + A'*(BQBᵀ\y))
    end
    x
end

"""
    Qaa,Qay,Qyy = rowcovariance(rowQ::AbstractVector{<:AbstractMatrix})

Takes row-wise covariance matrices `QAy[i]` and returns the full (sparse) covariance matrices. `rowQ = [cov([A[i,:] y[i]]) for i = 1:length(y)]`
"""
function rowcovariance(rowQ::AbstractVector{<:AbstractMatrix})
    n = length(rowQ)
    u = size(rowQ[1],1)-1
    Qaa,Qay,Qyy = spzeros(n*u,n*u), spzeros(n*u,n), Diagonal(zeros(n))
    for (i,Q) = enumerate(rowQ)
        for col = 1:u
            for row = 1:u
                aind1 = i + (row-1)*n
                aind2 = i + (col-1)*n
                Qaa[aind1,aind2] = Q[row,col]
            end
        end
        yind = i
        for row = 1:u
            aind = i + (row-1)*n
            Qay[aind,yind] = Q[row,end]
        end
        Qyy[yind,yind] = Q[end,end]
    end
    Qaa,Qay,Qyy
end

include("robustPCA.jl")


"""
    rtls(A,y; nukeA=false, kwargs...)

Solves a robust total least-squares problem Ax=y using the robust PCA method of "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices", https://people.eecs.berkeley.edu/~yima/psfile/Lin09-MP.pdf

# Arguments
- `A` Design matrix
- `y` RHS
- `kwargs` are passed to [`rpca`](@ref), see the docstring for [`rpca`](@ref) for more help.
"""
function rtls(A::AbstractArray,y::AbstractArray; kwargs...)
    AA  = [A y]
    Ah,Eh,s = rpca(AA; nukeA=false, kwargs...)
    tls!(s,size(A,2))
end

# function rtls_ga(A::AbstractArray,y::AbstractArray; μ = μ!, kwargs...)
#     AA  = [A y]
#     m = mean(AA, dims=1)
#     AA .-= m
#     V = rpca_ga(AA'; μ = μ, kwargs...)
#     n   = size(A,2)
#     V21 = V[1:n,n+1:end]
#     V22 = V[n+1:end,n+1:end]
#     x   = -V21/V22
# end

"""
    irls(A, y; tolx=0.001, tol=1.0e-6, verbose=false, iters=100)

Iteratively reweighted least squares. Solves
minimizeₓ ||Ax-y||₁

#Arguments:
- `A`: Design matrix
- `y`: Measurements
- `tolx`: Minimum change in `x` before quitting
- `tol`: Minimum change in error before quitting
- `iters`: Maximum number of iterations
"""
function irls(A,y; tolx=1e-4, tol=1e-6, verbose=false, iters=100)
    x = A\y
    zA = similar(A)
    w = similar(y)
    mul!(w,A,x) # w used for storage
    xold = copy(x)
    e = abs.(y - w)
    me = median(e)
    meold = me
    verbose && println("0: Error: ", round(me, sigdigits=3))
    w .= max.(e, 1e-5)
    for i = 1:iters
        x = wls!(zA,A,y,w)
        mul!(w,A,x) # w used for storage
        e .= abs.(y .- w)
        w .= max.(e, 1e-5)
        d = norm(x-xold)/norm(x)
        me = median(e)
        verbose && println("$i: Error: ", round(me, sigdigits=3), " relative step: ", round(d, sigdigits=3))
        if d < tolx || meold-me < tol
            verbose && println("Success")
            return x
        end
        meold = me
        xold .= x
    end
    verbose && println("Maximum number of iterations reached")
    x
end


"""
    sls(A, y; r = 1, iters = 100, verbose = false, tol = 1.0e-8, α0 = 0.1)

Simplex least-squares: minimizeₓ ||Ax-y||₂ s.t. sum(x) = r

# Arguments:
- `A`: Design matrix
- `y`: RHS
- `r`: Simplex radius. the default (1) is the probability simplex.
- `iters`: Maximum number of iterations of projected gradient
- `verbose`: Print stuff
- `tol`: Tolerance (change in x between iterations).
- `α0`: initial step size. This parameter influences speed of convergence and also to which point the algorithm converges.
"""
function sls(A, y; r=1, iters=100, verbose=false, tol=1e-8, α0 = 0.1)

    x = A\y
    proj_simplex!(x; r=r)
    xo = copy(x)
    g = similar(x)
    e = similar(y)
    verbose && @info "Iter 0 cost: $(norm(A*x-y))"
    for iter = 1:iters
        mul!(e,A,x)
        e .= y .- e
        mul!(g, A', e)
        ng = norm(g)
        α = α0 / sqrt(iter)
        x .+= α .* g
        proj_simplex!(x; r=r)
        step = sqrt(sum(abs2(x-xo) for (x,xo) in zip(x,xo)))
        verbose && @info "Iter $iter norm(g): $ng norm(x-xo): $step, cost: $(norm(e))"
        step < tol && break
        xo .= x

    end
    verbose &&  @info "Converged - cost function: $(norm(e))"
    x
end

"""
    proj_simplex!(x; iters = 1000, r = 1, tol = 1.0e-8)

Project x onto the simplex with radius `r` such that `sum(x) = r` and all `x >= 0`
"""
function proj_simplex!(x; iters=1000, r=1, tol=1e-8)
    μ = minimum(x) - r
    for iter = 1:iters
        cost = sum(max(x - μ, 0)  for x in x) - r
        df   = sum(-((x - μ) > 0) for x in x)
        μ   -= cost / df
        abs(cost) < tol && break
    end
    @. x = max(x - μ, 0)
end


end # module
