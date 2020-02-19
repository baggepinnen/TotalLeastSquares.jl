module TotalLeastSquares
export tls, tls!, wtls, wls, rtls, rowcovariance, toeplitz, istoeplitz, untoeplitz
export rpca, lowrankfilter, rpca_ga, entrywise_median, entrywise_trimmed_mean, μ!
using FillArrays, Printf, LinearAlgebra, SparseArrays, Statistics

"""
    wls(A,y,Σ)

Solves the weigted standard least-squares problem Ax = y + e, e ~ N(0,Σ)
# Arguments
- `A ∈ R(n,u)` Design matrix
- `y ∈ R(n)` RHS
- `Σ ∈ R(n,n)` Covariance matrix of the residuals (can be sparse or already factorized).
"""
function wls(A,y,Σ)
    (A'*(Σ\A))\A'*(Σ\y)
end

wls(A,y,Σ::Union{Matrix, SparseMatrixCSC}) = wls(A,y,factorize(Hermitian(Σ)))

"""
    tls(A,y)

Solves the total least-squares problem Ax=y using the SVD method
# Arguments
- `A` Design matrix
- `y` RHS
"""
function tls(A::AbstractArray,y::AbstractArray)
    AA  = [A y]
    s   = svd(AA)
    n   = size(A,2)
    V21 = s.V[1:n,n+1:end]
    V22 = s.V[n+1:end,n+1:end]
    x   = -V21/V22
end

"""
    tls!(Ay::AbstractArray, n::Integer)

Inplace version of `tls`. `Ay` is `[A y]` and `n` is the number of columns in `A`
"""
function tls!(Ay::AbstractArray, n::Integer)
    s   = svd!(Ay)
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
    Qaa,Qay,Qyy = spzeros(n*u,n*u), spzeros(n*u,n), Diagonal(spzeros(n,n))
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
    Ah,Eh = rpca(AA; nukeA=false, kwargs...)
    tls!(Ah,size(A,2))
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

end # module
