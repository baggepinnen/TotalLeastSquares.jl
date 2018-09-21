module TotalLeastSquares
export tls, wtls
using FillArrays, Printf, LinearAlgebra

"""
    tls(A,y)

Solves the total least-squares problem Ax=y using the SVD method
# Arguments
- `A` Design matrix
- `y` RHS
"""
function tls(A,y)
    AA  = [A y]
    s   = svd(AA)
    m,n = length(y),size(A,2)
    V21 = s.V[1:n,n+1:end]
    V22 = s.V[n+1:end,n+1:end]
    x   = -V21/V22
end

a ⊗ b = kron(a,b)

"""
    x = wtls(A,y,Qaa,Qay,Qyy; iters = 10)

Solves min nᵀQ⁻¹n s.t. (A+E)x = y + v
where Q = [Qaa Qay; Qay' Qyy], n = [vec(E); y]

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
- ` iters = 10` Maximum number of iterations
"""
function wtls(A,y,Qaa,Qay,Qyy; iters = 10)
    n,u = size(A)
    Iₙ,Iᵤ = Eye(n,n),Eye(u,u)
    x   = (A'*(Qyy\A))\A'*(Qyy\y) # Initialize with LS slution
    QₐₐQₐy = [Qaa Qay]
    QΠ = [Qaa Qay; [Qay' Qyy]]
    for i = 1:iters
        B = [(x' ⊗ Iₙ) -Iₙ]
        BQBᵀ = factorize(Symmetric(B*QΠ*B'))
        λ = BQBᵀ\(y-A*x)
        v = QₐₐQₐy*B'λ
        x = (A'*(BQBᵀ\A))\((Iᵤ ⊗ λ')*v + A'*(BQBᵀ\y))
    end
    x
end

end # module
