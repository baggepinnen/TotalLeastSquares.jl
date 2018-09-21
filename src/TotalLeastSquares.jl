module TotalLeastSquares
export tls, wtls
using FillArrays, Printf, LinearAlgebra

"""
    tls(A,b)

Solves the total least-squares problem Ax=b using the SVD method
# Arguments
- `A` Design matrix
- `b` RHS
"""
function tls(A,b)
    AA  = [A b]
    s   = svd(AA)
    m,n = length(b),size(A,2)
    V21 = s.V[1:n,n+1:end]
    V22 = s.V[n+1:end,n+1:end]
    x   = -V21/V22
end

a ⊗ b = kron(a,b)

"""
    x = wtls(A,b,Qaa,Qay,Qyy; iters = 10)

Solves min nᵀQ⁻¹n s.t. (A+E)x = b + v
where Q = [Qaa Qay; Qay' Qyy], n = [vec(E); v]

Uses algorithm 1 from
Weighted total least squares: necessary and sufficient conditions,
fixed and random parameters, Fang 2013
https://link.springer.com/article/10.1007/s00190-013-0643-2


# Arguments
- `A` Design matrix
- `b` RHS
- `Qaa` Covariance matrix of `e = vec(E)`
- `Qay` Covariance between A and y

# Keyword Arguments
- ` iters = 10` Maximum number of iterations
"""
function wtls(A,b,Qaa,Qay,Qyy; iters = 10)
    n,u = size(A)
    Iₙ,Iᵤ = Eye(n,n),Eye(u,u)
    x   = (A'*(Qyy\A))\A'*(Qyy\b) # Initialize with LS slution
    QₐₐQₐy = [Qaa Qay]
    QΠ = [Qaa Qay; [Qay' Qyy]]
    for i = 1:iters
        B = [(x' ⊗ Iₙ) -Iₙ]
        BQBᵀ = factorize(Symmetric(B*QΠ*B'))
        λ = BQBᵀ\(b-A*x)
        v = QₐₐQₐy*B'λ
        x = (A'*(BQBᵀ\A))\((Iᵤ ⊗ λ')*v + A'*(BQBᵀ\b))
    end
    x
end

end # module
