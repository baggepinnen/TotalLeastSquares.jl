# TotalLeastSquares.jl

Solve (weighted) total least-squares problems

These functions are exported:

- `x = tls(A,y)`
  Solves the standard TLS problem using the SVD method
- `x = wtls(A,y,Qaa,Qay,Qyy,iters=10)`
  Solves the weighted TLS problem using algorithm 1 from (Fang, 2013)
  The Q-matrices are the covariance matrices of the noise terms in `vec(A)` and `y` respectively.
- `Qaa,Qay,Qyy = rowcovariance(rowQ::AbstractVector{<:AbstractMatrix})`
  Takes row-wise covariance matrices `QAy[i]` and returns the full (sparse) covariance matrices. `rowQ = [cov([A[i,:] y[i]]) for i = 1:length(y)]`

## Example
```julia
using TotalLeastSquares, FillArrays
x   = randn(3)
A   = randn(50,3)
σa  = 1
σy  = 0.01
An  = A + σa*randn(size(A))
y   = A*x
yn  = y + σy*randn(size(y))
Qaa = σa^2*Eye(prod(size(A)))
Qay = 0Eye(prod(size(A)),length(y))
Qyy = σy^2*Eye(prod(size(y)))


x̂ = An\yn
@printf "Least squares error: %25.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)

x̂ = tls(An,yn)
@printf "Total Least squares error: %19.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)

x̂ = wtls(An,yn,Qaa,Qay,Qyy,iters=10)
@printf "Weighted Total Least squares error: %10.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)
println("----------------------------")
```
```julia
Least squares error:                 2.690e-01 -2.019e-01 -2.954e-01, Norm:  4.476e-01
Total Least squares error:           2.087e-01 -1.420e-01 -2.621e-01, Norm:  3.639e-01
Weighted Total Least squares error: -9.463e-02 -3.910e-02 -1.762e-01, Norm:  2.038e-01
----------------------------
```
