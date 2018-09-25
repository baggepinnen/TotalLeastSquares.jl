[![Build Status](https://travis-ci.org/baggepinnen/TotalLeastSquares.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/TotalLeastSquares.jl)
[![codecov](https://codecov.io/gh/baggepinnen/TotalLeastSquares.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/TotalLeastSquares.jl)


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
- `x = wls(A,y,Qyy)` Solves the weighted standard LS problem. `Qyy` is the covariance matrix of the residuals with side length equal to the length of `y`.

## Example
```julia
using TotalLeastSquares, FillArrays, Random, Printf
Random.seed!(0)
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

x̂ = wls(An,yn,Qyy)
@printf "Weigthed Least squares error: %16.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)

x̂ = tls(An,yn)
@printf "Total Least squares error: %19.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)

x̂ = wtls(An,yn,Qaa,Qay,Qyy,iters=10)
@printf "Weighted Total Least squares error: %10.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)
println("----------------------------")
```
```julia
Least squares error:                 3.753e-01  2.530e-01 -3.637e-01, Norm:  5.806e-01
Weigthed Least squares error:        3.753e-01  2.530e-01 -3.637e-01, Norm:  5.806e-01
Total Least squares error:           2.897e-01  1.062e-01 -2.976e-01, Norm:  4.287e-01
Weighted Total Least squares error:  1.213e-01 -1.933e-01 -9.527e-02, Norm:  2.473e-01
----------------------------
```
