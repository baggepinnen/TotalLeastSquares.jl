[![CI](https://github.com/baggepinnen/TotalLeastSquares.jl/workflows/CI/badge.svg)](https://github.com/baggepinnen/TotalLeastSquares.jl/actions)
[![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/T/TotalLeastSquares.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html)
[![codecov](https://codecov.io/gh/baggepinnen/TotalLeastSquares.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/TotalLeastSquares.jl)


# TotalLeastSquares.jl

Solve (weighted or robust) total least-squares problems, impute missing data and robustly filter time series.

These functions are exported:

#### Estimation
- `x = tls(A,y)`
  Solves the standard TLS problem using the SVD method. An inplace version `tls!(Ay, n)` also exists, for this you need to supply `Ay = [A y]` and the width of `A`, `n = size(A,2)`.
- `x = wtls(A,y,Qaa,Qay,Qyy,iters=10)`
  Solves the weighted TLS problem using algorithm 1 from (Fang, 2013)
  The Q-matrices are the covariance matrices of the noise terms in `vec(A)` and `y` respectively.
- `Qaa,Qay,Qyy = rowcovariance(rowQ::AbstractVector{<:AbstractMatrix})`
  Takes row-wise covariance matrices `QAy[i]` and returns the full (sparse) covariance matrices. `rowQ = [cov([A[i,:] y[i]]) for i = 1:length(y)]`
- `x = wls(A,y,Qyy)` Solves the weighted standard LS problem. `Qyy` is the covariance matrix of the residuals with side length equal to the length of `y`.
- `x = rtls(A,y)` Solves a robust TLS problem. Both `A` and `y` are assumed to be corrupted with high magnitude, but sparse, noise. See analysis below.
- `x = irls(A,y; tolx=0.001, tol=1.0e-6, verbose=false, iters=100)` minimizeₓ ||Ax-y||₁ using iteratively reweighted least squares.
- `x = sls(A,y; r = 1, iters = 100, verbose = false, tol = 1.0e-8)` Simplex least-squares: minimizeₓ ||Ax-y||₂ s.t. sum(x) = r




#### Matrix recovery
- `Â, Ê, s, sv = rpca(D; kwargs...)` robust matrix recovery using robust PCA. Solves `minimize_{A,E} ||A||_* + λ||E||₁ s.t. D = A+E`. Optionally force `A` or `E` to be non-negative.
- `Q = rpca_ga(X, r; kwargs...)` robust PCA using Grassmann averages. Returns the principal components up to rank `r`.
#### Time-series filtering
- `yf = lowrankfilter(y, n; kwargs...)` Filter time series `y` by forming a lag-embedding H (a Hankel matrix) and using [`rpca`](@ref) to recover a low-rank matrix from which the a filtered signal `yf` can be extracted. Robustly filters large sparse outliers.

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
@printf "Weighted Least squares error: %16.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)

x̂ = tls(An,yn)
@printf "Total Least squares error: %19.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)

x̂ = wtls(An,yn,Qaa,Qay,Qyy,iters=10)
@printf "Weighted Total Least squares error: %10.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)
println("----------------------------")
```
```julia
Least squares error:                 3.753e-01  2.530e-01 -3.637e-01, Norm:  5.806e-01
Weighted Least squares error:        3.753e-01  2.530e-01 -3.637e-01, Norm:  5.806e-01
Total Least squares error:           2.897e-01  1.062e-01 -2.976e-01, Norm:  4.287e-01
Weighted Total Least squares error:  1.213e-01 -1.933e-01 -9.527e-02, Norm:  2.473e-01
----------------------------
```

## Robust TLS analysis
The code for this analysis is [here](https://github.com/baggepinnen/TotalLeastSquares.jl/blob/master/total_vs_robust_demo.jl).

We generate random data on the form `Ax=y` where both `A` and `y` are corrupted with sparse noise, the entries in `A` are Gaussian random variables with unit variance and `size(A) = (500,5)`. The plots below show the norm of the error in the estimated `x` as functions of the noise variance and the noise sparsity.

![window](figs/e_vs_n.svg)
![window](figs/e_vs_s_5.svg)
![window](figs/e_vs_s_50.svg)

The results indicate that the robust method is to be preferred when the noise is large but sparse.

## Missing data imputation
The robust methods handle missing data the same way as they handle outliers. You may indicate that an entry is missing simply by setting it to a very large value, e.g.,
```julia
N = 500
y = sin.(0.1 .* (1:N)) # Sinus
miss = rand(N) .< 0.1  # 10% missing values
yn = y .+ miss .* 1e2 .+ 0.1*randn(N)   # Set missing values to very large number and add noise
yf = lowrankfilter(yn,40)    # Filter
mean(abs2,y-yf)/mean(abs2,y) # Normalized error
# 0.001500 # Less than 1 percent error in the recovery of y
```
To impute missing data in a matrix, we make use of `rpca`:
```julia
H = hankel(sin.(0.1 .* (1:N)), 5)  # A low-rank matrix
miss = rand(size(H)...) .< 0.1     # 10% missing values
Hn = H .+ 0.1randn(size(H)) .+ miss .* 1e2    # Set missing values to very large number
Ĥ, E = rpca(Hn)
mean(abs2,H-Ĥ)/mean(abs2,H) # Normalized error
# 0.06 # Six percent error in the recovery of H
```
The matrix `E` contains the estimated outliers
```julia
vec(E)'vec(miss)/(norm(E)*norm(miss)) # These should correlate if all missing values were identified
# 1.00
```

## Speeding up robust factorization
The function `rpca` internally performs several SVDs, which make up the bulk of the computation time. In order to speed this up, you may provide a custom `svd` function. An example using a randomized method from [RandomizedLinAlg.jl](https://haampie.github.io/RandomizedLinAlg.jl/latest/index.html#RandomizedLinAlg.rsvd):
```julia
lowrankfilter(xn, L, svd = rsvd, opnorm=x->rnorm(x,10)) # The same keywords are accepted by rpca
```
here, we provide both a randomized svd function as well as one for calculating the operator norm, which also takes a long time.


# Notes
This package was developed for the thesis  
[Bagge Carlson, F.](https://www.control.lth.se/staff/fredrik-bagge-carlson/), ["Machine Learning and System Identification for Estimation in Physical Systems"](https://lup.lub.lu.se/search/publication/ffb8dc85-ce12-4f75-8f2b-0881e492f6c0) (PhD Thesis 2018).
```bibtex
@thesis{bagge2018,
  title        = {Machine Learning and System Identification for Estimation in Physical Systems},
  author       = {Bagge Carlson, Fredrik},
  keyword      = {Machine Learning,System Identification,Robotics,Spectral estimation,Calibration,State estimation},
  month        = {12},
  type         = {PhD Thesis},
  number       = {TFRT-1122},
  institution  = {Dept. Automatic Control, Lund University, Sweden},
  year         = {2018},
  url          = {https://lup.lub.lu.se/search/publication/ffb8dc85-ce12-4f75-8f2b-0881e492f6c0},
}
```
