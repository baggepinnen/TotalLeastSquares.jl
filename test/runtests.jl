using Random, Statistics, LinearAlgebra, Test, FillArrays, Printf, TotalLeastSquares
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


@testset "TotalLeastSquares" begin
    x̂ = An\yn
    @printf "Least squares error: %25.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)
    @test norm(x-x̂) < 1

    x̂ = wls(An,yn,Qyy)
    @printf "Weigthed Least squares error: %16.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)
    @test norm(x-x̂) < 1

    x̂ = tls(An,yn)
    @printf "Total Least squares error: %19.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)
    @test norm(x-x̂) < 1

    x̂ = wtls(An,yn,Qaa,Qay,Qyy,iters=10)
    @printf "Weighted Total Least squares error: %10.3e %10.3e %10.3e, Norm: %10.3e\n" (x-x̂)... norm(x-x̂)
    println("----------------------------")
    @test norm(x-x̂) < 1

    @test tls(An,yn) ≈ tls!([An yn], size(An,2))

    rowC = rowcovariance([[σa^2*Eye(3) zeros(3); zeros(1,3) σy^2] for _ in 1:50])
    @test rowC[1] ≈ Qaa
    @test rowC[2] ≈ Qay
    @test rowC[3] ≈ Qyy
end
