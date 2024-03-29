using Random, Statistics, LinearAlgebra, Test, FillArrays, Printf, TotalLeastSquares, StatsBase
using RandomizedLinAlg
Random.seed!(0)


@testset "TotalLeastSquares" begin

    @testset "TLS" begin
        @info "Testing TLS"
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


    @testset "Robust PCA" begin
        @info "Testing Robust PCA"
        D = [0.462911    0.365901  0.00204357    0.692873    0.935861;
        0.0446199    0.108606   0.0664309   0.0736707    0.264429;
        0.320581    0.287788    0.073133    0.188872    0.526404;
        0.356266    0.197536 0.000718338    0.513795    0.370094;
        0.677814    0.011651    0.818047   0.0457694    0.471477]

        A = [0.462911   0.365901  0.00204356   0.345428   0.623104;
        0.0446199  0.108606  0.0429271    0.0736707  0.183814;
        0.320581   0.203777  0.073133     0.188872   0.472217;
        0.30725    0.197536  0.000717701  0.201626   0.370094;
        0.234245   0.011651  0.103622     0.0457694  0.279032]

        E = [0.0        0.0        0.0        0.347445  0.312757 ;
        0.0        0.0        0.0235038  0.0       0.0806151;
        0.0        0.0840109  0.0        0.0       0.0541868;
        0.0490157  0.0        6.5061e-7  0.312169  0.0      ;
        0.443569   0.0        0.714425   0.0       0.192445]

        Â, Ê,_,_ = rpca(D, nonnegE=true, nonnegA=true, verbose=false)

        @test Â ≈ A atol=1.0e-6
        @test Ê ≈ E atol=1.0e-6
        @test norm(D - (Â + Ê))/norm(D) < sqrt(eps())


        Â, Ê,_,_ = rpca(D, nonnegE=false, nonnegA=false, verbose=false)
        @test norm(D - (Â + Ê))/norm(D) < sqrt(eps())


        @testset "Missing values" begin
            @info "Testing Missing values"

            res = map(1:20) do _
                N = 500
                y = sin.(0.1 .* (1:N)) .+ 0.1*randn(N)
                miss = rand(N) .< 0.1
                yn = y .+ miss .* 1e2
                yf = lowrankfilter(yn,40)
                mean(abs2,y-yf)/mean(abs2,y)
            end
            @test mean(res) < 0.025

        end

        @testset "Complex" begin
            @info "Testing Complex"
            u = randn(ComplexF64, 100)
            v = randn(ComplexF64, 20)
            E = randn(ComplexF64,100,20) .* 10 .* (rand.() .< 0.01)
            A = u*v'
            D = A .+ E
            Ah, Eh, s, sv = rpca(D)
            @test sum(abs2, Eh-E)/sum(abs2, E) < 1e-5
            @test sum(abs2, Ah-A)/sum(abs2, A) < 1e-5


        end

    end

@testset "rtls" begin
    @info "Testing rtls"
    passes = map(1:1000) do _
        x   = randn(3)
        A   = randn(50,3)
        σ   = 5
        An  = A + σ*randn(size(A)) .* (rand(size(A)...) .< 0.1)
        y   = A*x
        yn  = y + σ*randn(size(y)) .* (rand(size(y)...) .< 0.1)

        AA  = [An yn]
        Ah,Eh,_,_ = rpca(AA, verbose=false)

        sum(abs2, Ah - [A y])/sum(abs2,[A y]) < sum(abs2, AA - [A y])/sum(abs2,[A y])
    end
    @show mean(passes)
    @test mean(passes) > 0.9

    passes = map(1:1000) do _
        x   = randn(3)
        A   = randn(50,3)
        σ   = 50
        An  = A + σ*randn(size(A)) .* (rand(size(A)...) .< 0.1)
        y   = A*x
        yn  = y + σ*randn(size(y)) .* (rand(size(y)...) .< 0.1)

        x̂t = tls(An,yn)
        x̂r = rtls(An,yn)

        norm(x-x̂r) < norm(x-x̂t)
    end
    @show mean(passes)
    @test mean(passes) > 0.9

    # @testset "Elastic Net rtls" begin
    #     @info "Testing Elastic Net rtls"
    #     passes = map(1:500) do _
    #         x     = randn(10)
    #         A     = randn(100,3) * randn(3,10)
    #         σs    = 50
    #         σd    = 0.5
    #         S     = σs*randn(size(A)) .* (rand(size(A)...) .< 0.1)
    #         D     = σd*randn(size(A))
    #         N     = S + D
    #         An    = A + N
    #
    #         Ah,Eh,_,_ = rpca(An, verbose=false)
    #
    #         λ = 1/sqrt(100)
    #         μ = 0.001λ
    #         Ah2,Eh2,_,_ = rpca(An, verbose=false, proxE = ElasticNet(λ, μ))
    #
    #         sum(abs2, Ah - A)/sum(abs2,A) < sum(abs2, An - A)/sum(abs2,A),
    #         sum(abs2, Eh2 - N)/sum(abs2,N) < sum(abs2, Eh - N)/sum(abs2,N)
    #     end
    #     @show mean(getindex.(passes, 1))
    #     @test mean(getindex.(passes, 1)) > 0.9
    #     @show mean(getindex.(passes, 2))
    #     @test mean(getindex.(passes, 2)) > 0.9
    #
    #
    # end

end


# @testset "rtls ga" begin
#     @info "Testing rtls ga"
#
#     passes = map(1:1000) do _
#         x   = randn(5)
#         A   = randn(500,5)
#         σ   = 50
#         An  = A + σ*randn(size(A)) .* (rand(size(A)...) .< 0.01)
#         y   = A*x
#         yn  = y + σ*randn(size(y)) .* (rand(size(y)...) .< 0.01)
#
#         x̂t = tls(An,yn)
#         x̂r = rtls_ga(An,yn, μ=entrywise_median)
#
#         norm(x-x̂r) < norm(x-x̂t)
#     end
#     @show mean(passes)
#     @test mean(passes) > 0.7
#
# end

@testset "soft hankel" begin
    @info "Testing soft hankel"

    @test hankel(1:20,2) == [1:19 2:20]
    @test hankel(1:20,3,2) == [1:2:17 2:2:18 3:2:19]

    A = hankel(1:8,4)
    @test ishankel(A)
    An = A + 0.1randn(size(A))
    @test !ishankel(An)
    Anc = copy(An)
    TotalLeastSquares.soft_hankel!(An, 0.1)
    @test sum(abs2,An-A) < sum(abs2,Anc-A)

    An = -A + 0.1randn(size(A))
    Anc = copy(An)
    TotalLeastSquares.soft_hankel!(An, 0.1)
    @test sum(abs2,An+A) < sum(abs2,Anc+A)

    passes = map(1:100) do _
        y = randn(20)
        yn = y .+ 0.1 .* randn.()
        A = hankel(y,3)
        An = hankel(yn,3)
        Anc = copy(An)
        A1,E1,_,_ = rpca(An, verbose=false, nukeA=false)
        A2,E2,_,_ = rpca(An, verbose=false, nukeA=false, hankel=true)
        sum(abs2,A2-A) < sum(abs2,A1-A)
    end
    @show mean(passes)
    @test mean(passes) > 0.7

    @info "Small random Hankel"
    passes = map(1:500) do _
        y = randn(100)
        A = hankel(y,5)
        @test ishankel(A)
        A1,E1,_,_ = rpca(A, verbose=false, nukeA=false)
        A2,E2,_,_ = rpca(A, verbose=false, nukeA=false, hankel=true)
        @test ishankel(A2)
        @test ishankel(E2)
        mean(abs2,A2-A) < mean(abs2,A1-A)
    end
    @show mean(passes)
    @test mean(passes) > 0.8

    @info "Big random Hankel"
    passes = map(1:10) do _
        y = randn(1000)
        A = hankel(y,50)
        @test ishankel(A)
        A1,E1,_,_ = rpca(A, verbose=false, nukeA=false)
        A2,E2,_,_ = rpca(A, verbose=false, nukeA=false, hankel=true)
        @test ishankel(A2)
        @test ishankel(E2)
        mean(abs2,A2-A) < mean(abs2,A1-A)
    end
    @show mean(passes)
    @test mean(passes) >= 0.8


end


@testset "lowrankfilter" begin
    @info "Testing lowrankfilter"
    qn(x) = x ./ quantile(abs.(x), 0.9)
    T = 1000
    t = 1:T
    y = sin.(0.1 .* t) |> qn

    H = hankel(y, 2)
    @test ishankel(H)
    @test unhankel(H) == y

    H = hankel(y,2,2)
    yh = unhankel(H,2,T)
    @test yh == y

    H = hankel(y,5,2)
    yh = unhankel(H,2,T)
    @test yh[1:end-1] ≈ y[1:end-1]

    y2 = randn(T)
    H = hankel([y y2],5,2)
    yh = unhankel(H,2,T,2)
    @test yh[1:end-1,:] ≈ [y y2][1:end-1,:]

    n = 20randn(T) .* (rand(T) .< 0.01) + 0.1randn(T)
    yf = lowrankfilter(y+n) |> qn
    @test mean(abs2, y-yf)/mean(abs2, n) < 0.001
    @show mean(abs2, y-yf)/mean(abs2, n)

    # Randomized lin alg
    yf = lowrankfilter(y+n, opnorm=x->rnorm(x,10)) |> qn
    @test mean(abs2, y-yf)/mean(abs2, n) < 0.001
    @show mean(abs2, y-yf)/mean(abs2, n)

    yf = lowrankfilter(y+n, svd=rsvd_fnkz) |> qn
    @test mean(abs2, y-yf)/mean(abs2, n) < 0.05
    @show mean(abs2, y-yf)/mean(abs2, n)

    yf = lowrankfilter(y+n, opnorm=x->rnorm(x,10), svd=rsvd_fnkz) |> qn
    @test mean(abs2, y-yf)/mean(abs2, n) < 0.05
    @show mean(abs2, y-yf)/mean(abs2, n)

    yf = lowrankfilter(y+n, opnorm=x->rnorm(x,10), svd=rsvd_fnkz, maxrank=5) |> qn
    @test mean(abs2, y-yf)/mean(abs2, n) < 0.05
    @show mean(abs2, y-yf)/mean(abs2, n)


    # Only SSA
    n = randn(T)
    yf = lowrankfilter(y+n, sv=2) |> qn
    @test mean(abs2, y-yf)/mean(abs2, n) < 0.05
    @show mean(abs2, y-yf)/mean(abs2, n)


    @testset "Multi channel" begin
        @info "Testing Multi channel"

        T = 1000
        t = 1:T
        y1 = sin.(0.1 .* t)
        y2 = sin.(0.3 .* t)
        y = [y1 y2+0.5y1]
        yn = y .+ 0.1 .* randn.() .+ (rand.() .< 0.01) .* 5 .* randn.()
        yf = lowrankfilter(yn, 20)
        @show mean(abs2,y-yf)/mean(abs2,y-yn)
        @test mean(abs2,y-yf)/mean(abs2,y-yn) < 0.05

        # There shoud be benefit to joint filtering since the two signals are correlated
        yf1 = lowrankfilter(yn[:,1], 20)
        @test mean(abs2,y1-yf1)/mean(abs2,y1-yf[:,1]) > 1.02

        # Same freq, different phase
        y1 = sin.(0.1 .* t)
        y2 = sin.(0.1 .* t .+ 0.01)
        y = [y1 y2]
        yn = y .+ 0.1 .* randn.() .+ (rand.() .< 0.01) .* 5 .* randn.()
        yf = lowrankfilter(yn, 20)
        @show mean(abs2,y-yf)/mean(abs2,y-yn)
        @test mean(abs2,y-yf)/mean(abs2,y-yn) < 0.02

        yf1 = lowrankfilter(yn[:,1], 20)
        @show mean(abs2,y1-yf1)/mean(abs2,y1-yf[:,1])
        @test mean(abs2,y1-yf1)/mean(abs2,y1-yf[:,1]) > 1.2


    end

end

@testset "rpca_ga" begin
    @info "Testing rpca_ga"

    Random.seed!(1)
    for r = 1:10, ϵ = exp10.(LinRange(-8, 0, 20))
        s = svd(randn(10,40))
        u,v = s.U[:,1:r], s.V[:,1:r]
        A = u*Diagonal(10*(1:r))*v' .+ ϵ .* randn.()
        Q = rpca_ga(A, r, verbose=false)
        @test_skip rank([Q u], atol=ϵ) == r # Q and u should span the same space, but they may not share any other properties (deactivated because it fails about 1/10)
        @test norm(Q'Q - I) < sqrt(eps())
    end

    # Below is the reverse dimensions of above
    for r = 1:10, ϵ = exp10.(LinRange(-8, 0, 20))
        s = svd(randn(40,10))
        u,v = s.U[:,1:r], s.V[:,1:r]
        A = u*Diagonal(10*(1:r))*v' .+ ϵ .* randn.()
        Q = rpca_ga(A, r, verbose=false)
        @test_skip rank([Q u], atol=ϵ) == r # Q and u should span the same space, but they may not share any other properties (deactivated because it fails about 1/10)
        @test norm(Q'Q - I) < sqrt(eps())
    end

    @testset "entrywise trimmed mean" begin
        @info "Testing entrywise trimmed mean"

        U = randn(10,10)
        s = zeros(10)
        w = ones(10)
        m1 = TotalLeastSquares.μ!(s,w,U)
        @test m1 ≈ mean(U, dims=2)

        m2 = TotalLeastSquares.entrywise_trimmed_mean(s,w,U,0)
        @test m2 ≈ mean(U, dims=2)

        w = randn(10)
        m1 = TotalLeastSquares.μ!(s,w,U)
        @test m1 ≈ sum(U .* w', dims=2)./ sum(w)

        m2 = TotalLeastSquares.entrywise_trimmed_mean(s,w,U,0)
        @test m2 ≈ sum(U .* w', dims=2)./ sum(w)

        w = ones(10)
        m2 = TotalLeastSquares.entrywise_trimmed_mean(s,w,U,0.1)
        for i in eachindex(m2)
            @test m2[i] ≈ mean(StatsBase.trim(U[i,:], prop=0.1))
        end

        r = 3; ϵ = 1e-8
        Random.seed!(1)
        passes = map(Iterators.product(1:4, exp10.(LinRange(-8, -1, 3)))) do (r,ϵ)
            # @show r, ϵ
            s = svd(randn(10,1000))
            u,v = s.U[:,1:r], s.V[:,1:r]
            # A = u*Diagonal(10*(1:r))*v' .+ ϵ .* randn.()
            A = u*Diagonal(s.S[1:r])*v' .+ ϵ .* randn.()
            A .+= 1000*randn.() .* (rand.() .< 0.01)

            Qtm = rpca_ga(A, r, verbose=false, μ = TotalLeastSquares.entrywise_trimmed_mean, iters=120)
            Qm = rpca_ga(A, r, verbose=false, iters=120)
            sum(svdvals([Qtm u])[r+1:2r]) < sum(svdvals([Qm u])[r+1:2r])
            # @test norm(Q'Q - I) < r*sqrt(eps()) # This test should pass but doesn't
        end
        @show mean(passes)
        @test mean(passes) > 0.8

        passes = map(Iterators.product(1:4, exp10.(LinRange(-8, -1, 3)))) do (r,ϵ)
            # @show r, ϵ
            s = svd(randn(10,1000))
            u,v = s.U[:,1:r], s.V[:,1:r]
            # A = u*Diagonal(10*(1:r))*v' .+ ϵ .* randn.()
            A = u*Diagonal(s.S[1:r])*v' .+ ϵ .* randn.()
            A .+= 1000*randn.() .* (rand.() .< 0.01)

            Qmed = rpca_ga(A, r, verbose=false, μ = TotalLeastSquares.entrywise_median, iters=120)
            Qm = rpca_ga(A, r, verbose=false, iters=120)
            sum(svdvals([Qmed u])[r+1:2r]) < sum(svdvals([Qm u])[r+1:2r])
            # @test norm(Q'Q - I) < r*sqrt(eps())  # This test should pass but doesn't
        end
        @show mean(passes)
        @test mean(passes) > 0.9
    end
end

@testset "irls" begin
    @info "Testing irls"

    N = 10000
    n = 20
    A = randn(N, n)
    x = randn(n)
    y = A*x
    e = 100*(rand(N) .< 0.01) + 0.1randn(N)
    @time xh = irls(A,y+e, verbose=true,iters=100)
    @test norm(x-xh)/norm(x) < 0.01
    xhls = A\(y+e)
    @test norm(x-xh) < norm(x-xhls)
end

@testset "sls" begin
    @info "Testing sls"

    for i = 1:100
        x = TotalLeastSquares.proj_simplex!(randn(i))
        @test sum(x) ≈ 1
        @test all(x .>= 0)
    end


    s1(x) = x ./= sum(x)
    ##
    n = 10
    N = 2
    results = map(1:100) do _
        a = [(rand(n)) for _ in 1:N]
        A = reduce(hcat,a)
        x = s1(rand(N))
        y = A*x
        yn = y .+ 0.01.*randn.()
        xh1 = TotalLeastSquares.proj_simplex!(A\yn)

        xh2 = sls(A,yn,verbose=false)
        @test sum(xh2) ≈ 1

        norm(x-xh2) < norm(x-xh1) || norm(yn-A*xh2) < norm(yn-A*xh1)
    end
    @test mean(results) >= 0.98
end

@testset "flts" begin
    @info "Testing flts"
    # Example from paper 
    x = 10randn(1000)
    a, b = 1, 2
    y = a*x .+ b 
    y[801:end] = 5randn(200) 
    x[801:end] = 5randn(200) .+ 50 
    xb = [x ones(1000)]
    # default
    res = flts(xb,y, verbose = true)
    @test a ≈ res[1]
    @test b ≈ res[2]
    # define % of outliers
    res1 = flts(xb,y, N = 10, outliers = 0.20, verbose = true)
    @test a ≈ res1[1]
    @test b ≈ res1[2]
    # define length of set H and check the subset
    (H, res2, Q) = flts(xb,y, N = 10, h = 800, return_set = true, verbose = true)
    @test a ≈ res2[1]
    @test b ≈ res2[2]
    @test all(H .< 801)
    @test Q + 1 ≈ 1
end
end
