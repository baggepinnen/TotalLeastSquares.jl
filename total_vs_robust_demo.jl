using TotalLeastSquares, Plots, ThreadTools


N = 500
σvec = exp10.(LinRange(-3, 3, 40))
results1 = tmap(σvec) do σ
      inner = map(1:N) do _
            x   = randn(5)
            A   = randn(500,5)
            An  = A + σ*randn(size(A)) .* (rand(size(A)...) .< 0.01)
            y   = A*x
            yn  = y + σ*randn(size(y)) .* (rand(size(y)...) .< 0.01)

            x̂t = tls(An,yn)
            x̂r = rtls(An,yn)

            norm(x-x̂t), norm(x-x̂r)
      end
      mean(getindex.(inner, 1)), mean(getindex.(inner, 2))
end

res_tls1, res_rtls1 = getindex.(results1, 1), getindex.(results1, 2)

scatter(σvec, [res_tls1 res_rtls1], xscale=:log10, yscale=:log10, lab=["TLS" "RTLS"], xlabel="Noise std", ylabel="Parameter error norm", legend=:topleft, title="Noise Sparsity = 0.01")

##
σ = 5
svec = exp10.(LinRange(-3, 0, 40))
results2 = tmap(svec) do s
      inner = map(1:N) do _
            x   = randn(5)
            A   = randn(500,5)
            An  = A + σ*randn(size(A)) .* (rand(size(A)...) .< s)
            y   = A*x
            yn  = y + σ*randn(size(y)) .* (rand(size(y)...) .< s)

            x̂t = tls(An,yn)
            x̂r = rtls(An,yn)

            norm(x-x̂t), norm(x-x̂r)
      end
      mean(getindex.(inner, 1)), mean(getindex.(inner, 2))
end

res_tls2, res_rtls2 = getindex.(results2, 1), getindex.(results2, 2)

scatter((svec), [res_tls2 res_rtls2], xscale=:log10, yscale=:log10, lab=["TLS" "RTLS"], xlabel="Noise sparsity", ylabel="Parameter error norm", legend=:topleft, title="Noise std = $σ")
