function runLorenz(σ :: Float64, β :: Float64, ρ :: Float64, s0 :: AbstractArray{Float64}, dt :: Float64, steps :: Int64)
    s = zeros(Float64, 3, steps+1)
    s[:, 1] = s0

    for t=2:steps+1
        x, y, z = s[:, t-1]
        s[:, t] = s[:, t-1] + dt * [ (σ * (y - x)) ; (x * (ρ - z) - y) ; (x*y - β*z)]
    end

    return s
end
