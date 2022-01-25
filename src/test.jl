include("SpectralAnalysis.jl")
include("Lorenz.jl")
using .SpectralAnalysis
using FFTW
using Statistics

function detrend(arr::AbstractArray)
    N = length(arr)
    t = collect(Float64, 1:N)
    A = zeros(Float64, N, 2)
    A[:, 1] .= 1.0
    A[:, 2] .= t
    arr_d = arr - A * ( (A' * A ) \ ( A' * arr ) )
end

T = 100


#x =  5.0 * sin.( 2π / T * t ) + 7.0 * cos.(6π / T * t) + 3.0 * sin.( 2π / T * t * 5 ) + randn(N)
println("Compute timeseries...")
s = runLorenz(10.0, 8/3, 28.0, [1.0 ; 1.0 ; 1.0], 0.01, 4000-1)
#x = detrend(s[1, :])
x = s[2, :]
x .+= randn(length(x)) * 3
println("done")

N = length(x)
t = collect(Float64, 1:N) .- 1.0



println("mean = ", mean(x))

x += randn(length(x)) * 5
x .-= mean(x)

spec_tukey, dω, λ_tukey = computeSpectrum(x; smoothing="Tukey")
spec_none, dω, λ_none = computeSpectrum(x)

spectrum_samples = []


for i=1:10
    println("The $i th spectrum")
    x_noise = x + randn(length(x)) * 20
    _spec_tukey, _, _ = computeSpectrum(x_noise; smoothing="Tukey")
    push!(spectrum_samples, _spec_tukey)
end

lowerbnd, upperbnd = computeCIRatio(N, λ_tukey; α=0.05)
 
fftw_transform = fft(x)
spec_fftw = abs.(fftw_transform[2:Int64(length(fftw_transform)/2)+1]).^2

println("∫ I(ω) dω = ", transpose(spec_none) * dω)
println("∫ I_tukey(ω) dω = ", transpose(spec_tukey) * dω)
println("Variance from data = ", var(x) * (N-1) / N)

white_noise = transpose(spec_none) * dω / sum(dω)

println("Plotting...")
using PyPlot
plt = PyPlot
fig, ax = plt.subplots(4, 1)

tt = collect(1:length(spec_none))

ax[1].plot(t, x)
ax[2].bar(tt, spec_none)
ax[2].plot(tt,spec_tukey)
ax[2].fill_between(tt, spec_tukey * upperbnd, spec_tukey * lowerbnd, color="red", alpha=0.2)

for i=1:length(spectrum_samples)
    ax[2].plot(tt, spectrum_samples[i], color="gray")
end

ax[2].plot([tt[1], tt[end]], [1.0, 1.0] * white_noise, "r--",)
ax[3].bar(tt, spec_fftw)

ax[4].plot(λ_none, label="λ_none")
ax[4].plot(λ_tukey, label="λ_tukey")

ax[4].legend()


#ax[2].set_xlim(0, 20)
#ax[3].set_xlim(0, 20)

plt.show()
