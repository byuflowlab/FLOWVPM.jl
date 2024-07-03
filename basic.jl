using FLOWVPM
using Random
using BSON

Random.seed!(123)

n = 2^12
s_cpu = FLOWVPM.ParticleField(n)
t_cpu = FLOWVPM.ParticleField(n)

s_gpu = FLOWVPM.ParticleField(n; useGPU=true)
t_gpu = FLOWVPM.ParticleField(n; useGPU=true)

mat = zeros(43, n)
mat[1:7, :] .= rand(7, n)
for i in 1:n
    FLOWVPM.add_particle(s_cpu, mat[:, i])
    FLOWVPM.add_particle(t_cpu, mat[:, i])

    FLOWVPM.add_particle(s_gpu, mat[:, i])
    FLOWVPM.add_particle(t_gpu, mat[:, i])
end

d_switch = FLOWVPM.FastMultipole.DerivativesSwitch()

@time FLOWVPM.fmm.direct!(t_cpu, 1:n, d_switch, s_cpu, 1:n)
@time FLOWVPM.fmm.direct!(t_gpu, 1:n, d_switch, s_gpu, 1:n)

println("Write out CPU file")
bson("t_cpu.bson", tmat=t_cpu.particles)

println("Write out GPU file")
bson("t_gpu.bson", tmat=t_gpu.particles)
