using Test
import FLOWVPM

# Data structure test
println("Data structure test: Add/remove particle...")
@test begin
    vpm = FLOWVPM
    pfield = vpm.ParticleField(10)

    # Add particles
    for i in 1:4
        vpm.add_particle(pfield, (i*10^0, i*10^1, i*10^2), zeros(3), i)
    end

    # Remove second particle
    vpm.remove_particle(pfield, 2)

    # Add particles
    for i in 5:11
        vpm.add_particle(pfield, (i*10^0, i*10^1, i*10^2), zeros(3), i)
    end

    vpm.get_np(pfield)==10 && vpm.get_X(pfield, 2)==[4*10^0, 4*10^1, 4*10^2]
end
