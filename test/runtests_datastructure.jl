using Test
import FLOWVPM


# Data structure test
println("\nData structure test: Julia->C++ communication...")
@test begin
    verbose = true
    vpm = FLOWVPM
    pfield = vpm.ParticleField(10)

    # Add particles
    for i in 1:4
        vpm.add_particle(pfield, (i*10^0, i*10^1, i*10^2), zeros(3), i)
    end

    # Modify particle in Julia
    pi = 3
    xi = 2
    P = vpm.get_particle(pfield, pi)
    P.X[xi] = -1

    if verbose; println("\tX in particle:\t$(P.X)"); end;

    # Check that the body in C++ was also modified
    body = vpm.fmm.getBody(pfield.bodies, pi-1)
    Xi = vpm.fmm.get_Xi(body, xi-1)

    if verbose; println("\tX in body:\t$(vpm.fmm.get_Xref(body))"); end;

    return Xi == -1
end


# Data structure test
println("\nData structure test: C++->Julia communication...")
@test begin
    verbose = true
    vpm = FLOWVPM
    pfield = vpm.ParticleField(10)

    # Add particles
    for i in 1:4
        vpm.add_particle(pfield, (i*10^0, i*10^1, i*10^2), zeros(3), i)
    end

    # Modify body in C++
    pi = 3
    xi = 2
    body = vpm.fmm.getBody(pfield.bodies, pi-1)
    vpm.fmm.set_Xi(body, xi-1, -9.0)

    if verbose; println("\tX in body:\t$(vpm.fmm.get_Xref(body))"); end;

    # Check that the particle in Julia was also modified
    P = vpm.get_particle(pfield, pi)
    Xi = P.X[xi]

    if verbose; println("\tX in particle:\t$(P.X)"); end;

    return Xi == -9
end

# Data structure test
println("\nData structure test: Add/remove particle...")
@test begin
    verbose = true
    vpm = FLOWVPM
    pfield = vpm.ParticleField(10)

    # Add particles
    for i in 1:4
        vpm.add_particle(pfield, (i*10^0, i*10^1, i*10^2), zeros(3), i)
    end

    if verbose
        println("\tInitial particle positions")
        for (i, P) in enumerate(vpm.iterator(pfield))
            println("\t\tParticle #$i:\t$(P.X)")
        end
    end

    # Remove second particle
    vpm.remove_particle(pfield, 2)

    if verbose
        println("\tParticle positions after removal")
        for (i, P) in enumerate(vpm.iterator(pfield))
            println("\t\tParticle #$i:\t$(P.X)")
        end
    end

    # Add particles
    for i in 5:11
        vpm.add_particle(pfield, (i*10^0, i*10^1, i*10^2), zeros(3), i)
    end

    if verbose
        println("\tParticle positions after addition")
        for (i, P) in enumerate(vpm.iterator(pfield))
            println("\t\tParticle #$i:\t$(P.X)")
        end
    end

    vpm.get_np(pfield)==10 && vpm.get_particle(pfield, 2).X==[4*10^0, 4*10^1, 4*10^2]
end
