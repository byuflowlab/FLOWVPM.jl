# Quick Start
This tutorial runs through setting up a basic particle field and simulating the field.

## Create a ParticleField
First, we create a particle field with a maximum of 10 particles.

```@example basic
using FLOWVPM

max_particles = 10
pfield = ParticleField(max_particles);
```

This creates an empty particle field that holds at most 10 particles.

## Add Particles
Now we can add particles to the field. Attempting to add more than `max_particles` will result in an error.

```@example basic
using Random

for i in 1:max_particles
    add_particle(pfield, rand(3), rand(3), rand(1)[1])
end
println("Number of particles: ", pfield.np)
```

## Propogate the ParticleField
The particles in the ParticleField can be propogated single steps using `FLOWVPM.nextstep(pfield, dt)`.

```@example basic
dt = 0.01
FLOWVPM.nextstep(pfield, dt)
```

For convinence `run_vpm!(pfield, dt, nsteps)` is also provided.

```@example basic
dt = 0.01
nsteps = 100
run_vpm!(pfield, dt, nsteps)
```

## Remove Particles
If we want to remove particles from the field we can use the `remove_particle(pfield, i)` function where `i` is the index of the particle we want removed.

```@example basic
for i in pfield.np:-1:1
    remove_particle(pfield, i)
end
println("Number of particles after removal ", pfield.np)
```