# Guided Examples

## Vortex Ring
```@example ring
using FLOWVPM
using LinearAlgebra

rm("ring", recursive=true, force=true)
max_particles = 20
pfield = ParticleField(max_particles);

# Ring parameters
r = 1.0                             # vortex ring radius
n_particles = max_particles         # number of particles in the ring
circulation = 10.0                   # circulation strength

sigma = 2*pi*r / n_particles
d_theta = 2*pi / n_particles
omega = circulation / (pi*r^2)      # Average vorticity
dr = pi*r^2 / n_particles

for i in 1:n_particles
    theta = d_theta * (i-1)
    X = [0.0, r*cos(theta), r*sin(theta)]
    Gamma_hat = cross(X, [-1.0, 0.0, 0.0])
    Gamma_hat = Gamma_hat / norm(Gamma_hat)
    Gamma = omega * Gamma_hat * dr

    add_particle(pfield, X, Gamma, sigma)
end

run_vpm!(pfield, 0.1, 200; save_path="ring")
```

```@setup ring
rm("ring", recursive=true, force=true)
```

## Leapfrogging Vortex Rings
```@example leapfrog
using FLOWVPM
using LinearAlgebra

function build_ring!(pfield, n_particles, r, circulation, x_center)
    sigma = 2*pi*r / n_particles
    d_theta = 2*pi / n_particles
    omega = circulation / (pi*r^2)      # Average vorticity
    dr = pi*r^2 / n_particles

    for i in 1:n_particles
        theta = d_theta * (i-1)
        X = [x_center, r*cos(theta), r*sin(theta)]
        Gamma_hat = cross(X, [-1.0, 0.0, 0.0])
        Gamma_hat = Gamma_hat / norm(Gamma_hat)
        Gamma = omega * Gamma_hat * dr

        add_particle(pfield, X, Gamma, sigma)
    end
    return pfield
end

rm("leapfrog", recursive=true, force=true)
max_particles = 40
pfield = ParticleField(max_particles);

# Ring parameters
r = 1.0                             # vortex ring radius
n_particles = max_particles         # number of particles in the ring
circulation = 10.0                   # circulation strength

build_ring!(pfield, max_particles/2, r, circulation, 0.0)
build_ring!(pfield, max_particles/2, r, circulation, r)

run_vpm!(pfield, 0.1, 200; save_path="leapfrog")
```

```@setup leapfrog
rm("leapfrog", recursive=true, force=true)
```
<img src="../media/vid/val_leapfrog04_6.gif" alt="Leapfrog simulation"/>