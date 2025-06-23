# ParticleField
FLOWVPM is centered around the ParticleField struct that contains all of the particle information and model settings.

`ParticleField(maxparticles)`

`maxparticles` is the maximum number of particles that can exist in the simulation. The memory for these particles is allocated at the time the `ParticleField` is created, so any attempt to add more than `maxparticles` particles will result in errors

## Keyword Arguments
`formulation`: VPM formulation. See Model Options for more details

`viscous`: Viscous scheme. With classic VPM viscous schemes help improve numerical stability, with the reformulated VPM they are not necessary unless significant viscosity actually exists.
<!-- I would say this uses a core spreading viscosity model via operator splitting to simulate fluid viscosity. I like the note that it's not needed for numerical stability -->

`np`: The number of particles in the field, defaults to 0 and is modifed by the `add_particle` and `remove_particle` functions.

`nt`: The current time step number. Defaults to 0.

`t`: Current time. Defaults to 0.0.

`transposed`: Determines how the vortex stretching is stored. Defaults to `true`. (There is no need for the user to modify this value.)
<!-- this actually refers to how the (v \cdot \nabla) \omega operator is implemented in code, and doesn't affect how stretching is stored; the transposed scheme has been recommended for stability -->

`fmm`: FMM settings. Defaults to `FMM(; p=4, ncrit=50, theta=0.4, nonzero_sigma=true, ε_tol=nothing)`. `p` is the maximum order of the multipole expansion, higher values result in higher accuracy. `ncrit` is the maximum leaf size. `theta` is the multipole acceptance criterion. `nonzero_sigma` improves the accuracy of the FMM when bodies have finite radius (should always be true as vortex particles have a nonzero radius). `ε_tol` (varepsilon) is the maximum absolute error allowed in the FMM calculations, by setting this value, the FMM dynamically sets the expansion order to satsify this error tolerance. See `Fastmultipole.jl` for more details.

`M`: Auxiliary storage for particle computations; users should never modify the default.

`toggle_rbf`: Switch to determine if the vorticity field should be calculated. This value changes frequently during time steps and should never be modified by users.

`toggle_sfs`: Switch to determine if the stretching term should be calculated. This value changes frequently during time steps and should never be modified by users.

`SFS`: The subfilter-scale scheme used. See Model Options for details.

`kernel`: The vortex particle kernel. See Model Options for details.

`UJ`: If set to UJ_fmm (default) uses the FMM to solve the particle interactions in O(n) time. If set to UJ_direct uses only direct calculations so solve for particle interactions in O(n^2) time. Unless debugging or checking for error produced by the FMM settings there is no need to use UJ_direct.

`Uinf`: Freestream velocity, takes the form `f(t)` where `t` is the current time.

`relaxation`: The divergence relaxation scheme, see Model Options for more details.

`integration`: Time integration scheme. Defaults to an RK3 method.

`useGPU`: Defaults to 0, still experimental. Uses CUDA to accelerate SFS calculations.

## Accessing Particle Information
`FLOWVPM` provides getter and setter functions for particles. Getter functions take the form `get_property(P)` where `P` is a particle, or `get_property(pfield::ParticleField, i::Int)` where `i` is the particle index. Getter functions return views. Setter functions take the from `set_property(P, val)` and `set_property(pfield, i, val)` where `val` is the new value to be set and is applied via broadcasting. Getter and setter functions are used by replacing "property" with the desired particle property such as `get_X(P)`. The following are the particle properties: 

- `X`: position
- `Gamma`: vortex strength
- `sigma`: radius
- `vol`: volume
- `circulation`: circulation
- `U`: velocity
- `vorticity`: vorticity 
- `J`: velocity gradient
- `M`: auxiliary computational storage
- `C`: SFS model parameters
- `static`: tag indicating whether this particle's states should evolve in time or not (sometimes used to represent solid bodies)
- `PSE`: Storage needed for `ParticleStrengthExchange`
- `SFS`: Storage of SFS values

## Adding Particles to the ParticleField
Calling `add_particle(pfield::ParticleField, X, Gamma, sigma)` will add a particle to the particle field with position `X`, circulation `Gamma`, and radius `sigma`. The function will error if `pfield.maxparticles` is exceeded. Another option is to call `add_particle(pfield, P)` where `P` is a particle.

## Removing Particles from the ParticleField
Calling `remove_particle(pfield::ParticleField, i:Int)` will remove particle `i` from `pfield`.

## Propogating Particles
Calling `nextstep(pfield::ParticleField, dt::Real; relax=false, custom_UJ=nothing)` will propogate the particles in `pfield` forward `dt` in time. `relax` can be used to set relaxation to only occur under certain conditions. `custom_UJ` can be set if the user wishes to calculate the induced velocity and velocity gradient manually.

run_vpm!

## Model Options
For details on models please see:

Alvarez, Eduardo J., "Reformulated Vortex Particle Method and Meshless Large Eddy Simulation of Multirotor Aircraft" (2022). Theses and Dissertations. 9589.
https://scholarsarchive.byu.edu/etd/9589

### VPM Fomulation
All VPM formulations use the same basic equations with varying coefficients $f$ and $g$ as defined in \cite{}.
- `rVPM` enforces conservation of mass and momentum for a spherical fluid element and is the default formulation ($f=0$, $g=1/5$)
- `cVPM` refers to the classic VPM equation, and enforces neither conservation of mass nor momentum ($f=g=0$)
- `formulation_tube_continuity` enforces conservation of mass for a vortex tube ($f=1/2$, $g=0$)
- `formulation_tube_momentum` enforces conservation of angular momentum for a vortex tube ($f=1/4$, $g=1/4$)

### Kernels
- `singular`
- `gaussian`
- `gaussianerf` (default)
- `winckelmans`

### Relaxation Schemes
These relaxation schemes are designed to enforce a divergence free velocity field and improves numerical stability

- `norelaxation`
- `pedrizzetti` (default)
- `correctedpedrizzetti` is a modification to the pedrizzetti relaxation that preserves the vortex strength magnitude.

### Subfilter-scale (SFS) Models
The SFS model is designed to model turbulent diffusion

- `noSFS` (default)
- `SFS_Cs_nobackscatter`
- `SFS_Cd_twolevel_nobackscatter` (recommended)
- `SFS_Cd_threelevel_nobackscatter`

### Viscous Schemes

The following schemes are used to simulate fluid viscosity.

Note: classic VPM often introduces non-physical levels of viscosity to improve numerical stability. The `rVPM` formulation has been shown to provide sufficient stability without introducing artificial viscosity.

- `Inviscid` (default)
- `CoreSpreading(nu, sgm0)` where `nu` is kinematic viscosity and `sgm0` is the core size after reset. This method works by increasing particle size at each time step to account for the viscous effects of the fluid. To keep particles from becoming too large the particle sizes are reset if particle size exceeds a certain threshold that can be set with keyword arguments. 
  - Only compatible with `gaussianerf` kernel
- `ParticleStrengthExchange(nu)` where `nu` is the kinematic viscosity. Uses the diffusion equation to smear the vortex strength over nearby particles.

### Time Stepping Schemes
- `euler` first-order Euler scheme.
- `rungekutta3` third-order Runge-Kutta scheme. (default)