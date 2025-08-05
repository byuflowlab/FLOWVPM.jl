# Advanced Usage
For more advanced examples please see the examples folder.

## Accessing the ParticleField
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

## Particle Field Settings
Here we provide details concerning all settings for a [`ParticleField`](@ref). Keyword arguments to the constructor (such as governing equations, fast multipole backend settings, and the turbulence model) are reviewed in the following subsections.

### `formulation::Formulation`
The VPM formulation to be used. Options include:
- [`rVPM`](@ref FLOWVPM.rVPM): The reformulated VPM equations (default).
- [`cVPM`](@ref FLOWVPM.cVPM): The classic VPM equations.
- [`formulation_tube_continuity`](@ref FLOWVPM.formulation_tube_continuity): VPM equations derived from tube conservation of mass.
- [`formulation_tube_momentum`](@ref FLOWVPM.formulation_tube_momentum): VPM equations derived from tube conservation of momentum.

### `viscous::ViscousScheme`
The viscous model used when propagating particles.
- [`Inviscid`](@ref FLOWVPM.Inviscid): Uses no viscous modeling for the particles (default).
- [`CoreSpreading`](@ref FLOWVPM.CoreSpreading): Uses a core spreading viscous model in which particles grow in size at each step. After a defined amount of time the particle size is reset and the particle strength recalculated to maintain the vorticity field strength.
- [`ParticleStrengthExchange`](@ref FLOWVPM.ParticleStrengthExchange): Uses particle strength exchange to simulate viscosity.

### `kernel::Kernel`
Regularization scheme used to de-singularize the vorticity and velocity fields.
- [`singular`](@ref FLOWVPM.singular): Singular vortex kernel (e.g. no regularization).
- [`gaussian`](@ref FLOWVPM.gaussian): Gaussian vortex kernel.
- [`gaussianerf`](@ref FLOWVPM.gaussianerf): Gaussian error function vortex kernel (default).
- [`winckelmans`](@ref FLOWVPM.winckelmans): Winckelman vortex kernel

### `UJ`
Function used to solve the ``N``-body problem to obtain the velocity field, vorticity field, and subfilter scale contributions.
- [`UJ_fmm`](@ref FLOWVPM.UJ_fmm): Uses the fast multipole method to solve the ``N``-body problem (default).
- [`UJ_direct`](@ref FLOWVPM.UJ_direct(pfield::FLOWVPM.ParticleField)): Loops through all particle interactions directly.

### `Uinf::Function`
The freestream fluid velocity. Defaults to `(t) -> SVector{3,Float64}(0,0,0)`.

### `SFS::SubFilterScale`
The subfilter-scale (SFS) model used in the reformulated VPM equations. The SFS model simulated the turbulent energy cascade. For simulations with little turbulence the model may be unnecessary. Custom modifications can be made, setting the coefficient or the alpha used is possible.
- [`noSFS`](@ref FLOWVPM.noSFS): No SFS model is used (default).
- [`SFS_Cs_nobackscatter`](@ref FLOWVPM.SFS_Cs_nobackscatter): Uses a constant coefficient for the SFS model.
- [`SFS_Cd_twolevel_nobackscatter`](@ref FLOWVPM.SFS_Cd_twolevel_nobackscatter): Uses a dynamic coefficient for the SFS model. Sets `alpha=0.999`. Recommended for high fidelity modeling.
- [`SFS_Cd_threelevel_nobackscatter`](@ref FLOWVPM.SFS_Cd_threelevel_nobackscatter): Uses a dynamic coefficient for the SFS model. Sets `alpha=0.667`, which reduces the effect of the SFS significantly.

### `integration`
The time integration scheme used in propogating particles states forward in time.
- [`euler`](@ref FLOWVPM.euler): Uses an euler time stepping scheme.
- [`rungekutta3`](@ref FLOWVPM.rungekutta3): Uses a low storage RK3 time stepping scheme (default).

### `transposed::Bool`
Determines how the stretching term is calculated. Defaults to `true`, which is recommended for stability.

### `relaxation::Relaxation`
Scheme used to ensure the field is divergence free.
- [`norelaxation`](@ref FLOWVPM.norelaxation): Does nothing (e.g. no relaxation).
- [`pedrizzetti`](@ref FLOWVPM.pedrizzetti): Relaxation scheme where the vortex strength is aligned with the local vorticity. (default).
- [`correctedpedrizzetti`](@ref FLOWVPM.correctedpedrizzetti): Relaxation scheme where the vortex strength is aligned with the local vorticity while preserving the magnitude of the particle strength.

### `fmm::FMM`
Settings for the fast multipole solver. Defaults autotune the FMM so that the absolute and relative tolerance of the velocity calculations are within 1e-3. See the the [`FMM`](@ref FLOWVPM.FMM) API for detailed settings

### `useGPU::Int`
Determines whether or not to use GPU acceleration with `CUDA.jl` to solve the n-body problem. Still experimental and does not work on SFS calculation, only works on NVIDIA GPUs. To allow GPU usage set to values greater than 0. (default is 0).

## Code Integration
For examples of integrating FLOWVPM into other codes please see [FLOWUnsteady](https://github.com/byuflowlab/FLOWUnsteady) and
[VortexLattice](https://github.com/byuflowlab/VortexLattice)