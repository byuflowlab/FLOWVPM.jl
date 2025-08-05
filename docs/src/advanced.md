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
Here we provide details on all relevant keyword setting for a `ParticleField`.

### formulation
The VPM formulation to be used
- [`rVPM`](@ref FLOWVPM.rVPM): The reformulated VPM equations (default).
- [`cVPM`](@ref FLOWVPM.cVPM): The classic VPM equations.
- [`formulation_tube_continuity`](@ref FLOWVPM.formulation_tube_continuity): VPM equations derived from tube conservation of mass.
- [`formulation_tube_momentum`](@ref FLOWVPM.formulation_tube_momentum): VPM equations derived from tube conservation of momentum.

### viscous
The viscous model used when propagating particles. Generally, aerodynamic simulations do not require viscous modeling.
- [`Inviscid`](@ref FLOWVPM.Inviscid): Uses no viscous modeling for the particles (default).
- [`CoreSpreading`](@ref FLOWVPM.CoreSpreading): Uses a core spreading viscous model in which particles grow in size at each step. After a defined amount of time the particle size is reset and the particle strength recalculated to maintain the vorticity field strength.
- [`ParticleStrengthExchange`](@ref FLOWVPM.ParticleStrengthExchange): Uses particle strength exchange to simulate viscosity.

### kernel
Describes how vorticity is distributed by the vortex particles.
- [`singular`](@ref FLOWVPM.singular): Singular vortex kernel.
- [`gaussian`](@ref FLOWVPM.gaussian): Gaussian vortex kernel.
- [`gaussianerf`](@ref FLOWVPM.gaussianerf): Gaussian error function vortex kernel (default).
- [`winckelmans`](@ref FLOWVPM.winckelmans): Winckelman vortex kernel

### UJ
Determines how the n-body particle interaction is solved. Users can provide custom UJ functions if desired.
- [`UJ_fmm`](@ref FLOWVPM.UJ_fmm): Uses the fast multipole method to solve the n-body problem (default).
- [`UJ_direct`](@ref FLOWVPM_direct): Loops through all particle interactions directly.

### Uinf
The freestream fluid velocity. Defaults to `SVector{3,Float64}(0,0,0)`.

### SFS
The subfilter-scale (SFS) model used in the reformulated VPM equations. The SFS model simulated the turbulent energy cascade. For simulations with little turbulence the model may be unnecessary. Custom modifications can be made, setting the coefficient or the alpha used is possible.
- [`noSFS`](@ref FLOWVPM.noSFS): No SFS model is used (default).
- [`SFS_Cs_nobackscatter`](@ref FLOWVPM.SFS_Cs_nobackscatter): Uses a constant coefficient for the SFS model.
- [`SFS_Cd_twolevel_nobackscatter`](@ref FLOWVPM.SFS_Cd_twolevel_nobackscatter): Uses a dynamic coefficient for the SFS model. Sets `alpha=0.999`. Recommended for high fidelity modeling.
- [`SFS_Cd_threelevel_nobackscatter`](@ref FLOWVPM.SFS_Cd_threelevel_nobackscatter): Uses a dynamic coefficient for the SFS model. Sets `alpha=0.667`, which reduces the effect of the SFS significantly.

### integration
The time integration scheme used in propogating particles
- [`euler`](@ref FLOWVPM.euler): Uses an euler time stepping scheme.
- [`rungekutta3`](@ref FLOWVPM.rungekutta3): Uses a low storage RK3 time stepping scheme (default).

### transposed
Determines how the stretching term is stored. Defaults to `true`. There is no need for users to modify this value.

### relaxation
To avoid numerical instability the particles are relaxed in order to ensure the field is divergence free.
- [`norelaxation`](@ref FLOWVPM.norelaxation): Does not relax the particles.
- [`pedrizzetti`](@ref FLOWVPM.pedrizzetti): Relaxation scheme where the vortex strength is aligned with the local vorticity. (default).
- [`correctedpedrizzetti`](@ref FLOWVPM.correctedpedrizzetti): Relaxation scheme where the vortex strength is aligned with the local vorticity. This version fixes the error in Pedrizzetti's relaxation that made the strength
to continually decrease over time.

### fmm
Settings for the fast multipole solver. Defaults autotune the FMM so that the absolute and relative tolerance of the velocity calculations are within 1e-3. See the the API for detailed settings
- [`FMM`](@ref FLOWVPM.FMM)

### useGPU
Determines whether or not to use GPU acceleration to solve the n-body problem. Defaults to CPU usage. Still experimental and does not work on SFS calculation, only works on NVIDIA GPUs. To allow GPU usage set to values greater than 0.

## Code Integration
For examples of integrating FLOWVPM into other codes please see [FLOWUnsteady](https://github.com/byuflowlab/FLOWUnsteady) and
[VortexLattice](https://github.com/byuflowlab/VortexLattice)