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

## Code Integration
For examples of integrating FLOWVPM into other codes please see [FLOWUnsteady](https://github.com/byuflowlab/FLOWUnsteady) and
[VortexLattice](https://github.com/byuflowlab/VortexLattice)