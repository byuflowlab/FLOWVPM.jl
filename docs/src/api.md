# Reference

## ParticleField
```@docs
ParticleField
```

## Formulations
```@docs
FLOWVPM.rVPM
FLOWVPM.cVPM
FLOWVPM.formulation_tube_continuity
FLOWVPM.formulation_tube_momentum
```

## Viscous Models
```@docs
FLOWVPM.Inviscid
FLOWVPM.CoreSpreading
FLOWVPM.ParticleStrengthExchange
```

## FMM
```@docs
FLOWVPM.FMM
```

## Subfilter-Scale Models
```@docs
FLOWVPM.noSFS
FLOWVPM.SFS_Cs_nobackscatter
FLOWVPM.SFS_Cd_twolevel_nobackscatter
FLOWVPM.SFS_Cd_threelevel_nobackscatter
```

## Particle Kernels
```@docs
FLOWVPM.singular
FLOWVPM.gaussian
FLOWVPM.gaussianerf
FLOWVPM.winckelmans
```

## Relaxation Schemes
```@docs
FLOWVPM.norelaxation
FLOWVPM.pedrizzetti
FLOWVPM.correctedpedrizzetti
```

## Time Integration
```@docs
FLOWVPM.euler
FLOWVPM.rungekutta3
```