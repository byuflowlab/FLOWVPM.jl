#=##############################################################################
# DESCRIPTION
    Particle field struct definition.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################


################################################################################
# PARTICLE FIELD STRUCT
################################################################################
mutable struct ParticleFieldStretch{T, V<:ViscousScheme} <: AbstractParticleField{T, V}
    # User inputs
    maxparticles::Int                           # Maximum number of particles
    particles::Array{ParticleStretch{T}, 1}     # Array of particles
    bodies::fmm.Bodies                          # ExaFMM array of bodies
    viscous::V                                  # Viscous scheme

    # Internal properties
    np::Int                                     # Number of particles in the field
    nt::Int                                     # Current time step number
    t::Float64                                  # Current time

    # Solver setting
    kernel::Kernel                              # Vortex particle kernel
    UJ::Function                                # Particle-to-particle calculation

    # Optional inputs
    Uinf::Function                              # Uniform freestream function Uinf(t)
    transposed::Bool                            # Transposed vortex stretch scheme
    relax::Bool                                 # Activates relaxation scheme
    rlxf::Float64                               # Relaxation factor (fraction of dt)
    integration::Function                       # Time integration scheme
    fmm::FMM                                    # Fast-multipole settings
    splitparticles::T                           # l/l0 crit to when to split particles


    ParticleFieldStretch{T, V}(
                        maxparticles,
                        particles, bodies, viscous;
                        np=0, nt=0, t=0.0,
                        kernel=gaussianerf,
                        UJ=UJ_fmm,
                        Uinf=t->zeros(3),
                        transposed=true,
                        relax=true, rlxf=0.3,
                        integration=rungekutta3,
                        fmm=FMM(),
                        splitparticles=T(1.5)
                 ) where {T, V} = new(
                        maxparticles,
                        particles, bodies, viscous,
                        np, nt, t,
                        kernel,
                        UJ,
                        Uinf,
                        transposed,
                        relax, rlxf,
                        integration,
                        fmm,
                        splitparticles
                  )
end

function ParticleFieldStretch(maxparticles::Int; viscous::V=Inviscid(),
                                            optargs...) where {V<:ViscousScheme}
    # Memory allocation by C++
    bodies = fmm.genBodies(maxparticles)

    # Have Julia point to the same memory than C++
    particles = [ParticleStretch(fmm.getBody(bodies, i-1)) for i in 1:maxparticles]

    # Set index of each particle
    for (i, P) in enumerate(particles)
        P.index[1] = i
    end

    # Generate and return ParticleField
    return ParticleFieldStretch{RealFMM, V}(maxparticles, particles, bodies, viscous;
                                                               np=0, optargs...)
end
##### FUNCTIONS ################################################################
"""
  `add_particle(self::ParticleField, X, Gamma, sigma; vol=0, index=np)`

Add a particle to the field.
"""
function add_particle(self::ParticleFieldStretch, X, circulation, l0, sigma; index=-1)
    # ERROR CASES
    if get_np(self)==self.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(self.maxparticles)"*
                                                            " has been reached")
    end

    # Fetch next empty particle in the field
    P = get_particle(self, get_np(self)+1; emptyparticle=true)

    # Populate the empty particle
    P.X .= X
    P.circulation .= circulation
    P.l0 .= l0
    P.sigma .= sigma
    P.vol .= pi*sigma.^2 * sqrt(P.l0[1]^2 + P.l0[2]^2 + P.l0[3]^2)
    P.l .= l0
    P.Gamma .= l0
    P.Gamma .*= circulation
    P.index .= index==-1 ? get_np(self) : index

    # Add particle to the field
    self.np += 1

    return nothing
end

"""
  `remove_particle(pfield::ParticleField, i)`

Remove the i-th particle in the field. This is done by moving the last particle
that entered the field into the memory slot of the target particle. To remove
particles sequentally, you will need to go from the last particle back to the
first one (see documentation of `get_particleiterator` for an example).
"""
function remove_particle(self::ParticleFieldStretch, i::Int)
    if i<=0
        error("Requested removal of invalid particle index $i")
    elseif i>get_np(self)
        error("Requested removal of particle $i, but there is only"*
                                " $(get_np(self)) particles in the field.")
    end

    if i != get_np(self)
        # Overwrite target particle with last particle in the field
        fmm.overwriteBody(self.bodies, i-1, get_np(self)-1)
    end

    Ptarg = get_particle(self, i)
    Plast = get_particle(self, get_np(self))

    Ptarg.circulation .= Plast.circulation
    Ptarg.l0 .= Plast.l0
    Ptarg.l .= Plast.l

    # Remove last particle in the field
    self.np -= 1

    return nothing
end


"""
    `get_particleiterator(pfield::ParticleField; start_i=1, end_i=np)`

Return an iterator over particles that can be used as follows

```julia-repl
julia> # Initiate particle field
       pfield = FLOWVPM.ParticleField(10);

julia> # Add particles
       for i in 1:4
           FLOWVPM.add_particle(pfield, (i*10^0, i*10^1, i*10^2), zeros(3), 1.0)
       end

julia> # Iterate over particles
       for P in FLOWVPM.get_particleiterator(pfield)
           println(P.X)
       end
[1.0, 10.0, 100.0]
[2.0, 20.0, 200.0]
[3.0, 30.0, 300.0]
[4.0, 40.0, 400.0]
```
"""
function get_particleiterator(self::ParticleFieldStretch{T};
                                        start_i::Int=1, end_i::Int=-1, reverse=false) where {T}
    # ERROR CASES
    if end_i > get_np(self)
        error("Requested end_i=$(end_i), but there is only $(get_np(self))"*
                                                    " particles in the field.")
    end

    strt = reverse ? (end_i==-1 ? get_np(self) : end_i) : start_i
    stp = reverse ? -1 : 1
    nd = reverse ? start_i : (end_i==-1 ? get_np(self) : end_i)

    return view( self.particles, strt:stp:nd
                )::SubArray{FLOWVPM.ParticleStretch{T},1,Array{FLOWVPM.ParticleStretch{T},1},Tuple{StepRange{Int64,Int64}},true}
end

"""
  `nextstep(self::ParticleField, dt; relax=false)`

Steps the particle field in time by a step `dt`.
"""
function nextstep(self::ParticleFieldStretch, dt::Real; optargs...)

    # Convert vortex tube length into vectorial circulation
    for P in iterator(self)
        P.Gamma .= P.l
        P.Gamma .*= P.circulation[1]
    end

    # Step in time
    if get_np(self)!=0
        self.integration(self, dt; optargs...)
    end

    # Convert vectorial circulation into vortex tube length and adjust
    # cross-sectional area
    for P in iterator(self)
        P.l .= P.Gamma
        P.l ./= P.circulation[1]
        # NOTE: This overwrites the core-speading scheme!
        P.sigma .= sqrt( P.vol[1] / (pi*sqrt(P.l[1]^2 + P.l[2]^2 + P.l[3]^2)) )
    end

    # Split particles
    for (pi, P) in enumerate(iterator(self; reverse=true))

        norml = sqrt(P.l[1]^2+P.l[2]^2+P.l[3]^2)
        crit = norml/sqrt(P.l0[1]^2+P.l0[2]^2+P.l0[3]^2)
        if crit >= self.splitparticles

            # Number of particles that would make the length
            # smaller than the criterion
            nsplit = ceil(Int, crit)
            # println("\t\tSplitting into $nsplit particles")

            # Length of each section
            lsplit = norml/nsplit

            # Unit vector of vortex tube
            ldir1 = P.l[1]/norml
            ldir2 = P.l[2]/norml
            ldir3 = P.l[3]/norml

            # Starting and end point of vortex tube
            Xstr1 = P.X[1] - norml/2*ldir1
            Xstr2 = P.X[2] - norml/2*ldir2
            Xstr3 = P.X[3] - norml/2*ldir3
            Xend1 = P.X[1] + norml/2*ldir1
            Xend2 = P.X[2] + norml/2*ldir2
            Xend3 = P.X[3] + norml/2*ldir3

            # Total properties to distribute
            circulation = P.circulation[1]
            voltot = P.vol[1]
            sigma = P.sigma[1]

            # Remove particle
            remove_particle(self, pi)

            # New properties
            l0 = (lsplit*ldir1, lsplit*ldir2, lsplit*ldir3)

            # Create new particles
            for n in 1:nsplit
                aux = (n-1)*lsplit + lsplit/2
                X1 = Xstr1 + aux*ldir1
                X2 = Xstr2 + aux*ldir2
                X3 = Xstr3 + aux*ldir3
                add_particle(self, (X1, X2, X3), circulation, l0, sigma)
            end
        end

    end
    println("\t\t\t$(get_np(self))")

    # Updates time
    self.t += dt
    self.nt += 1
end

##### INTERNAL FUNCTIONS #######################################################
function _reset_particles(self::ParticleFieldStretch{T}) where {T}
    tzero = zero(T)
    for P in iterator(self)
        P.U[1] = tzero
        P.U[2] = tzero
        P.U[3] = tzero
        P.J[1, 1] = tzero
        P.J[2, 1] = tzero
        P.J[3, 1] = tzero
        P.J[1, 2] = tzero
        P.J[2, 2] = tzero
        P.J[3, 2] = tzero
        P.J[1, 3] = tzero
        P.J[2, 3] = tzero
        P.J[3, 3] = tzero
    end
end
##### END OF PARTICLE FIELD#####################################################
