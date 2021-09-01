#=##############################################################################
# DESCRIPTION
    Subfilter-scale (SFS) turbulence schemes for large eddy simulation. See
20210901 notebook for theory and implementation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################


################################################################################
# ABSTRACT SFS SCHEME TYPE
################################################################################
abstract type SubFilterScale end

# Make SFS object callable
"""
    Implementation of calculations associated with subfilter-scale turbulence
model.

NOTE: Any implementation is expected to evaluate UJ and SFS terms of the
particles which will be used by the time integration routine so make sure they
are stored in the memory (see implementation of `ConstantSFS` as an example).
"""
function (SFS::SubFilterScale)(pfield)
    error("SFS evaluation not implemented!")
end
##### END OF SFS SCHEME ########################################################





################################################################################
# NO SFS SCHEME
################################################################################
struct NoSFS <: SubFilterScale end

function (SFS::NoSFS)(pfield; optargs...)
    # Reset U and J to zero
    _reset_particles(pfield)

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)
end
##### END OF NO SFS SCHEME #####################################################





################################################################################
# CONSTANT-COEFFICIENT SFS SCHEME
################################################################################
struct ConstantSFS{R} <: SubFilterScale
    model::Function                 # Model of subfilter scale contributions
    Cs::R                           # Model coefficient

    ConstantSFS{R}(model, Cs=R(1)) where {R} = new(model, Cs)
end

function (SFS::ConstantSFS)(pfield; optargs...)
    # Reset U and J to zero
    _reset_particles(pfield)

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)

    # Calculate subgrid-scale contributions
    _reset_particles_sfs(pfield)
    SFS.model(pfield)

    # "Calculate" model coefficient
    for p in iterator(pfield)
        p.C[1] = SFS.Cs
    end
end
##### END OF CONSTANT SFS SCHEME ###############################################





################################################################################
# DYNAMIC-PROCEDURE SFS SCHEME
################################################################################
struct DynamicSFS <: SubFilterScale
    model::Function                 # Model of subfilter scale contributions
end

function (SFS::DynamicSFS)(pfield; a=1, b=1)
    # Reset U and J to zero
    _reset_particles(pfield)

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)

    # Calculate subgrid-scale contributions
    _reset_particles_sfs(pfield)
    SFS.model(pfield)

    # "Calculate" model coefficient
    nothing
end
##### END OF DYNAMIC SFS SCHEME ################################################
