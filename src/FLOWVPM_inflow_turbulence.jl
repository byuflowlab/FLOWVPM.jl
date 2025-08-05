#=##############################################################################
# DESCRIPTION
    Inflow Turbulence schemes.

# AUTHORSHIP
  * Author    : Benjamin Varela
  * Email     : ben.varela@me.com
  * Created   : Aug 2025
=###############################################################################

abstract type InflowTurbulenceScheme end

function inflow_turbulence(pfield, scheme::InflowTurbulenceScheme, dt)
    error("Inflow turbulence scheme not implemented for type: $(typeof(scheme))")
end

inflow_turbulence(pfield, dt) = inflow_turbulence(pfield, pfield.inflow_turbulence)

# NoInflowTurbulence is a placeholder for when no inflow turbulence is needed.
struct NoInflowTurbulence <: InflowTurbulenceScheme end

function inflow_turbulence(pfield, scheme::NoInflowTurbulence) return nothing end
function inflow_turbulence_convect(pfield, scheme::NoInflowTurbulence, dt) return nothing end



struct SyntheticEddyMethod <: InflowTurbulenceScheme
    eddydomain::SyntheticEddy.EddyDomain
end

function inflow_turbulence(pfield, scheme::SyntheticEddyMethod)
    U = view(pfield.particles, U_INDEX, 1:pfield.np)
    J = view(pfield.particles, J_INDEX, 1:pfield.np)
    X = view(pfield.particles, X_INDEX, 1:pfield.np)
    SyntheticEddy.compute_fluctuations!(U, X, scheme.eddydomain; compute_hessian=J)
    return nothing
end

function inflow_turbulence_convect(pfield, scheme::SyntheticEddyMethod, dt)
    SyntheticEddy.convect_eddies!(scheme.eddydomain, dt)
    return nothing
end

