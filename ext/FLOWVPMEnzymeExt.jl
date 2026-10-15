#=##############################################################################
# DESCRIPTION
    Enzyme activity rules: pieces of a time step that carry no derivative, so
    that Enzyme reverse mode (default settings) differentiates a UJ_direct step.
=###############################################################################
module FLOWVPMEnzymeExt

import FLOWVPM
import EnzymeCore: EnzymeRules

const vpm = FLOWVPM

# Regularization-radius tuning (Roots.Brent breaks Enzyme type analysis). Exact
# while ρ/σ is fixed (autotune_reg_error=false; the direct path never reads it).
EnzymeRules.inactive(::typeof(vpm.solve_ρ_over_σ), args...) = nothing

# Filament topology (BitMatrix fields); built by the ParticleField constructor.
EnzymeRules.inactive_type(::Type{<:vpm.FilamentEdgeGraph}) = true
EnzymeRules.inactive_type(::Type{<:vpm.FilamentEdgeWorkspace}) = true

# Two-level dynamic SFS request/delivery flags (global IdDict{Any,Bool}): Bool plumbing.
EnzymeRules.inactive(::typeof(vpm._sfs_dsigma_request!), args...) = nothing
EnzymeRules.inactive(::typeof(vpm._sfs_dsigma_delivered!), args...) = nothing
EnzymeRules.inactive(::typeof(vpm._sfs_dsigma_requested), args...) = nothing
EnzymeRules.inactive(::typeof(vpm._sfs_dsigma_delivered), args...) = nothing

# Cache plumbing: drops the radix coupling after a reset (delete! on a global
# IdDict, a runtime-activity error under Enzyme); carries no derivative.
EnzymeRules.inactive(::typeof(vpm.clear_radix_fmm_cache!), args...) = nothing

end
