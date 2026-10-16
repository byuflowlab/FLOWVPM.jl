#=##############################################################################
# DESCRIPTION
    Enzyme activity rules: pieces of a time step that carry no derivative, so
    that Enzyme reverse mode (default settings) differentiates a UJ_direct step.
=###############################################################################
module FLOWVPMEnzymeExt

import FLOWVPM
import EnzymeCore
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

# Enzyme forward mode leaves the shadow of an array that a non-inlined callee
# fills with a constant (`zeros`, `fill!`) and returns uninitialized: the
# tangent then reads heap garbage (NaN, 1e300, wrong values, varying per call;
# Enzyme 0.13.216). UJ_direct's FastMultipole buffers are allocated that way, so
# return explicitly zeroed shadows.
for f in (vpm.fmm.allocate_target_buffer, vpm.fmm.allocate_source_buffer)
    @eval function EnzymeRules.forward(config::EnzymeRules.FwdConfig, func::EnzymeCore.Const{typeof($f)},
                                       ::Type{RT}, args::EnzymeCore.Annotation...) where RT
        b = func.val(map(a -> a.val, args)...)
        N = EnzymeRules.width(config)
        db = N == 1 ? zero(b) : ntuple(_ -> zero(b), Val(N))
        if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
            return N == 1 ? EnzymeCore.Duplicated(b, db) : EnzymeCore.BatchDuplicated(b, db)
        end
        return EnzymeRules.needs_shadow(config) ? db : EnzymeRules.needs_primal(config) ? b : nothing
    end
end

# Device-kernel scheme/time-step constants (dt, RK3 coefficients, rlxf, SFS
# bounds): no derivative, and KernelAbstractions rejects Active GPU arguments.
EnzymeRules.inactive_type(::Type{<:vpm.KConst}) = true

# Cache plumbing: drops the radix coupling after a reset (delete! on a global
# IdDict, a runtime-activity error under Enzyme); carries no derivative.
EnzymeRules.inactive(::typeof(vpm.clear_radix_fmm_cache!), args...) = nothing

# Core-size reset (Galerkin): the selection is integer bookkeeping and the report
# only prints; the linear solve on the selected geometry carries an implicit rule
# (its adjoint is one solve with the same preconditioned CG, `_gal_solve_adjoint`)
# instead of being differentiated iteration by iteration.
EnzymeRules.inactive(::typeof(vpm._gal_plan), args...) = nothing
# which particles to reset (index lists of the cores above a threshold / the largest)
EnzymeRules.inactive(::typeof(vpm._radix_oversize_above), args...) = nothing
EnzymeRules.inactive(::typeof(vpm._radix_oversize_top), args...) = nothing
EnzymeRules.inactive(::typeof(vpm._gal_report), args...) = nothing
EnzymeRules.inactive_type(::Type{<:vpm._GalSolve}) = true

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(vpm._gal_solve)}, ::Type{RT},
                                      Y, v, Xo, vo, Go, x0, sol) where RT
    c, sys, x = vpm._gal_solve_sys(Y.val, v.val, Xo.val, vo.val, Go.val, x0.val, sol.val)
    A = sys.A                  # its stored pairs live in the GalerkinReset's workspace: keep a copy
    A isa vpm._GalOp && (sys = merge(sys, (; A = vpm._GalOp(A.R, A.rp, copy(A.col), copy(A.val), A.y, A.ib, A.rb))))
    dc = zero(c)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? c : nothing,
                                       EnzymeRules.needs_shadow(config) ? dc : nothing, (sys, x, dc))
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(vpm._gal_solve)}, ::Type{RT}, tape,
                             Y, v, Xo, vo, Go, x0, sol) where RT
    sys, x, dc = tape
    g = vpm._gal_solve_adjoint(sys, x, dc, sol.val)
    for (a, ga) in ((Y, g.Y), (v, g.v), (Xo, g.Xo), (vo, g.vo), (Go, g.Go))
        a isa EnzymeCore.Duplicated && (a.dval .+= ga)
    end
    dc .= 0                    # x0 (warm start) does not change the solution: no adjoint
    return ntuple(_ -> nothing, 7)
end

end
