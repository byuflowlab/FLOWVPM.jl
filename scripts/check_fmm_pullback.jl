using ForwardDiff
using ReverseDiff
using FLOWVPM
vpm = FLOWVPM
using StaticArrays

# generates a pfield with all entries random (except for static, which is set at zero).
function make_full_pfield(input_array; maxp = 10, UJ=vpm.UJ_direct, SFS=vpm.SFS_default, relax=vpm.norelaxation, kernel=vpm.kernel_default)

    T = eltype(input_array)
    pfield = vpm.ParticleField(maxp, T; UJ=UJ, SFS=SFS, relaxation=relax, kernel=kernel)
    for i=1:maxp
        vpm.add_particle(pfield, input_array[1:3, 1], ones(T, 3), 1.4*one(T)) # makes sure memory is properly initialized
    end
    vpm.set_all_fields(pfield, input_array)
    for i=1:maxp
        vpm.set_static(pfield, i, zero(T))
    end
    return pfield

end

function run_fmm_pullback_check(; np = 10, use_forwarddiff = true)

    println("running fake fmm check")
    input_vector = rand(Float64, np*vpm.nfields)
    function _f(_input_vector)

        pfield = make_full_pfield(reshape(_input_vector, vpm.nfields, np); maxp = np, UJ=vpm.UJ_fmm)
        pfield.UJ(pfield; reset_sfs=vpm.isSFSenabled(pfield.SFS), reset=true, sfs=vpm.isSFSenabled(pfield.SFS))

        return sum(pfield.particles)/vpm.nfields/np
    end

    if use_forwarddiff == true
        @time primal = _f(input_vector)
        @time fd_gradient = ForwardDiff.gradient(_f, input_vector)
        @time rd_gradient = ReverseDiff.gradient(_f, input_vector)

        println("primal (sum of particle states): $primal")
        println("maximum forwarddiff deriv: $(maximum(abs.(fd_gradient)))")
        println("maximum reversediff deriv: $(maximum(abs.(rd_gradient)))")
        println("maximum error: $(maximum(abs.(fd_gradient .- rd_gradient)))")
    else
        @time primal = _f(input_vector)
        @time rd_gradient = ReverseDiff.gradient(_f, input_vector)

        println("primal (sum of particle states): $primal")
        println("reverse-mode gradient: $rd_gradient")
        println("maximum reversediff deriv: $(maximum(abs.(rd_gradient)))")
    end

end

println("running fmm pullback check with 2 particles:")
run_fmm_pullback_check(; np = 2)

println("running fmm pullback check with 10 particles:")
run_fmm_pullback_check(; np = 10)

println("running fmm pullback check with 100 particles:")
run_fmm_pullback_check(; np = 100)
