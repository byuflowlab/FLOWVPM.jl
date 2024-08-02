#=##############################################################################
# DESCRIPTION
    Functions for generating, probing, and processing the fluid domain (velocity
    and vorticity fields) induced by particle field solutions.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Oct 2021
=###############################################################################


"""
    computefluiddomain( pfield::FLOWVPM.ParticleField,
                        grids::Array{<:GeometricTools.AbstractGrid};
                        optargs...
                        )

Evaluate the velocity and vorticity field induced by the particle field
`pfield` at all nodes in a set of grids `grids`. The fields are added as
solution fields `U` and `W` in each grid. The analytic Jacobian of the velocity
field can also be saved using the optional argument `add_J=true`.


# OPTIONAL ARGUMENTS

## Processing options
* `add_J::Bool=false`       : Add the solution fields `J1`, `J2`, and `J3` to
                                each grid, where Ji[j]=dUi/dxj.
* `add_Uinf::Bool=false`    : It evaluates and adds the uniform freestream to
                                the `U` field.
* `add_Wapprox::Bool=false` : It evaluates and saves the RBF-approximated
                                vorticity field under the field `Wapprox`.
* `zeta::Function=FLOWVPM.zeta_fmm` : Method for evaluating RBF-approximated vorticity
                                (used only if `add_Wapprox==true`).
* `scale_sigma::Real=1.0`   : It rescales the smoothing radius of each particle
                                by this factor before evaluating the particle
                                field.

## Output options
* `save_path::String`       : If used, it will save the grids as XDMF files
                                under this path.
* `file_pref::String`       : Prefix for XDMF files.
* `grid_names::String`      : Name of each grid for XDMF files. If not given, it
                                will generate their names automatically.
* `num::Int`                : If given, the name of the XDMF files will be
                                `"\$(file_pref)\$(grid_names[i]).\$(num).vtk"`
* `verbose::Bool=true`      : Activate/deactivate verbose.
* `v_lvl::Int=0`            : Indentation level for printing verbose.


**NOTE:** The solution fields `U`, `W`, and Jacobian do not include the freestream
        field, but rather they only include the fields induced by the particles.
        To add the freestream to `U`, use the optional argument `add_Uinf=true`.
"""
function computefluiddomain(pfield::vpm.ParticleField,
                                    grids::Array{<:gt.AbstractGrid};
                                    # PROCESSING OPTIONS
                                    add_J=false,
                                    add_Uinf=false,
                                    add_Wapprox=false,
                                    zeta=vpm.zeta_fmm,
                                    scale_sigma=1.0,
                                    f_Gamma=1e-2,   # Factor used to add the nodes as particles
                                    f_sigma=0.5,    # Factor used to add the nodes as particles
                                    remove_nodeparticles=true,
                                    # OUTPUT OPTIONS
                                    save_path=nothing,
                                    file_pref="",
                                    grid_names="automatic",
                                    num=nothing,
                                    verbose=true, v_lvl=0,
                                    Uinf=zeros(3),
                                    )

    _grid_names = grid_names=="automatic" ? ("Grid$(gi)" for gi in 1:length(grids)) : grid_names
    str = ""

    # t = @elapsed begin

        np = vpm.get_np(pfield)           # Original number of particles

        # Rescale smoothing radii
        for P in vpm.iterate(pfield; include_static=true)
            sigma = vpm.get_sigma(P)
            sigma[1] *= scale_sigma
        end

        # Estimate average sigma and minimum Gamma
        meansigma = 0
        minnormGamma = Inf
        for P in vpm.iterate(pfield; include_static=true)
            sigma = vpm.get_sigma(P)
            Gamma = vpm.get_Gamma(P)
            meansigma += sigma[1]

            normGamma = sqrt(Gamma[1]^2 + Gamma[2]^2 + Gamma[3]^2)
            if normGamma < minnormGamma
                minnormGamma = normGamma
            end
        end
        meansigma /= np

        # Add grid nodes to the particle field
        Gamma = (f_Gamma*minnormGamma for i in 1:3)
        sigma = f_sigma*meansigma

        for grid in grids
            for ni in 1:grid.nnodes
                vpm.add_particle(pfield, view(grid.nodes, 1:3, ni), Gamma, sigma)
            end
        end

    # end

    # if verbose
    #     println("\t"^(v_lvl)*"Add nodes as particles:\t$(round(t, digits=1)) s")
    #     println("\t"^(v_lvl)*"Number of particles:\t$(vpm.get_np(pfield))")
    # end

    # Pre-allocate memory for U and W in grids
    fields = ["U", "W"]
    if add_J; for i in 1:3; push!(fields, "J$i"); end; end;
    if add_Wapprox; push!(fields, "Wapprox"); end;

    # t = @elapsed begin
        for field_name in fields
            for grid in grids
                if !(field_name in keys(grid.field))
                    arr = zeros(3, grid.nnodes)
                    gt.add_field(grid, field_name, "vector", arr, "node")
                end
            end
        end
    # end

    # if verbose
    #     println("\t"^(v_lvl)*"Pre-allocate U and W memory:\t$(round(t, digits=1)) s")
    # end

    # Evaluate particle field
    pfield.UJ(pfield; reset=true)

    # if verbose
    #     println("\t"^(v_lvl)*"Evaluate UJ:\t\t$(round(t, digits=1)) s")
    # end

    # Add freestream
    if add_Uinf
        # Uinf = pfield.Uinf(pfield.t)
        for P in vpm.iterate(pfield; start_i=np+1)
            vpm.get_U(P) .+= Uinf
        end
    end

    # Evaluate RBF-approximated W
    if add_Wapprox
        zeta(pfield)

        if verbose
            println("\t"^(v_lvl)*"Evaluate Wapprox:\t\t$(round(t, digits=1)) s")
        end
    end

    # t = @elapsed begin

        prev_np = np

        for (grid, gridname) in zip(grids, _grid_names)

            nnodes = grid.nnodes
            rng = prev_np+1:prev_np+nnodes

            particles = vpm.iterate(pfield; start_i=rng.start, end_i=rng.stop, include_static=true)

            U  = grid.field["U"]["field_data"]
            U .= (vpm.get_U(P)[i] for i in 1:3, P in particles)

            W  = grid.field["W"]["field_data"]
            W .= (fun(P) for fun in (vpm.get_W1, vpm.get_W2, vpm.get_W3), P in particles)

            if add_J
                for i in 1:3
                    Ji = grid.field["J$(i)"]["field_data"]
                    Ji .= (P.J[i, j] for j in 1:3, P in particles)
                end
            end

            if add_Wapprox
                Wapprox = grid.field["Wapprox"]["field_data"]
                Wapprox .= (get_J(P)[i] for i in 1:3, P in particles)
            end

            # Save fluid domain as VTK file
            if save_path != nothing
                str *= gt.save(grid, file_pref*gridname;
                                              path=save_path, num=num, time=pfield.t)
            end

            prev_np += nnodes
        # end

    end

    # if verbose
    #     println("\t"^(v_lvl)*"Save VTK:\t\t$(round(t, digits=1)) s")
    # end

    # Remove node particles
    if remove_nodeparticles
        for pi in vpm.get_np(pfield):-1:np+1
            vpm.remove_particle(pfield, pi)
        end
    end

    # Restore original smoothing radii
    for P in vpm.iterate(pfield; include_static=true)
        sigma = vpm.get_sigma(P)
        sigma[1] /= scale_sigma
    end

    return str
end


"""
    computefluiddomain(pfield::vpm.ParticleField,
                        nums::Vector{Int}, read_path::String, file_pref::String,
                        grids;
                        origin=nothing,
                        orientation=nothing,
                        other_read_paths=[],
                        other_file_prefs=[],
                        userfunction_pfield=(pfield, num, grids)->nothing,
                        optargs...
                        )

Evaluate the fluid domain at each time step in `nums` that is induced by
a particle field saved under `read_path`. `file_pref` indicates the
prefix of the .h5 files to read.

To translate and re-orient the grids at each time step, the user can pass the
new origin vector and orientation matrix through the functions `origin` and
`orientation`, which will be called as `origin(pfield, num)` and
`orientation(pfield, num)` at each time step.

`pfield` is a place holder for loading the particles that are read, so the
user must make sure that sufficient memory has been preallocated to hold
the number of particles of each time step that will be read, plus the number
of nodes in the grids. The fluid domain will be evaluated using the UJ and FMM
configuration of the given `pfield` particle field.

To read and add more than one particle field at each time step, pass a list
of paths and prefixes through `other_read_paths` and `other_file_prefs`. This
is useful for reading and incluiding a set of static particles, for example.

Give it a function `userfunction_pfield` to pre-process the resulting particle
field before evaluating the fluid domain (*e.g.*, remove particles, resize core
sizes, etc).
"""
function computefluiddomain(pfield::vpm.ParticleField,
                                    nums, read_path::String, file_pref::String,
                                    grids;
                                    origin=nothing,
                                    orientation=nothing,
                                    other_file_prefs=[],
                                    other_read_paths=[],
                                    userfunction_pfield=(pfield, num, grids)->nothing,
                                    verbose=true, v_lvl=0, optargs...)

    # Memory pre-allocation
    O_prev, Oaxis_prev = zeros(3), Float64[i==j for i in 1:3, j in 1:3]
    T, M, Maux         = zeros(3), Float64[i==j for i in 1:3, j in 1:3], zeros(3, 3)
    O_new, Oaxis_new   = zeros(3), Float64[i==j for i in 1:3, j in 1:3]

    for (numi, num) in enumerate(nums)
        if verbose
            println("\t"^(v_lvl)*"Processing step $(num)"*
                    "\t($(numi) out of $(length(nums)))")
        end

        # Read particle field
        vpm.read!(pfield, file_pref*".$(num).h5"; path=read_path, overwrite=true)

        # Read additional particle fields
        for (fi, other_file_pref) in enumerate(other_file_prefs)
            vpm.read!(pfield, other_file_pref*".$(num).h5";
                                    path=other_read_paths[fi], overwrite=false)
        end

        # Translate and re-orient the grids
        if origin!=nothing || orientation!=nothing

            # New position of grids
            O_new .= origin!=nothing ? origin(pfield, num) : O_new

            # New orientation of grids
            Oaxis_new .= orientation!=nothing ? orientation(pfield, num) : Oaxis_new

            # Bring grids back to origin and reorient with global axes
            # and translate and rotate to new position---all at the same time
            # NOTE: I haven't verified this transformation so it may be buggy

            # M .= collect(Oaxis_prev')*Oaxis_new
            Maux .= transpose(Oaxis_prev)
            for i in 1:3
                for j in 1:3
                    M[i, j] = Maux[i,1]*Oaxis_new[1,j] + Maux[i,2]*Oaxis_new[2,j] + Maux[i,3]*Oaxis_new[3,j]
                end
            end

            # T .= O_new - collect(Oaxis_new')*Oaxis_prev*O_prev
            T .= O_new
            for i in 1:3
                T[i] -= M[1,i]*O_prev[1] + M[2,i]*O_prev[2] + M[3,i]*O_prev[3]
            end

            for grid in grids
                gt.lintransform!(grid, M, T; reset_fields=true)
            end

            O_prev .= O_new
            Oaxis_prev .= Oaxis_new
        end


        # Pass particle field to user-defined pre-processing function
        userfunction_pfield(pfield, num, grids)

        # Proceed to evaluate the particle field on the nodes of each grid
        computefluiddomain(pfield, grids;
                                    remove_nodeparticles=false,
                                     verbose=verbose, v_lvl=v_lvl+1, num=num,
                                     optargs...)
    end

    return nothing
end

"""
    computefluiddomain(maxparticles::Int, args...;
                        UJ::Function=vpm.UJ_fmm,
                        fmm::FLOWVPM.FMM=vpm.FMM(; p=4, ncrit=50, theta=0.4, nonzero_sigma=true),
                        pfield_optargs=[]
                        optargs...)

Like the other `computefluiddomain(args...; optargs...)` methods, but
automatically pre-allocating and initializing the particle field with the
given maximum number of particles, UJ evaluation method, and FMM configuration
(if FMM is used by UJ).

Use `pfield_optargs` to pass any additional optional arguments to the particle
field constructor.
"""
function computefluiddomain(maxparticles::Int, args...;
                                    UJ=vpm.UJ_fmm,
                                    fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, nonzero_sigma=true),
                                    pfield_optargs=[],
                                    verbose=true, v_lvl=0,
                                    optargs...)

    if verbose
        println("\t"^(v_lvl)*"Pre-allocating memory for $(maxparticles) max particles...")
    end

    # Pre-allocate memory
    pfield = vpm.ParticleField(maxparticles; UJ=UJ, fmm=fmm, pfield_optargs...)

    return computefluiddomain(pfield, args...;
                                      verbose=verbose, v_lvl=v_lvl, optargs...)
end



"""
    computefluiddomain(P_min, P_max, NDIVS, args...;
                        spacetransform=nothing,
                        O=zeros(3), Oaxis=Float64[i==j for i in 1:3, j in 1:3],
                        optargs...)`

Like the other `computefluiddomain(args...; optargs...)` methods,
but automatically generating a fluid domain grid. The grid is generated as a
Cartesian box with minimum and maximum corners `P_min` and `P_max` and `NDIVS`
cells.

For instance, `P_min=[-1, -1, -1]`, `P_max=[-1, -1, -1]`, and
`NDIVS=[10, 10, 50]` will grid the volumetric space between -1 and 1 in all
directions, with 10 cells in both the x and y-direction, and 50 cells in the
z-direction.

Even though the grid is first generated as a Cartesian grid, this can be
transformed into any other structured space through the argument
`spacetransform`, which is a function that takes any vector
and returns another vector of the same dimensions. For instance,
`P_min=[0.5, 0, 0], P_max=[1, 2*pi, 5], NDIVS=[10, 20, 30],
spacetransform=GeometricTools.cylindrical3D` will generate a cylindrical grid
discretizing the radial annulus from 0.5 to 1 with 10 cells, the polar angle
from 0 to 360deg with 20 cells, and the axial z-distance from 0 through 5 with
30 cells.

Any number of dimensions can be used, but make sure that `P_min`,
`P_max`, and `NDIVS` always have three dimensions and indicate the dimensions
that are "collapsed" with a 0 in `NDIVS`. Even though the grid is defined in
the Cartesian axes, the origin and orientation of the grid can be specified
with the `O` and `Oaxis` optional arguments.
For instance, `P_min=[0, 0, 1], P_max=[2, 3.5, 1], NDIVS=[10, 10, 0]`
will generate a 2D surface laying in the xy-plane at z=1.0, spanning from
(x,y)=(0,0) to (x,y)=(2,3.5). Use `O=[0, 0, -1]` to move the surface back to the
xy-plane at z=0. Use `Oaxis=[1 0 0; 0 0 -1; 0 1 0]` to re-orient the
surface to lay in the zx-plane. The same thing can be achieved with
`Oaxis=GeometricTools.rotation_matrix2(-90, 0, 0)` which generates the rotation
matrix corresponding to a -90deg rotation about the x-axis.

**NOTE:** The order of operation is (1) Cartesian grid generation, (2) space
transformation if any, and (3) translation and re-orientation to the given
origin and orientation.
"""
function computefluiddomain(P_min, P_max, NDIVS, args...;
                                    spacetransform=nothing,
                                    O=zeros(3),
                                    Oaxis=Float64[i==j for i in 1:3, j in 1:3],
                                    grid_optargs=[],
                                    verbose=true, v_lvl=0,
                                    debug=false, save_path=nothing,
                                    optargs...)

    if verbose
        println("\t"^(v_lvl)*"Generating fluid domain grid...")
    end

    # Generate Cartesian grid
    grid = gt.Grid(P_min, P_max, NDIVS; grid_optargs...)

    # Apply space transformation
    if spacetransform!=nothing
        gt.transform!(grid, spacetransform)
    end

    # Translate and rotate the field to the given origin and orientation
    gt.lintransform!(grid, Oaxis, O)

    if verbose
        println("\t"^(v_lvl+1)*"Number of nodes:\t$(grid.nnodes)")
    end


    if debug && save_path!=nothing

        str = save_path
        str *= gt.save(grid, "fdom-debug"; path=save_path)

        run(`paraview --data=$(str)`)
    end

    # Proceed to evaluate the fluid domain on this grid
    out = computefluiddomain(args..., [grid];
                                      save_path=save_path,
                                      verbose=verbose, v_lvl=v_lvl, optargs...)

    return out, grid
end

# Deprecated name
evaluate_fluiddomain_vtk = computefluiddomain
