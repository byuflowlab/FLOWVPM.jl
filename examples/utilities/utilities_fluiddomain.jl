#=##############################################################################
# DESCRIPTION
    Functions for generating, probing, and processing the fluid domain (velocity
    and vorticity fields) induced by particle field solutions.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Oct 2021
  * Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################


"""
    `evaluate_fluiddomain_vtk(pfield::FLOWVPM.ParticleField,
grids::Array{<:GeometricTools.AbstractGrid}; optargs...)`

    Evaluate the velocity and vorticity field induced by the particle field
`pfield` at all nodes in a set of grids `grids`. The fields are added as
solution fields `U` and `W` in each grid. The analytic Jacobian of the velocity
field can also be added with the optional argument `add_J=true`.


# OPTIONAL ARGUMENTS

## Processing options
* `add_J::Bool`             : Add the solution fields `J1`, `J2`, and `J3` to
                                each grid, where Ji[j]=dUi/dxj.
* `add_Uinf::Bool`          : It evaluates and adds a uniform freestream to the
                                `U` field.
* `scale_sigma::Real`       : It rescales the smoothing radius of each particle
                                by this factor before evaluating the particle
                                field.
* `f_Gamma::Real`           : Factor used to add the nodes as particles.
* `f_sigma::Real`           : Factor used to add the nodes as particles.

## Output options
* `save_path::String`       : If used, it will save the grids as VTK files under
                                this path.
* `file_pref::String`       : Prefix for VTK files.
* `grid_names::String`      : Name of each grid for VTK file. If not given, it
                                will generate their names automatically.
* `num::Int`                : If given, the name of the VTK file will be
                                `"\$(file_pref)\$(grid_names[i]).\$(num).vtk"`
* `verbose::Bool`           : Activate/deactivate verbose.


NOTE: The solution fields `U`, `W`, and Jacobian do not include the freestream
        field, but rather they only include the fields induced by the particles.
        To add the freestream to the `U`, use the optional argument
        `add_Uinf=true`.
"""
function evaluate_fluiddomain_vtk(pfield::vpm.ParticleField,
                                    grids::Array{<:gt.AbstractGrid};
                                    # PROCESSING OPTIONS
                                    add_J=false,
                                    add_Uinf=false,
                                    scale_sigma=1.0,
                                    f_Gamma=1e-2,
                                    f_sigma=0.5,
                                    remove_nodeparticles=true,
                                    # OUTPUT OPTIONS
                                    save_path=nothing,
                                    file_pref="",
                                    grid_names="automatic",
                                    num=nothing,
                                    verbose=true, v_lvl=0,
                                    )

    _grid_names = grid_names=="automatic" ? ("Grid$(gi)" for gi in 1:length(grids)) : grid_names
    str = ""

    t = @elapsed begin

        np = vpm.get_np(pfield)           # Original number of particles

        # Rescale smoothing radii
        for P in vpm.iterate(pfield; include_static=true)
            P.sigma[1] *= scale_sigma
        end

        # Estimate average sigma and minimum Gamma
        meansigma = 0
        minnormGamma = Inf
        for P in vpm.iterate(pfield; include_static=true)
            meansigma += P.sigma[1]

            normGamma = sqrt(P.Gamma[1]^2 + P.Gamma[2]^2 + P.Gamma[3]^2)
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

    end

    if verbose
        println("\t"^(v_lvl)*"Add nodes as particles:\t$(round(t, digits=1)) s")
        println("\t"^(v_lvl)*"Number of particles:\t$(vpm.get_np(pfield))")
    end

    # Evaluate particle field
    vpm._reset_particles(pfield)
    t = @elapsed pfield.UJ(pfield)

    if verbose
        println("\t"^(v_lvl)*"Evaluate UJ:\t\t$(round(t, digits=1)) s")
    end

    # Add freestream
    if add_Uinf
        Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)
        for P in vpm.iterate(pfield; start_i=np+1)
            P.U .+= Uinf
        end
    end

    t = @elapsed begin

        prev_np = np

        for (grid, gridname) in zip(grids, _grid_names)

            nnodes = grid.nnodes
            rng = prev_np+1:prev_np+nnodes

            # U = collect(vpm.get_U(pfield, i) for i in prev_np+1:prev_np+nnodes)
            # W = collect(vpm.get_W(pfield, i) for i in prev_np+1:prev_np+nnodes)

            # NOTE: This avoid memory allocation, but it could generate issues
            #           if the solution fields are accessed beyond being saved
            #           as VTK files
            U = (vpm.get_U(pfield, i) for i in rng)
            W = (vpm.get_W(pfield, i) for i in rng)

            gt.add_field(grid, "U", "vector", U, "node"; raise_warn=false)
            gt.add_field(grid, "W", "vector", W, "node"; raise_warn=false)

            if add_J
                particles = vpm.iterate(pfield;
                                              start_i=rng.start, end_i=rng.stop)
                for i in 1:3
                    Ji = (view(P.J, i, :) for P in particles)
                    gt.add_field(grid, "J$(i)", "vector", Ji, "node";
                                                               raise_warn=false)
                end
            end

            # Save fluid domain as VTK file
            if save_path != nothing
                str *= gt.save(grid, file_pref*gridname;
                                              path=save_path, num=num, time=num)
            end

            prev_np += nnodes
        end

    end

    if verbose
        println("\t"^(v_lvl)*"Save VTK:\t\t$(round(t, digits=1)) s")
    end

    # Remove node particles
    if remove_nodeparticles
        for pi in vpm.get_np(pfield):-1:np+1
            vpm.remove_particle(pfield, pi)
        end
    end

    # Restore original smoothing radii
    for P in vpm.iterate(pfield; include_static=true)
        P.sigma[1] /= scale_sigma
    end

    return str
end


"""
    `evaluate_fluiddomain_vtk(pfield::vpm.ParticleField, nums,
read_path::String, file_pref::String, grids; origin=nothing,
orientation=nothing, optargs...)`

    Evaluate the fluid domain induced by all the time steps `nums` found in
a particle field simulation saved under `read_path`. `file_pref` indicates the
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
"""
function evaluate_fluiddomain_vtk(pfield::vpm.ParticleField,
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
    T, M               = zeros(3), Float64[i==j for i in 1:3, j in 1:3]
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
            M .= transpose(Oaxis_prev)
            for i in 1:3
                for j in 1:3
                    M[i, j] = M[i,1]*Oaxis_new[1,j] + M[i,2]*Oaxis_new[2,j] + M[i,3]*Oaxis_new[3,j]
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
        evaluate_fluiddomain_vtk(pfield, grids;
                                    remove_nodeparticles=false,
                                     verbose=verbose, v_lvl=v_lvl+1, num=num,
                                     optargs...)
    end

    return nothing
end

"""
    `evaluate_fluiddomain_vtk(maxparticles::Int, args...; UJ::Function=vpm.UJ_fmm,
fmm::FLOWVPM.FMM=vpm.FMM(; p=4, ncrit=50, theta=0.4, phi=0.5), optargs...)`

    Just like the other `evaluate_fluiddomain_vtk(args...; optargs...)` methods
but automatically pre-allocating and initializing the particle field with the
given maximum number of particles, UJ evaluation method, and FMM configuration
(if FMM is used by UJ).
"""
function evaluate_fluiddomain_vtk(maxparticles::Int, args...;
                                    UJ=vpm.UJ_fmm,
                                    fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, phi=0.5),
                                    pfield_optargs=[],
                                    verbose=true, v_lvl=0,
                                    optargs...)

    if verbose
        println("\t"^(v_lvl)*"Pre-allocating memory for $(maxparticles) max particles...")
    end

    # Pre-allocate memory
    pfield = vpm.ParticleField(maxparticles; UJ=UJ, fmm=fmm, pfield_optargs...)

    return evaluate_fluiddomain_vtk(pfield, args...;
                                      verbose=verbose, v_lvl=v_lvl, optargs...)
end



"""
    `evaluate_fluiddomain_vtk(P_min, P_max, NDIVS, args...;
spacetransform=nothing, O=zeros(3), Oaxis=Float64[i==j for i in 1:3, j in 1:3],
optargs...)`

    Just like the other `evaluate_fluiddomain_vtk(args...; optargs...)` methods
but automatically generating a fluid domain grid. The grid is generated as a
Cartesian box with minimum and maximum corners `P_min` and `P_max` and `NDIVS`
cells.

For instance, `P_min=[-1, -1, -1], P_max=[-1, -1, -1], NDIVS=[10, 10, 50]`
will grid the volumetric space between -1 and 1 in all directions, with 10
cells in both the x and y-direction, and 50 cells in the z-direction.

Even though the grid is first generated as a Cartesian grid, this can be
transformed into any other structured space through the argument
`spacetransform` which is intended to receive a function that takes any vector
and returns another vector of the same dimension. For instance,
`P_min=[0.5, 0, 0], P_max=[1, 2*pi, 5], NDIVS=[10, 20, 30],
spacetransform=GeometricTools.cylindrical3D` will generate a cylindrical grid
discretizing the radial annulus from 0.5 to 1 with 10 cells, angle from 0 to
360deg with 20 cells, and axial z-distance from 0 through 5 with 30 cells.

Any number of dimensions can be given, just make sure that `P_min`,
`P_max`, and `NDIVS` always have three dimensions and indicate the dimensions
that are "collapsed" with a 0 in `NDIVS`. Even though the grid is defined in
the Cartesian axes, the origin and orientation of the grid can be specified
with the `O` and `Oaxis` optional arguments.
For instance, `P_min=[0, 0, 1], P_max=[2, 3.5, 1], NDIVS=[10, 10, 0]`
will generate a 2D surface laying in the xy-plane at z=1.0, spanning from
(x,y)=(0,0) to (x,y)=(2,3.5). Use `O=[0, 0, -1]` to move the surface back to the
xy-plane at z=0. Use `Oaxis=[1 0 0; 0 0 -1; 0 1 0]` to re-orient the
surface to lay in the zx-plane. The same thing can be achieved with
`Oaxis=gt.rotation_matrix2(-90, 0, 0)` which generates the rotation matrix
corresponding to a -90deg rotation about the x-axis.

NOTE: The order of operation goes from (1) Cartesian grid generation, (2) space
transformation if any, and (3) translation and re-orientation to the given
origin and orientation.
"""
function evaluate_fluiddomain_vtk(P_min, P_max, NDIVS, args...;
                                    spacetransform=nothing,
                                    O=zeros(3),
                                    Oaxis=Float64[i==j for i in 1:3, j in 1:3],
                                    grid_optargs=[],
                                    verbose=true, v_lvl=0,
                                    debug=false, save_path=nothing,
                                    optargs...) where {T}

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
    out = evaluate_fluiddomain_vtk(args..., [grid];
                                      save_path=save_path,
                                      verbose=verbose, v_lvl=v_lvl, optargs...)

    return out, grid
end
