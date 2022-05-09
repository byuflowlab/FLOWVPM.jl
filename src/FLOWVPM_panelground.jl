struct PanelGround{TF}
    panels::PS.Panels{TF}
    control_particles::ParticleField
end

function PanelGround(panels::PS.Panels; formulation=formulation_default, viscous=Inviscid(), SFS=SFS_default, optargs...)
    # extract info
    centroids = panels.centroids
    dims, m, n, p = size(centroids)

    # create particle field
    max_particles = m*n*p
    cp_pfield = ParticleField(max_particles; formulation, viscous, SFS, optargs...)
    null_gamma = zeros(3)
    println("Sherlock! Building PanelGround cp_pfield:")
    ip=1
    for k in 1:p
        for j in 1:n
            for i in 1:m
                add_particle(cp_pfield, centroids[:,i,j,k], null_gamma, 1.0)
                # @show ip PS._index_mat_2_vec(panels, i, j, k)
                # ip+=1
                # note: (i,j,k) corresponds to
            end
        end
    end

    return PanelGround(panels, cp_pfield)
end

function update_velocities!(panel_ground::PanelGround, Vext=(X,t)->[0.0,0,0])
    for p in iterator(panel_ground.control_particles)
        p.U .+= PS.v_induced(panel_ground.panels, PS.Source(), PS.Quad(), p.X) + Vext(p.X,0)
    end
end

function get_tag(nt)
    return ".$(nt+1).vts"
end

"Updates the pfield.Uinf function to reflect the influence of panels of appropriate strengths."
function ground_effect!(pfield::ParticleField, panel_ground::PanelGround, dt; panel_collection=nothing, save_field=true, name="", savepath="", update_A = false, kwargs...)
    # add control point particles to pfield
    control_particles = panel_ground.control_particles
    np = control_particles.np
    pfield_np = pfield.np
    for ip in 1:np
        add_particle(pfield, get_X(control_particles,ip), get_Gamma(control_particles,ip), get_sigma(control_particles,ip))
    end

    # get wake induced velocities at control particles
    _reset_particles(pfield)
    pfield.UJ(pfield)
    velocities = panel_ground.panels.velocities
    dims, m, n, p = size(panel_ground.panels.centroids)
    iparticle = pfield_np+1
    for k in 1:p
        for j in 1:n
            for i in 1:m
                # iparticle = PS._index_mat_2_vec(panel_ground.panels, i, j, k)
                velocities[:,i,j,k] .= get_U(pfield, iparticle)
                iparticle += 1
            end
        end
    end

    # solve panels
    PS.solve!(panel_ground.panels, velocities) # solve for panel strengths
    # wake_velocities = deepcopy(velocities)
    velocities .= 0.0 # only look at the panel-induced velocities
    PS.update_velocities!(panel_ground.panels, PS.Source(), PS.Quad()) # add panel induced velocity to the velocity field

    # remove cp particles
    for ip in pfield_np+np:pfield_np+1
        remove_particle(pfield,ip)
    end

    # update_velocities!(panel_ground)

    # NOTE: Requires that the Uextra function is set as follows
    # kernel = PS.Source()
    # panel_shape = PS.Quad()
    # pfield.Uextra = (X) -> PS.v_induced(panel_ground.panels, kernel, panel_shape, X)

    # save particle field
    if save_field
        start_i = pfield_np+1
        end_i = pfield_np+np
        pfield.t += dt
        # save(pfield, name*"_ground_cps"; path=savepath, start_i=start_i, end_i=end_i)
        tag = get_tag(pfield.nt)
        if !isnothing(panel_collection)
            PS.save_vtk!(panel_collection, pfield.t, panel_ground.panels, joinpath(savepath,name*"_panels"); tag=tag)
        else
            PS.save_vtk(panel_ground.panels, joinpath(savepath,name*"_panels"); tag=tag)
        end
        pfield.t -= dt
    end

    return false
end
