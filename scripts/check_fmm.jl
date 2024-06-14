using FLOWVPM
const vpm = FLOWVPM
const ONE_OVER_4PI = 1/(4*pi)

function psi(target_x, source_x, source_gamma)
    dx = target_x - source_x
    dx_norm = sqrt(dx' * dx)
    return source_gamma ./ dx_norm * ONE_OVER_4PI
end

function dpsidx(target_x, source_x, source_gamma)
    dx = target_x - source_x
    dx_norm = sqrt(dx' * dx)
    x, y, z = dx
    jacobian = [
        -x*source_gamma[1] -x*source_gamma[2] -x*source_gamma[3];
        -y*source_gamma[1] -y*source_gamma[2] -y*source_gamma[3];
        -z*source_gamma[1] -z*source_gamma[2] -z*source_gamma[3];
    ] ./ dx_norm^3 * ONE_OVER_4PI
    return jacobian
end

function d2psidx2(target_x, source_x, source_gamma)
    dx = target_x - source_x
    dx_norm = sqrt(dx' * dx)
    x, y, z = dx
    hessian = zeros(3,3,3)
    d2dr2 = [
        2x^2-y^2-z^2 3x*y 3x*z;
        3x*y 2y^2-x^2-z^2 3y*z;
        3x*z 3y*z 2z^2-x^2-y^2
    ] / dx_norm^5
    hessian[:,:,1] = d2dr2 * source_gamma[1] * ONE_OVER_4PI
    hessian[:,:,2] = d2dr2 * source_gamma[2] * ONE_OVER_4PI
    hessian[:,:,3] = d2dr2 * source_gamma[3] * ONE_OVER_4PI
    return hessian
end

function u(target_x, source_x, source_gamma)
    dx = target_x  - source_x
    dx_norm = sqrt(dx' * dx)
    return 1/4/pi/dx_norm^3 * [
        -dx[2]*source_gamma[3] + dx[3]*source_gamma[2],
        -dx[3]*source_gamma[1] + dx[1]*source_gamma[3],
        -dx[1]*source_gamma[2] + dx[2]*source_gamma[1]
    ]
end

function duidxj_fd_fun(target_x, source_x, source_gamma; h=1e-8)
    duidx = (u(target_x+[h,0,0], source_x, source_gamma) - u(target_x,source_x,source_gamma))/h
    duidy = (u(target_x+[0,h,0], source_x, source_gamma) - u(target_x,source_x,source_gamma))/h
    duidz = (u(target_x+[0,0,h], source_x, source_gamma) - u(target_x,source_x,source_gamma))/h
    duidxj_res = hcat(duidx, duidy, duidz) .* 4 * pi
    return duidxj_res
end

function duidxj(target_x, source_x, source_gamma)
    dx = target_x - source_x
    x, y, z = dx
    xy = x*y
    yz = y*z
    xz = x*z
    gx, gy, gz = source_gamma
    dx_norm = sqrt(dx' * dx)
    duidxj = [
        (3xy*gz-3xz*gy) ((2y^2-x^2-z^2)*gz-3yz*gy) (3yz*gz-(2z^2-x^2-y^2)*gy);
        (3xz*gx-(2x^2-y^2-z^2)*gz) (3yz*gx-3xy*gz) ((2z^2-x^2-y^2)*gx-3xz*gz);
        ((2x^2-y^2-z^2)*gy-3xy*gx) (3xy*gy-(2y^2-x^2-z^2)*gx) (3xz*gy-3yz*gx)
    ]/dx_norm^5
    return 1/4/pi*duidxj
end

function stretching(target_x, source_x, target_gamma, source_gamma)
    dx = target_x - source_x
    x, y, z = dx
    xy = x*y
    yz = y*z
    xz = x*z
    gx, gy, gz = source_gamma
    dx_norm = sqrt(dx' * dx)
    duidxj = [
        (3xy*gz-3xz*gy) ((2y^2-x^2-z^2)*gz-3yz*gy) (3yz*gz-(2z^2-x^2-y^2)*gy);
        (3xz*gx-(2x^2-y^2-z^2)*gz) (3yz*gx-3xy*gz) ((2z^2-x^2-y^2)*gx-3xz*gz);
        ((2x^2-y^2-z^2)*gy-3xy*gx) (3xy*gy-(2y^2-x^2-z^2)*gx) (3xz*gy-3yz*gx)
    ]/dx_norm^5
    stretch = 1/4/pi*duidxj*target_gamma
    return stretch
end

bodies = [
    0.4 0.1
    0.1 -0.5
    -0.3 0.2
    1/8 1/8
    0.3 -0.4
    -0.1 -0.2
    0.08 0.5
]

maxparticles = 2
viscous = vpm.Inviscid()
formulation = vpm.ClassicVPM{vpm.FLOAT_TYPE}()
p, ncrit, theta = 10, 1, 1.0
transposed = false

pfield = vpm.ParticleField(maxparticles;
    formulation, viscous,
    np=0, nt=0, t=vpm.FLOAT_TYPE(0.0),
    kernel=vpm.kernel_default,
    UJ=vpm.UJ_fmm,
    Uinf=vpm.Uinf_default,
    SFS=vpm.SFS_default,
    integration=vpm.euler,
    # integration=vpm.rungekutta3,
    transposed=transposed,
    relaxation=vpm.relaxation_none,
    # relaxation=vpm.relaxation_default,
    fmm=vpm.FMM(;p=p, ncrit=ncrit, theta=theta, nonzero_sigma=true),
    M=zeros(vpm.FLOAT_TYPE, 4),
    toggle_rbf=false, toggle_sfs=false)

vpm.add_particle(pfield, bodies[1:3,1], bodies[5:7,1], bodies[4,1];
    vol=0, circulation=1, # as best I can tell, vol is used only in the viscous model, and circulation is never used in this package
    C=0, static=false, index=-1)

vpm.add_particle(pfield, bodies[1:3,2], bodies[5:7,2], bodies[4,2];
    vol=0, circulation=1, # as best I can tell, vol is used only in the viscous model, and circulation is never used in this package
    C=0, static=false, index=-1)

function get_stretching!(p::vpm.Particle, transposed)
    if transposed
        p.S[1] = p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3]
        p.S[2] = p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3]
        p.S[3] = p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3]
    else
        p.S[1] = p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3]
        p.S[2] = p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3]
        p.S[3] = p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3]
    end
end

function get_stretching!(pfield::vpm.ParticleField)
    for p in vpm.iterator(pfield)
        get_stretching!(p, pfield.transposed)
    end
end

# get induced velocity directly
vpm.UJ_direct(pfield)
get_stretching!(pfield)

# @show pfield.particles[1].U
# @show pfield.particles[2].U
# @show pfield.particles[1].J
# @show pfield.particles[2].J
# @show pfield.particles[1].S
# @show pfield.particles[2].S

# get induced velocity analytically
psis = zeros(3,2)
psis[:,1] = psi(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2])
psis[:,2] = psi(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1])
hessians = zeros(3,3,3,2)
hessians[:,:,:,1] = d2psidx2(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2])
hessians[:,:,:,2] = d2psidx2(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1])
us = zeros(3,2)
us[:,1] = u(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2])
us[:,2] = u(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1])
J1 = duidxj(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2])
J2 = duidxj(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1])
ss = zeros(3,2)
ss[:,1] = stretching(bodies[1:3,1], bodies[1:3,2], bodies[5:7,1], bodies[5:7,2])
ss[:,2] = stretching(bodies[1:3,2], bodies[1:3,1], bodies[5:7,2], bodies[5:7,1])
println("UJ_direct")
@show maximum(abs.(us[:,1]-pfield.particles[1].U))/maximum(abs.(us[:,1])) maximum(abs.(us[:,2]-pfield.particles[2].U))/maximum(abs.(us[:,2])) maximum(abs.(J1-pfield.particles[1].J))/maximum(abs.(J1)) maximum(abs.(J2-pfield.particles[2].J))/maximum(abs.(J2)) maximum(abs.(ss[:,1]-pfield.particles[1].S))/maximum(abs.(ss[:,1])) maximum(abs.(ss[:,2]-pfield.particles[2].S))/maximum(abs.(ss[:,2]))
# @show maximum(abs.(us[:,1]-pfield.particles[1].U)) maximum(abs.(us[:,2]-pfield.particles[2].U)) maximum(abs.(J1-pfield.particles[1].J)) maximum(abs.(J2-pfield.particles[2].J)) maximum(abs.(ss[:,1]-pfield.particles[1].S)) maximum(abs.(ss[:,2]-pfield.particles[2].S))

# get induced velocity using FMM
vpm._reset_particles(pfield)
vpm.UJ_fmm(pfield)
get_stretching!(pfield)
println("\n\nUJ_fmm")
@show maximum(abs.(us[:,1]-pfield.particles[1].U))/maximum(abs.(us[:,1])) maximum(abs.(us[:,2]-pfield.particles[2].U))/maximum(abs.(us[:,2])) maximum(abs.(J1-pfield.particles[1].J))/maximum(abs.(J1)) maximum(abs.(J2-pfield.particles[2].J))/maximum(abs.(J2)) maximum(abs.(ss[:,1]-pfield.particles[1].S))/maximum(abs.(ss[:,1])) maximum(abs.(ss[:,2]-pfield.particles[2].S))/maximum(abs.(ss[:,2]))
# @show maximum(abs.(us[:,1]-pfield.particles[1].U)) maximum(abs.(us[:,2]-pfield.particles[2].U)) maximum(abs.(J1-pfield.particles[1].J)) maximum(abs.(J2-pfield.particles[2].J)) maximum(abs.(ss[:,1]-pfield.particles[1].S)) maximum(abs.(ss[:,2]-pfield.particles[2].S))

# more complicated field