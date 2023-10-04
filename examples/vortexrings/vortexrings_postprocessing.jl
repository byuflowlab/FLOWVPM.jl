#=##############################################################################
# DESCRIPTION
    Post processing of vortex ring simulations.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Jul 2021
  * Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################



function plot_dynamics(read_path;
                            filename="vortexring-dynamics1.csv",
                            cols_per_ring=5,
                            to_plot=[#  ((label_x, index_x, scale_x), (label_y, index_y, scale_y))
                                        (("Time (s)", "t", 1.0), ("Ring radius R (m)", 4, 1.0))
                                        (("Time (s)", "t", 1.0), ("Ring centroid Z (m)", 3, 1.0))
                                        (("Time (s)", "t", 1.0), ("Cross-sectional radius a (m)", 5, 1.0))
                                    ],
                            figname="vortexrings", figsize=[7,5]*7/9,
                            plot_vpm=true,
                            vpm_stl=".", clrs="rbcmgy"^10, vpm_lbl=" VPM", alt_vpm_lbl=nothing, vpm_alpha=0.10, vpm_optargs=[],
                            plot_ana=false, ana_args=[], ana_optargs=[], lbl_pref="automatic",
                            ana_stl="-", ana_lbl=" Analytic", ana_alpha=1.0, alt_ana_lbl=nothing, ana_clrs=nothing,
                            grid_optargs=(color="0.8", linestyle=":"),
                            sidelegend=false, _fig=nothing, _axs=nothing
                            )

    data = CSV.read(joinpath(read_path, filename), DataFrames.DataFrame)
    nrings = Int((size(data, 2) - 1) / cols_per_ring)
    ts = data[!, 1]

    if plot_ana
        ts_ana, Rs_ana, Zs_ana, as_ana = analytic_coaxialrings(ana_args...;
                                                                 ana_optargs...)
        data_ana = (zeros(length(ts_ana)), zeros(length(ts_ana)),
                                                         Zs_ana, Rs_ana, as_ana)
    end

    plot_grid = [size(to_plot, 1), size(to_plot, 2)]
    plot_gridt = [size(to_plot, 2), size(to_plot, 1)]

    fig = _fig == nothing ? plt.figure(figname, figsize=figsize.*plot_grid) : _fig
    axs = _axs == nothing ? fig.subplots(plot_gridt...) : _axs

    if length(to_plot)==1 && _axs==nothing; axs = [axs]; end;

    for (ploti, ((label_x, index_x, scale_x), (label_y, index_y, scale_y))) in enumerate(to_plot)

        for ri in 1:nrings

            xs =         index_x=="t" ? ts :
                 index_x isa Function ? index_x(data, ri, cols_per_ring) :
                                        data[!, 1 + cols_per_ring*(ri - 1) + index_x]
            ys =         index_y=="t" ? ts :
                 index_y isa Function ? index_y(data, ri, cols_per_ring) :
                                        data[!, 1 + cols_per_ring*(ri - 1) + index_y]

            ax = axs[ploti]

            if plot_ana
                xs_ana =     index_x=="t" ? ts_ana :
                     index_x isa Function ? index_x(data_ana, ri, cols_per_ring; ana=true, ts_ana=ts_ana) :
                                            data_ana[index_x][ri]
                ys_ana =     index_y=="t" ? ts_ana :
                     index_y isa Function ? index_y(data_ana, ri, cols_per_ring; ana=true, ts_ana=ts_ana) :
                                            data_ana[index_y][ri]

                ax.plot(scale_x*xs_ana, scale_y*ys_ana, ana_stl;
                            label=alt_ana_lbl!=nothing ? alt_ana_lbl[ri] : (lbl_pref=="automatic" ? "Ring $(ri)" : lbl_pref)*ana_lbl,
                            color=ana_clrs!=nothing ? "$(ana_clrs[ri])" : "$(clrs[ri])",
                            alpha=ana_alpha)
            end

            if plot_vpm
                ax.plot(scale_x*xs, scale_y*ys, vpm_stl;
                        label= alt_vpm_lbl!=nothing ? alt_vpm_lbl[ri] : (lbl_pref=="automatic" ? "Ring $(ri)" : lbl_pref)*vpm_lbl, color="$(clrs[ri])",
                        alpha=vpm_alpha, vpm_optargs...)
            end

            if ri == nrings
                ax.set_xlabel(label_x)
                ax.set_ylabel(label_y)
                ax.grid(true; grid_optargs...)
                if ploti == length(to_plot)
                    if sidelegend
                        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=false)
                    else
                        ax.legend(loc="best", frameon=true, fontsize=10)
                    end
                end
            end

        end

    end

    fig.tight_layout()

    return fig, axs
end

plot_dynamics1n2 = plot_dynamics

"""
    solve(derivative!, u0, tspan; dt, verbose, p)

Solve the ode using a simple forward euler step. (Eliminates DifferentialEquations as a dependency.)

# Arguments

- `derivative!::Function`- computes the derivative in place, as `derivative!(du, u, p, t)`, where `du` is the derivative, `u` the current state, `p` parameters, and `t` the current time
- `u0::Vector{Float64}`- vector of the initial states
- `tspan::Tuple{Float64,Float64}`- tuple containing the initial and final times
- `dt::Float64`- time step
- `verbose::Bool`- whether or not to use verbose output
- `p::NTuple{N,Any}`- tuple of parameters required by the `derivative!` function

# Returns

- `t::Range`- time vector
- `u::Vector{Vector{Float64}}`- vector of time solutions of each state; `u[i][j]` contains the value of the i-th state at the j-th time
"""
function solve(derivative!, u0, tspan; dt=1e-7, verbose=false, p=())
    verbose && (println("== SOLVE DIFFERENTIAL EQUATION ==\n"))
    t = range(tspan[1], tspan[2], step=dt)
    if t[end] != tspan[2]
        t = vcat(collect(t), tspan[end])
    end
    n_points = length(t)
    n_states = length(u0)
    u = [zeros(n_points) for _ in 1:n_states]
    
    # initial conditions
    for i_state in eachindex(u0)
        u[i_state] = u0[i]
    end

    # set up state and derivative containers
    this_u = deepcopy(u0)
    this_u_dot = similar(u0)

    # step through time
    verbose && (println("\tBegin Euler steps:"))
    for i_point in 2:n_points
        previous_time = t[i_point-1]
        this_dt = t[i_point] - previous_time
        verbose && (println("\t\tt=$(round(previous_time,digits=4)), u=$(this_u)"))
        derivative!(this_u_dot, this_u, p, previous_time)
        for i_state in 1:n_states
            this_u[i_state] = u[i_state][i_point] = u[i_state][i_point-1] + this_u_dot[i_state] * this_dt
        end
        i_point == n_points && verbose && (println("\t\tt=$(round(t[n_points],digits=4)), u=$(this_u)"))
    end

    verbose && (println("== DONE =="))
    # return result
    return t, u
end

"""
Solve dynamics of a system of coaxial inviscid rings using the analytic system
of ODEs shown in Borisov, A.V., Kilin, A.A. & Mamaev, I.S., 2013, "The dynamics
of vortex rings: Leapfrogging, choreographies and the stability problem".

In these equations, the interactions between rings ignore the thickness of each
ring. The thickness of each ring only affects the self-induced velocity. Use
`dynamica=true` to change the thickness of the ring to conserve volume.
The classic VPM with matches exactly the analytic solution with
`dynamica=false`.

`nrings` is the number of rings to simulate, `Gammas` is the circulation of each
ring, `Rs` is the initial radius of each ring, `Zs` is the initial centroid
position of each ring along the z-axis, `as` is the initial cross-sectional
radius of each ring, and `Deltas` is the shape factor for self-induced velocity
associated with vorticity distribution inside the ring core (Winckelmans' kernel
corresponds to `Delta=0`).
"""
function analytic_coaxialrings(nrings, Gammas, Rs, Zs, as, Deltas;
                                tend=20.0, dtmax=1e-3, dynamica=true, nu=nothing,
                                thickgaussian=false)

    # Initial conditions
    u0 = vcat([[Gammas[ri], Rs[ri], Zs[ri], as[ri], Deltas[ri]] for ri in 1:nrings]...)

    """
    u[5*(i-1) + 1] = Gamma of i-th ring (circulation)
    u[5*(i-1) + 2] = R of i-th ring (toroid radius)
    u[5*(i-1) + 3] = Z of i-th ring (toroid centroid)
    u[5*(i-1) + 4] = a of i-th ring (core radius)
    u[5*(i-1) + 5] = Delta of i-th ring (velocity factor)
    """
    function borisov2013!(du, u, p, t)
        nstates = 5                      # Number of states per ring
        nrings = Int(length(u)/nstates)  # Number of rings

        for ri in 1:nrings               # Iterate over target rings

            # Equivalent index of target
            ri_eq = nstates*(ri-1)

            # State variables of target
            Gammai = u[ri_eq+1]
            Ri = u[ri_eq+2]
            Zi = u[ri_eq+3]
            ai = u[ri_eq+4]
            Deltai = u[ri_eq+5]

            # Derivatives of target
            du[ri_eq+1] = 0
            du[ri_eq+2] = 0
            du[ri_eq+3] = Vz(Gammai, Ri, ai, Deltai)
            # du[ri_eq+4] = 0

            for rj in 1:nrings           # Iterate over source rings
                if ri != rj

                    # Equivalent index of source
                    rj_eq = nstates*(rj-1)

                    # State variables of target
                    Gammaj = u[rj_eq+1]
                    Rj = u[rj_eq+2]
                    Zj = u[rj_eq+3]

                    # Build dotRi
                    du[ri_eq+2] -= 1/Ri *  Gammaj * dGdZi(Ri, Zi, Rj, Zj)

                    # Build dotZi
                    du[ri_eq+3] += 1/Ri *  Gammaj * dGdRi(Ri, Zi, Rj, Zj)
                end
            end

            # Calculate dotai = -ai /(2*Ri) * dotRi
            if dynamica
                du[ri_eq+4] = -ai/(2*Ri) * du[ri_eq+2]
            else
                du[ri_eq+4] = 0
            end

            # Calculate dotai = nu / ai
            if nu != nothing
                du[ri_eq+4] += 2*nu/ai
            end

            # Calculate dotDeltai = -1.12*2*eps*doteps - 5.0*4*eps^3*doteps
            if thickgaussian
                epsi = ai/Ri
                depsidt = du[ri_eq+4]/Ri - ai/Ri^2*du[ri_eq+2]
                du[ri_eq+5] = -1.12*2*epsi*depsidt -5.0*4*epsi^3*depsidt
            else
                du[ri_eq+5] = 0
            end


        end

    end

    # Time span
    tspan = (0.0, tend)

    # prob = DifferentialEquations.ODEProblem(borisov2013!, u0, tspan)
    # sol = DifferentialEquations.solve(prob; dtmax=dtmax, verbose=true)
    t, u = solve(borisov2013!, u0, tspan; dt=dt, verbose=verbose)


    ts = sol.t
    solRs = [[u[5*(ri-1) + 2] for u in sol.u] for ri in 1:nrings]
    solZs = [[u[5*(ri-1) + 3] for u in sol.u] for ri in 1:nrings]
    solas = [[u[5*(ri-1) + 4] for u in sol.u] for ri in 1:nrings]

    return sol.t, solRs, solZs, solas, sol
end

K(k) = Elliptic.K(k^2)
E(k) = Elliptic.E(k^2)

k(z, r, zt, rt) = sqrt( 4*r*rt / ( (z-zt)^2 + (r+rt)^2 ) )

function G(z, r, zt, rt)
    this_k = k(z, r, zt, rt)
    return sqrt(r*rt) / (2*pi) * ( (2/this_k - this_k)*K(this_k) - 2/this_k*E(this_k) )
end

Gwrap(Ri, Zi, Rj, Zj) = G(Zi, Ri, Zj, Rj)

# NOTE: How trust worthy are this derivatives? Is this step good at all?
dGdZi(Ri, Zi, Rj, Zj; h=1e-5) = (Gwrap(Ri, Zi+h, Rj, Zj) - Gwrap(Ri, Zi, Rj, Zj)) / h
dGdRi(Ri, Zi, Rj, Zj; h=1e-5) = (Gwrap(Ri+h, Zi, Rj, Zj) - Gwrap(Ri, Zi, Rj, Zj)) / h

# NOTE: I'm assuming that Winckelmans' kernel leads to Delta - 1/2 = -0.5. Is that right?
# Hollow ring:            Delta - 0.5 = -0.5       (=> Delta = 0)
# Uniform distribution:   Delta - 0.5 = -0.25      (=> Delta = 0.25)
# Gaussian distribution:  Delta - 0.5 = -0.558     (=> Delta = -0.058)
Vz(Gamma, R, a, Delta) = Gamma/(4*pi*R) * (log(8*R/a) - 1/2 + Delta)
