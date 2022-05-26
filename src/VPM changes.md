VPM changes:

1. Time integration is run through DifferentialEquations.jl. The call to run a simulation is the same, but rather than
running time stepping in a for loop, the code sets numerically solves the system of ODEs using DifferentialEquations.jl tools.
2. Particle and ParticleField structs had their functionality significantly extended to be compatible with DifferentialEquations.jl.
They are now treated like 1D arrays.
3. Additional solver options: default for running the old time integration (for backwards compatibility), fullstep (name subject to change)
for running a simulation with DifferentialEquations.jl, and adjoint for calculating sensitivities with the DifferentialEquations unsteady
adjoint calculations.
4. Minor: A new struct was defined for viscous diffusion to cast it in ODE form. This ends up turning the core spreading operation
into a single line in the ODE (as well as some optional callbacks).
5. Potentially breaking: Adjusted interface for setting up and running a simulation to take a few extra solver-related settings.
6. Pending, probably breaking: Separated solver settings from the ParticleField struct to allow differentiation with respect to numerical settings.
7. Implementation change: Monitors/runtime functions are run through callbacks. This improves compatibility with DE.jl.
8. Pending: Use the Julia-based FMM rather than calling the C++ one.
9. Assorted type asserts adjusted to be forwarddiff compatible.

Update:
After some discussion with Ryan, we came up with a new approach that's a lot more elegant and should avoid some of the issues I've seen.
The core idea is to have a function that generates the time integration function (which is then passed to the DE.jl solver). This avoids
issues 

