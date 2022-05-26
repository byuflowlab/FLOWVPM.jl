Big todo items:

Finite differencing direct derivatve calculation -- works; runs really slowly (as expected)
ForwardDiff direct derivative calculation
ReverseDiff direct derivative calculation -- One function call at a time works; I'll need to run the full thing.

Adjoint calculation - finite differencing -- works; is so slow that I never let it actually finish calculations. iirc I left it running for like an hour before killing the process. It might be faster now, since I fixed some issues with tiny timesteps; however, I'll probably only use tiny simulations for benchmarking this
Adjoint calculation - ForwardDiff-based -- done
Adjoint calculation - ReverseDiff-based -- Either errors (dot access errors/uninitialized data/setindex errors) or returns all zeros.

Taylor's discrete adjoint - ForwardDiff VJP - Errors (type mismatch with structs) or returns all zeros (pure vector formulation).
Taylor's discrete adjoint - ReverseDiff VJP - Errors (dot access errors and NaN values)

FMM
VLM -- temporarily on hold

Performance comparisons, benchmarking -- in progress