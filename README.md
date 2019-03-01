# FLOWVPM

Implementation of the three-dimensional viscous Vortex Particle Method written in Julia 1.0.

## Features
  * Single node CPU parallel processing (threads).
  * Single node GPU parallel processing.
  * Automatic differentiation
    * ForwardDiff: Tested on CPU. Capable of working on GPU if enough cache memory is available

## Folders
  * `src`         : Source code.
  * `examples`    : Validation cases and examples of how to use this code.

## Notes
  * Parallel processing is done through [multi-threading](https://docs.julialang.org/en/v1.0/manual/parallel-computing/#Multi-Threading-(Experimental)-1), which requires declaring the number of threads (or cores available) through the system environment variable `JULIA_NUM_THREADS` that Julia reads before launching. To do so, either do `export JULIA_NUM_THREADS=4` before launching Julia, or add this line (after changing the number of threads according to the number of cores aviable to your machine) to the system environment (i.e., `~/.bashrc` and/or `~/.bash_profile`). Verify the correct declaration of threads with `Threads.nthreads()` after launching Julia.

## Copyright
Copyrighted to Eduardo J. Alvarez. All rights reserved. No public license, use, or modification of this code without consent.
