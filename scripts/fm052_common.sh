#!/bin/bash
# Shared, immutable p018_L1_ov3 production environment for task 052.
# Mature CPU/GPU continuations and the long GPU arm source this exact array.

FM052_CHECKPOINT_ROOT="${FM052_CHECKPOINT_ROOT:-/home/rander39/projects/FLOWPanel.jl/data/p018_L1_ov3}"
FM052_RESTART_NAME="${FM052_RESTART_NAME:-p018_L1_ov3}"
FM052_RESTART_STEP="${FM052_RESTART_STEP:-719}"

FM052_PRODUCTION_ENV=(
  RHPC_MESH=45_185_ct4
  RPM=5400
  BERNOULLI_ONLY=true
  RUN_KJ=false
  SAVE_VTK=true
  SPINUP_REVS=1.5
  SPINUP_START_FRACTION=0.4
  MAGVINF_START=0.0
  MAGVINF_PEAK=5.0
  MAGVINF_END=0.0
  FREESTREAM_RAMP_REVS=1.0
  FREESTREAM_HOLD_REVS=1.5
  FREESTREAM_WITHDRAW_REVS=4
  SETTLE_REVS=12
  CONVERGENCE_REVS=10
  CONVERGENCE_MEAN_TOL=0.005
  CONVERGENCE_PTP_TOL=0.02
  DAS_KINEMATIC_ARC=false
  RHPC_FORMULATION=velocity
  NT=36
  TRUNCATION_DEPTH_R=4
  RELAX_RLXF=0.3
  PARTICLE_SHEDDING=sigma_overlap
  DAS_ETA_KINEMATIC=1.0
  NWAKEROWS=4
  OVERLAP=3.0
  P_PER_STEP=14
  MERGE_R_FACTOR=0.0052
  FMM_BODY_EXPANSION_ORDER=17
  FMM_BODY_ACCEPTANCE=0.7
  FMM_BODY_LEAF_SIZE=109
  FMM_WAKE_EXPANSION_ORDER=16
  FMM_WAKE_ACCEPTANCE=0.6
  FMM_WAKE_LEAF_SIZE=38
  RHPC_SOLVER_S=true
  BLAS_NUM_THREADS=64
  BLAS_NUM_THREADS_MARCH=8
)

FM052_GPU_ENV=(
  VPM_ARRAYTYPE=cuarray
  FLOWPANEL_GPU_INFLUENCE=cuda
  RHPC_SOLVER_S_GPU=true
  RHPC_SOLVER_S_GPU_RESERVE_GIB=32
  RHPC_SOLVER_S_GPU_EMERGENCY_GIB=4
  RHPC_SOLVER_S_GPU_SAMPLE_INTERVAL=10
)

fm052_preflight_checkpoint() {
  local root="$FM052_CHECKPOINT_ROOT"
  local name="$FM052_RESTART_NAME"
  local step="$FM052_RESTART_STEP"
  local required
  for required in \
    "$root/${name}_body1.pvd" \
    "$root/${name}_body1/${name}_body1.${step}.vtu" \
    "$root/${name}_wake1_particles/${name}_wake1_particles.${step}.vtp" \
    "$root/${name}.metadata.toml"
  do
    test -f "$required" || {
      echo "checkpoint preflight missing file: $required" >&2
      return 1
    }
  done
  compgen -G "$root/${name}_wake1/${name}_wake1.*.${step}.vts" >/dev/null || {
    echo "checkpoint preflight missing wake grid: " \
      "$root/${name}_wake1/${name}_wake1.*.${step}.vts" >&2
    return 1
  }
  echo "checkpoint preflight passed: $root step $step"
}
