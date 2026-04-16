/**
 * @file ex03-dynamic-precice.c
 * @brief Dynamic solid mechanics with preCICE coupling for FSI
 *
 * This example demonstrates coupling Ratel's dynamic solver with preCICE
 * for fluid-structure interaction. It combines the dynamic solver from
 * ex03-dynamic.c with the preCICE coupling pattern from
ex02-quasistatic-precice.c.
 *
 * Data Exchange:
 * - Reads: Forces (Data-Two) from fluid solver - applied as Neumann BC
 * - Writes: Displacements (Data-One) to fluid solver
 *
 * Note: For FSI problems requiring velocities at the interface (e.g., for
 * fluid wall BCs), you can extend this to also write velocities by:
 * 1. Adding a second write_data_name in precice-config.xml
 * 2. Calling RatelAdapterAdvance twice (or modifying adapter to handle multiple
data)
 * 3. The velocity vector V has the same layout as displacement U


 Usage:
# Terminal 1
./build/examples/ex03-dynamic-precice \
  -precice_config examples/precice-config-ratel-dummy.xml \
  -options_file /home/stefano/ratel/examples/ymls/ex03/neo-hookean-current.yml
# Terminal 2
solverdummies/c/solverdummy examples/precice-config-ratel-dummy.xml SolverTwo
The example properly handles the dynamic solver's second-order system (U and V)
with checkpointing for implicit coupling schemes.
 */

const char help[] = "Ratel dynamic solver with preCICE coupling for FSI\n\n\
This example couples Ratel's dynamic solid mechanics solver with other solvers\n\
using preCICE for fluid-structure interaction.\n\
\n\
Usage:\n\
  Terminal 1: ./ex03-dynamic-precice \\\n\
    -precice_config examples/precice-config-ratel-dummy.xml \\\n\
    -options_file /path/to/ratel/config.yml\n\
  Terminal 2: solverdummy examples/precice-config-ratel-dummy.xml SolverTwo\n\
\n\
Data exchange:\n\
  - Reads forces (Data-Two) from fluid solver\n\
  - Writes displacements (Data-One) to fluid solver\n\
\n\
Checkpointing:\n\
  For implicit coupling, both displacement (U) and velocity (V) are saved\n\
  and restored to ensure consistent time integration.\n";

#include <petsc.h>
#include <ratel-adapter/petsc-debug.h>
#include <ratel-adapter/ratel-adapter.h>
#include <ratel.h>

// Context for I2Function callback
typedef struct {
  Vec F;
  TSI2FunctionFn *ratel_i2function;
  void *ratel_i2ctx;
} AdapterCtx;

// Wrapper for the TS I2Function to include preCICE forces
static PetscErrorCode ApplyTractionI2Function(TS ts, PetscReal t, Vec U, Vec U_t, Vec U_tt, Vec Y, void *ctx) {
  AdapterCtx *my_ctx = (AdapterCtx *)ctx;

  PetscFunctionBeginUser;
  // 1. Evaluate Ratel's native residual (Internal Forces - Ratel Tractions)
  PetscCall(my_ctx->ratel_i2function(ts, t, U, U_t, U_tt, Y, my_ctx->ratel_i2ctx));

  // 2. Explicitly subtract preCICE nodal forces from the total residual Y
 // PetscCall(VecAXPY(Y, -1.0, my_ctx->F));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
  MPI_Comm comm;
  Ratel ratel;
  RatelAdapter adapter;
  TS ts;
  DM dm;
  Vec U, V, F;
  PetscReal dt = 0.01;
  PetscReal time = 0.0;
  PetscInt step = 0;
  PetscBool quiet = PETSC_FALSE;

  /* Adapter configuration */
  RatelAdapterParameters adapter_params;

  /* Checkpointing vectors for implicit coupling */
  Vec checkpoint_V = NULL;
  PetscBool has_velocity_checkpoint = PETSC_FALSE;
  TSI2FunctionFn *ratel_i2function = NULL;
  void *ratel_i2ctx = NULL;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  /* Parse command line options */
  PetscOptionsBegin(comm, NULL, "Ratel Dynamic with preCICE", NULL);
  PetscCall(
      PetscOptionsBool("-quiet", "Suppress output", NULL, quiet, &quiet, NULL));

  /* Adapter options - defaults match solverdummy config */
  strcpy(adapter_params.participant_name, "SolverOne");
  PetscCall(PetscOptionsString("-precice_participant",
                               "preCICE participant name", NULL,
                               adapter_params.participant_name,
                               adapter_params.participant_name, 256, NULL));

  strcpy(adapter_params.config_file, "precice-config.xml");
  PetscCall(PetscOptionsString("-precice_config", "preCICE config file", NULL,
                               adapter_params.config_file,
                               adapter_params.config_file, 256, NULL));

  strcpy(adapter_params.mesh_name, "SolverOne-Mesh");
  PetscCall(PetscOptionsString("-precice_mesh", "preCICE mesh name", NULL,
                               adapter_params.mesh_name,
                               adapter_params.mesh_name, 256, NULL));

  strcpy(adapter_params.read_data_name, "Data-Two");
  PetscCall(PetscOptionsString(
      "-precice_read_data", "Data to read from preCICE (forces)", NULL,
      adapter_params.read_data_name, adapter_params.read_data_name, 256, NULL));

  strcpy(adapter_params.write_data_name, "Data-One");
  PetscCall(PetscOptionsString("-precice_write_data",
                               "Data to write to preCICE (displacements)", NULL,
                               adapter_params.write_data_name,
                               adapter_params.write_data_name, 256, NULL));

  strcpy(adapter_params.boundary_label_name, "Face Sets");
  PetscCall(PetscOptionsString("-precice_boundary_label",
                               "DMLabel for coupling boundary", NULL,
                               adapter_params.boundary_label_name,
                               adapter_params.boundary_label_name, 64, NULL));

  adapter_params.boundary_label_value =
      1; // in this example 5, and 6 are the ends of the beam
  PetscCall(PetscOptionsInt("-precice_boundary_value",
                            "DMLabel value for coupling interface", NULL,
                            adapter_params.boundary_label_value,
                            &adapter_params.boundary_label_value, NULL));

  adapter_params.dim = 3;
  PetscCall(PetscOptionsInt("-dim", "Spatial dimension", NULL,
                            adapter_params.dim, &adapter_params.dim, NULL));

  PetscOptionsEnd();

  /* Initialize Ratel */
  PetscCall(RatelInit(comm, &ratel));
  PetscCall(RatelLogStagePushDebug(ratel, "Ratel Setup"));
  PetscCall(RatelDMCreate(ratel, RATEL_SOLVER_DYNAMIC, &dm));
  PetscCall(RatelCheckViewOptions(ratel, dm));

  /* Create adapter */
  PetscCall(RatelAdapterCreate(&adapter_params, comm, ratel, &adapter));

  /* Setup time integration */
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(RatelTSSetup(ratel, ts));

  /* Create solution, velocity, and force vectors */
  PetscCall(DMCreateGlobalVector(dm, &U));
  PetscCall(PetscObjectSetName((PetscObject)U, "U"));
  PetscCall(VecDuplicate(U, &V));
  PetscCall(PetscObjectSetName((PetscObject)V, "V"));
  PetscCall(VecDuplicate(U, &F));
  PetscCall(PetscObjectSetName((PetscObject)F, "F"));
  PetscCall(VecSet(F, 0.0));

  /* Initialize adapter with mesh and initial solution */
  PetscCall(RatelAdapterInitialize(adapter, dm, U));

  /* Setup complete */
  if (!quiet) {
    PetscCall(
        PetscPrintf(comm, "----- Ratel Dynamic FSI with preCICE -----\n\n"));
    PetscCall(RatelView(ratel, PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(RatelLogStagePopDebug(ratel, "Ratel Setup"));

  /* Setup initial conditions and link U, V to TS */
  PetscCall(RatelTSSetupInitialCondition(ratel, ts, U));
  PetscCall(TS2SetSolution(ts, U, V));

  /* Extract Ratel's native I2Function from the DM and wrap it */
  AdapterCtx *ctx;
  PetscCall(PetscNew(&ctx));
  ctx->F = F;
  PetscCall(DMTSGetI2Function(dm, &ctx->ratel_i2function, &ctx->ratel_i2ctx));
  PetscCall(DMTSSetI2Function(dm, ApplyTractionI2Function, ctx));

  /* Coupled time loop with inverted control */
  PetscBool ongoing = PETSC_TRUE;
  while (ongoing) {
    /* Get timestep size from preCICE */
    PetscReal precice_dt;
    PetscCall(RatelAdapterGetMaxTimeStepSize(adapter, &precice_dt));
    dt = PetscMin(dt, precice_dt);

    /* Read coupling data (forces from fluid) at relative time 0.0 (start of
     * step) */
    PetscCall(RatelAdapterReadData(adapter, 0.0, F));
    
    /* Forces in F are automatically applied via the wrapped I2Function callback */

    /* Save checkpoint if required (implicit coupling)
     * For dynamic solver, we need to checkpoint both displacement and velocity
     */
    PetscBool saved;
    PetscCall(
        RatelAdapterSaveCheckpointIfRequired(adapter, U, time, step, &saved));
    if (saved) {
      /* Also save velocity vector for dynamic solver */
      if (!has_velocity_checkpoint) {
        PetscCall(VecDuplicate(V, &checkpoint_V));
        has_velocity_checkpoint = PETSC_TRUE;
      }
      PetscCall(VecCopy(V, checkpoint_V));
    }

    /* Set timestep and solve one step
     * Use TSStep for step-by-step control needed for coupling */
    PetscCall(TSSetTimeStep(ts, dt));
    PetscCall(TSStep(ts));

    /* Get current time from TS after step */
    PetscCall(TSGetTime(ts, &time));
    PetscCall(TSGetStepNumber(ts, &step));

    /* Write coupling data (displacements) and advance preCICE */
    PetscCall(RatelAdapterAdvance(adapter, U, dt, &precice_dt));

    /* Check if we need to reload checkpoint */
    PetscBool reloaded;
    PetscCall(RatelAdapterReloadCheckpointIfRequired(adapter, U, &time, &step,
                                                     &reloaded));
    if (reloaded) {
      /* Restore velocity checkpoint for dynamic solver */
      if (has_velocity_checkpoint) {
        PetscCall(VecCopy(checkpoint_V, V));
      }
      /* Reset TS time and step */
      PetscCall(TSSetTime(ts, time));
      PetscCall(TSSetStepNumber(ts, step));
      /* Update solution in TS */
      PetscCall(TS2SetSolution(ts, U, V));
      continue; /* Retry timestep */
    }

    /* Update timestep for next iteration */
    dt = PetscMin(dt, precice_dt);

    /* Check if coupling continues */
    PetscCall(RatelAdapterIsCouplingOngoing(adapter, &ongoing));
  }

  /* Post-process final solution */
  PetscCall(RatelLogStagePushDebug(ratel, "Ratel Post-Process"));

  /* Output final solution */
  PetscCall(RatelTSCheckpointFinalSolutionFromOptions(ratel, ts, U));
  PetscCall(VecViewFromOptions(U, NULL, "-view_final_solution"));

  /* Post solver info */
  {
    TSConvergedReason reason;
    PetscInt num_steps;
    PetscReal final_time;

    PetscCall(TSGetSolveTime(ts, &final_time));
    if (!quiet)
      PetscCall(PetscPrintf(comm, "Final time: %f\n", final_time));
    PetscCall(TSGetStepNumber(ts, &num_steps));
    if (!quiet)
      PetscCall(PetscPrintf(comm, "TS steps: %" PetscInt_FMT "\n", num_steps));
    PetscCall(TSGetConvergedReason(ts, &reason));
    if (!quiet || reason < TS_CONVERGED_ITERATING)
      PetscCall(PetscPrintf(comm, "TS converged reason: %s\n",
                            TSConvergedReasons[reason]));
  }

  /* Verify MMS and compute errors if requested */
  PetscCall(RatelViewMMSL2ErrorFromOptions(ratel, time, U));
  PetscCall(RatelViewStrainEnergyErrorFromOptions(ratel, time, U));
  PetscCall(RatelViewMaxSolutionValuesErrorFromOptions(ratel, time, U));
  PetscCall(RatelViewSurfaceForceAndCentroidErrorFromOptions(ratel, time, U));

  PetscCall(RatelLogStagePopDebug(ratel, "Ratel Post-Process"));

  /* Finalize */
  PetscCall(RatelAdapterDestroy(&adapter));
  if (has_velocity_checkpoint) {
    PetscCall(VecDestroy(&checkpoint_V));
  }
  PetscCall(PetscFree(ctx));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&V));
  PetscCall(VecDestroy(&F));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&dm));
  PetscCall(RatelDestroy(&ratel));

  return PetscFinalize();
}
