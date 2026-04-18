// @file
/// Ratel dynamic example with preCICE coupling using callbacks and I2Function wrapper
const char help[] = "Ratel - dynamic example with preCICE coupling using callbacks\n";
#include <petsc.h>
#include <ratel.h>
#include <stddef.h>
#include <ratel-adapter/ratel-adapter.h>
// Context for TS callbacks and I2Function wrapper



typedef struct {
  RatelAdapter adapter;
  Vec U;
  Vec V;
  Vec F;
  Vec checkpoint_V;
  PetscBool has_velocity_checkpoint;
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
  // Residual = M*U_tt + InternalForces - ExternalForces
  PetscCall(VecAXPY(Y, -1.0, my_ctx->F));
  PetscFunctionReturn(PETSC_SUCCESS);
}





// Pre-step callback for preCICE coupling
static PetscErrorCode PreStepAdapter(TS ts) {
  AdapterCtx *ctx;
  PetscReal time, dt, precice_dt;
  PetscInt step;
  PetscBool saved;
  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &ctx));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepNumber(ts, &step));
  // Get timestep size from preCICE
  PetscCall(RatelAdapterGetMaxTimeStepSize(ctx->adapter, &precice_dt));
  dt = PetscMin(dt, precice_dt);
  PetscCall(TSSetTimeStep(ts, dt));
  // Read coupling data (forces from fluid) at relative time 0.0
  // These are stored in ctx->F and used by ApplyTractionI2Function during the SNES solve
  PetscCall(RatelAdapterReadData(ctx->adapter, 0.0, ctx->F));
  // Save checkpoint if required (implicit coupling)
  PetscCall(RatelAdapterSaveCheckpointIfRequired(ctx->adapter, ctx->U, time, step, &saved));
  if (saved) {
    if (!ctx->has_velocity_checkpoint) {
      PetscCall(VecDuplicate(ctx->V, &ctx->checkpoint_V));
      ctx->has_velocity_checkpoint = PETSC_TRUE;
    }
    PetscCall(VecCopy(ctx->V, ctx->checkpoint_V));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}




// Post-step callback for preCICE coupling
static PetscErrorCode PostStepAdapter(TS ts) {
  AdapterCtx *ctx;
  PetscReal time, dt, precice_dt;
  PetscInt step;
  PetscBool reloaded, ongoing;
  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &ctx));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepNumber(ts, &step));
  // Write coupling data (displacements) and advance preCICE
  PetscCall(RatelAdapterAdvance(ctx->adapter, ctx->U, dt, &precice_dt));
  // Check if we need to reload checkpoint
  PetscCall(RatelAdapterReloadCheckpointIfRequired(ctx->adapter, ctx->U, &time, &step, &reloaded));
  if (reloaded) {
    if (ctx->has_velocity_checkpoint) {
      PetscCall(VecCopy(ctx->checkpoint_V, ctx->V));
    }
    // Reset TS time and step
    PetscCall(TSSetTime(ts, time));
    PetscCall(TSSetStepNumber(ts, step));
    // Update solution in TS
    PetscCall(TS2SetSolution(ts, ctx->U, ctx->V));
       // Tell TS that the step was rejected so it retries
       PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_ITERATING));
     }

     // Check if coupling continues
     PetscCall(RatelAdapterIsCouplingOngoing(ctx->adapter, &ongoing));
     if (!ongoing) {
       PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_USER));
     } else {
       // If coupling is ongoing, reset the converged reason to iterating so TSSolve continues
       PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_ITERATING));
     }

     PetscFunctionReturn(PETSC_SUCCESS);
   }




   int main(int argc, char **argv) {
     MPI_Comm    comm;
     Ratel       ratel;
     RatelAdapter adapter;
     RatelAdapterParameters adapter_params;
     TS          ts;
     DM          dm;
     Vec         U, V, F;
     PetscScalar final_time = 1.0;
     PetscBool   quiet      = PETSC_FALSE;
     AdapterCtx  *ctx;

     PetscCall(PetscInitialize(&argc, &argv, NULL, help));
     comm = PETSC_COMM_WORLD;

     // Read command line options
     PetscOptionsBegin(comm, NULL, "Ratel dynamic example", NULL);
     PetscCall(PetscOptionsBool("-quiet", "Suppress summary outputs", NULL, quiet, &quiet, NULL));

     // Adapter options - defaults match solverdummy config 
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
         "-precice_read_data", "Data to read from preCICE", NULL,
         adapter_params.read_data_name, adapter_params.read_data_name, 256, NULL));

     strcpy(adapter_params.write_data_name, "Data-One");
     PetscCall(PetscOptionsString("-precice_write_data",
                                  "Data to write to preCICE", NULL,
                                  adapter_params.write_data_name,
                                  adapter_params.write_data_name, 256, NULL));

     strcpy(adapter_params.boundary_label_name, "Face Sets");
     PetscCall(PetscOptionsString("-precice_boundary_label",
                                  "DMLabel for coupling boundary", NULL,
                                  adapter_params.boundary_label_name,
                                  adapter_params.boundary_label_name, 64, NULL));

     adapter_params.boundary_label_value = 1;
     PetscCall(PetscOptionsInt("-precice_boundary_value",
                               "DMLabel value for coupling", NULL,
                               adapter_params.boundary_label_value,
                               &adapter_params.boundary_label_value, NULL));

     adapter_params.dim = 3;
     PetscCall(PetscOptionsInt("-dim", "Spatial dimension", NULL,
                               adapter_params.dim, &adapter_params.dim, NULL));

     PetscOptionsEnd();

     // Initialize Ratel context and create DM
     PetscCall(RatelInit(comm, &ratel));
     PetscCall(RatelLogStagePushDebug(ratel, "Ratel Setup"));
     PetscCall(RatelDMCreate(ratel, RATEL_SOLVER_DYNAMIC, &dm));
     PetscCall(RatelCheckViewOptions(ratel, dm));

     // Create adapter 
     PetscCall(RatelAdapterCreate(&adapter_params, comm, ratel, &adapter));

     // Create TS
     PetscCall(TSCreate(comm, &ts));

     // Avoid stepping past the final loading condition
     PetscCall(TSSetMaxTime(ts, final_time));
     PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

     // Set options
     PetscCall(RatelTSSetup(ratel, ts));

     // View Ratel setup
     if (!quiet) {
       PetscCall(PetscPrintf(comm, "----- Ratel Dynamic Example with preCICE Callbacks -----\n\n"));
       PetscCall(RatelView(ratel, PETSC_VIEWER_STDOUT_WORLD));
     }

     // Solution vectors
     PetscCall(DMCreateGlobalVector(dm, &U));
     PetscCall(PetscObjectSetName((PetscObject)U, "U"));
     PetscCall(VecDuplicate(U, &V));
     PetscCall(PetscObjectSetName((PetscObject)V, "V"));
     PetscCall(VecDuplicate(U, &F));
     PetscCall(PetscObjectSetName((PetscObject)F, "F"));
     PetscCall(VecSet(F, 0.0));
     PetscCall(RatelLogStagePopDebug(ratel, "Ratel Setup"));

     // Initialize adapter with mesh and initial solution 
     PetscCall(RatelAdapterInitialize(adapter, dm, U));

     // Setup Context and Callbacks 
     PetscCall(PetscNew(&ctx));
     ctx->adapter = adapter;
     ctx->U = U;
     ctx->V = V;
     ctx->F = F;
     ctx->checkpoint_V = NULL;
     ctx->has_velocity_checkpoint = PETSC_FALSE;

     // Extract Ratel's native I2Function from the DM and wrap it
     PetscCall(DMTSGetI2Function(dm, &ctx->ratel_i2function, &ctx->ratel_i2ctx));
     PetscCall(DMTSSetI2Function(dm, ApplyTractionI2Function, ctx));

     PetscCall(TSSetApplicationContext(ts, ctx));
     PetscCall(TSSetPreStep(ts, PreStepAdapter));
     PetscCall(TSSetPostStep(ts, PostStepAdapter));

     // Solve
     PetscPreLoadBegin(PETSC_FALSE, "Ratel Solve");
     PetscCall(RatelLogStagePushDebug(ratel, "Ratel Solve"));
     PetscCall(RatelTSSetupInitialCondition(ratel, ts, U));
     PetscCall(TS2SetSolution(ts, U, V));
     PetscCall(PetscLogDefaultBegin());
     
     if (PetscPreLoadingOn) {
       SNES      snes;
       PetscReal rtol;
       PetscCall(TSGetSNES(ts, &snes));
       PetscCall(SNESGetTolerances(snes, NULL, &rtol, NULL, NULL, NULL));
       PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, .99, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
       PetscCall(TSStep(ts));
       PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
     } else {
       PetscCall(TSSolve(ts, NULL));
     }
     PetscCall(RatelLogStagePopDebug(ratel, "Ratel Solve"));
     PetscPreLoadEnd();

     PetscCall(RatelLogStagePushDebug(ratel, "Ratel Post-Process"));

     // Output solution
     PetscCall(RatelTSCheckpointFinalSolutionFromOptions(ratel, ts, U));
     PetscCall(VecViewFromOptions(U, NULL, "-view_final_solution"));

     // Post solver info
     {
       TSConvergedReason reason;
       PetscInt          num_steps;

       PetscCall(TSGetSolveTime(ts, &final_time));
       if (!quiet) PetscCall(PetscPrintf(comm, "Final time: %f\n", final_time));
       PetscCall(TSGetStepNumber(ts, &num_steps));
       if (!quiet) PetscCall(PetscPrintf(comm, "TS steps: %" PetscInt_FMT "\n", num_steps));
       PetscCall(TSGetConvergedReason(ts, &reason));
       if (!quiet || reason < TS_CONVERGED_ITERATING) PetscCall(PetscPrintf(comm, "TS converged reason: %s\n", TSConvergedReasons[reason]));
     }



     // Cleanup
     PetscCall(RatelAdapterDestroy(&adapter));
     if (ctx->has_velocity_checkpoint) {
       PetscCall(VecDestroy(&ctx->checkpoint_V));
     }
     PetscCall(PetscFree(ctx));
     PetscCall(TSDestroy(&ts));
     PetscCall(DMDestroy(&dm));
     PetscCall(VecDestroy(&U));
     PetscCall(VecDestroy(&V));
     PetscCall(VecDestroy(&F));
     PetscCall(RatelDestroy(&ratel));
     return PetscFinalize();
   }
