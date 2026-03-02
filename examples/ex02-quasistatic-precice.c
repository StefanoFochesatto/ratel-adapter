/**
 * @file ex02-quasistatic-precice.c
 * @brief Quasistatic solid mechanics with preCICE coupling
 *
 * This example demonstrates coupling Ratel's quasistatic solver with preCICE.
 * It follows the same pattern as the deal.II adapter examples.
 */

const char help[] = "Ratel quasistatic solver with preCICE coupling\n\n\
This example couples Ratel with other solvers using preCICE.\n\
It expects a preCICE configuration file (precice-config.xml) in the\n\
working directory.\n";

#include <petsc.h>
#include <ratel-adapter/ratel-adapter.h>
#include <ratel.h>

int main(int argc, char **argv) {
  MPI_Comm comm;
  Ratel ratel;
  RatelAdapter adapter;
  TS ts;
  DM dm, dm_solution;
  Vec U, F;
  PetscReal dt = 0.01;
  PetscReal time = 0.0;
  PetscInt step = 0;
  PetscBool quiet = PETSC_FALSE;

  /* Adapter configuration */
  RatelAdapterParameters adapter_params;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  /* Parse command line options */
  PetscOptionsBegin(comm, NULL, "Ratel Quasistatic with preCICE", NULL);
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

  /* Second mesh name (from other participant) */
  strcpy(adapter_params.mesh_couple, "SolverTwo-Mesh");
  PetscCall(PetscOptionsString(
      "-precice_second_mesh", "Second preCICE mesh name (other participant)",
      NULL, adapter_params.mesh_couple, adapter_params.mesh_couple, 256, NULL));

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

  adapter_params.boundary_label_value = 6;
  PetscCall(PetscOptionsInt("-precice_boundary_value",
                            "DMLabel value for coupling", NULL,
                            adapter_params.boundary_label_value,
                            &adapter_params.boundary_label_value, NULL));

  adapter_params.dim = 3;
  PetscCall(PetscOptionsInt("-dim", "Spatial dimension", NULL,
                            adapter_params.dim, &adapter_params.dim, NULL));

  PetscOptionsEnd();

  /* Initialize Ratel */
  PetscCall(RatelInit(comm, &ratel));
  PetscCall(RatelLogStagePushDebug(ratel, "Ratel Setup"));
  PetscCall(RatelDMCreate(ratel, RATEL_SOLVER_QUASISTATIC, &dm));
  PetscCall(RatelGetSolutionMeshDM(ratel, &dm_solution));
  PetscCall(RatelCheckViewOptions(ratel, dm_solution));

  /* Create adapter */
  PetscCall(RatelAdapterCreate(&adapter_params, comm, ratel, &adapter));

  /* Setup time integration */
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(RatelTSSetup(ratel, ts));

  /* Create solution and force vectors */
  PetscCall(DMCreateGlobalVector(dm_solution, &U));
  PetscCall(PetscObjectSetName((PetscObject)U, "U"));
  PetscCall(VecDuplicate(U, &F));
  PetscCall(PetscObjectSetName((PetscObject)F, "F"));

  /* Initialize adapter with mesh and initial solution */
  PetscCall(RatelAdapterInitialize(adapter, dm_solution, U));

  /* Setup complete */
  if (!quiet) {
    PetscCall(
        PetscPrintf(comm, "----- Ratel Quasistatic with preCICE -----\\n\\n"));
    PetscCall(RatelView(ratel, PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(RatelLogStagePopDebug(ratel, "Ratel Setup"));

  PetscBool ongoing = PETSC_TRUE;
  while (ongoing) {
    /* Get timestep size from preCICE */
    PetscReal precice_dt;
    PetscCall(RatelAdapterGetMaxTimeStepSize(adapter, &precice_dt));
    dt = PetscMin(dt, precice_dt);

    /* Read coupling data (forces from fluid) */
    PetscCall(RatelAdapterReadData(adapter, 0.0, F));
    /* TODO: Apply F as Neumann BC to Ratel */

    /* Save checkpoint if required (implicit coupling) */
    PetscBool saved;
    PetscCall(
        RatelAdapterSaveCheckpointIfRequired(adapter, U, time, step, &saved));

    /* Solve one timestep */
    PetscCall(TSSetTimeStep(ts, dt));
    PetscCall(TSSetMaxSteps(ts, 1));
    PetscCall(TSSolve(ts, U));

    /* Write coupling data (displacements) and advance */
    PetscCall(RatelAdapterAdvance(adapter, U, dt, &precice_dt));

    /* Check if we need to reload checkpoint */
    PetscBool reloaded;
    PetscCall(RatelAdapterReloadCheckpointIfRequired(adapter, U, &time, &step,
                                                     &reloaded));
    if (reloaded) {
      /* Reset TS time */
      PetscCall(TSSetTime(ts, time));
      continue; /* Retry timestep */
    }

    /* Update time and step */
    time += dt;
    step++;
    dt = PetscMin(dt, precice_dt);

    /* Check if coupling continues */
    PetscCall(RatelAdapterIsCouplingOngoing(adapter, &ongoing));
  }

  /* Finalize */
  PetscCall(RatelAdapterDestroy(&adapter));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&F));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dm_solution));
  PetscCall(RatelDestroy(&ratel));

  return PetscFinalize();
}
