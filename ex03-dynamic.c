/// @file
/// Ratel dynamic example

const char help[] = "Ratel - dynamic example\n";

#include <petsc.h>
#include <ratel.h>
#include <stddef.h>

int main(int argc, char **argv) {
  MPI_Comm    comm;
  Ratel       ratel;
  TS          ts;
  DM          dm;
  Vec         U, V;
  PetscScalar final_time = 1.0;
  PetscBool   quiet      = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "Ratel dynamic example", NULL);
  PetscCall(PetscOptionsBool("-quiet", "Suppress summary outputs", NULL, quiet, &quiet, NULL));
  PetscOptionsEnd();

  // Initialize Ratel context and create DM
  PetscCall(RatelInit(comm, &ratel));
  PetscCall(RatelLogStagePushDebug(ratel, "Ratel Setup"));
  PetscCall(RatelDMCreate(ratel, RATEL_SOLVER_DYNAMIC, &dm));
  PetscCall(RatelCheckViewOptions(ratel, dm));

  // Create TS
  PetscCall(TSCreate(comm, &ts));

  // Avoid stepping past the final loading condition (because the solution might not be valid there)
  PetscCall(TSSetMaxTime(ts, final_time));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  // Set options
  PetscCall(RatelTSSetup(ratel, ts));

  // View Ratel setup
  if (!quiet) {
    // LCOV_EXCL_START
    PetscCall(PetscPrintf(comm, "----- Ratel Dynamic Example -----\n\n"));
    PetscCall(RatelView(ratel, PETSC_VIEWER_STDOUT_WORLD));
    // LCOV_EXCL_STOP
  }

  // Solution vector
  PetscCall(DMCreateGlobalVector(dm, &U));
  // Name vector so it isn't automatically named (via address) in output files
  PetscCall(PetscObjectSetName((PetscObject)U, "U"));
  PetscCall(VecDuplicate(U, &V));
  PetscCall(RatelLogStagePopDebug(ratel, "Ratel Setup"));

  // Solve
  PetscPreLoadBegin(PETSC_FALSE, "Ratel Solve");
  PetscCall(RatelLogStagePushDebug(ratel, "Ratel Solve"));
  PetscCall(RatelTSSetupInitialCondition(ratel, ts, U));
  PetscCall(TS2SetSolution(ts, U, V));
  PetscCall(PetscLogDefaultBegin());  // So we can use PetscLogStageGetPerfInfo without -log_view
  if (PetscPreLoadingOn) {
    // LCOV_EXCL_START
    SNES      snes;
    PetscReal rtol;
    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESGetTolerances(snes, NULL, &rtol, NULL, NULL, NULL));
    PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, .99, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(TSStep(ts));
    PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    // LCOV_EXCL_STOP
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

  // Verify MMS
  PetscCall(RatelViewMMSL2ErrorFromOptions(ratel, final_time, U));

  // Compute and verify strain energy
  PetscCall(RatelViewStrainEnergyErrorFromOptions(ratel, final_time, U));

  // Compute and verify max displacement
  PetscCall(RatelViewMaxSolutionValuesErrorFromOptions(ratel, final_time, U));

  // Compute and verify surface forces and centroids
  PetscCall(RatelViewSurfaceForceAndCentroidErrorFromOptions(ratel, final_time, U));

  // Compute and verify maximum model field values
  PetscCall(RatelViewMaxOutputFieldsErrorByNameFromOptions(ratel, final_time, U));

  // Compute and view model field values
  PetscCall(RatelViewOutputFieldsFromOptions(ratel, final_time, U));

  // Report total solve time
  {
    PetscLogStage      stage_id;
    PetscEventPerfInfo stage_perf_info;

    PetscCall(PetscLogStageGetId("Ratel Solve", &stage_id));
    PetscCall(PetscLogStageGetPerfInfo(stage_id, &stage_perf_info));
    if (!quiet) PetscCall(PetscPrintf(comm, "Time taken to compute solution (sec): %g\n", stage_perf_info.time));
  }

  PetscCall(RatelLogStagePopDebug(ratel, "Ratel Post-Process"));

  // Cleanup
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&V));
  PetscCall(RatelDestroy(&ratel));
  return PetscFinalize();
}

// ---------- Test Cases ----------

// Material model tests
//TESTARGS(name="neo-Hookean current")                                                                  -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/neo-hookean-current.yml
//TESTARGS(name="neo-Hookean current Enzyme AD",only="ad-enzyme")                                       -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/neo-hookean-current.yml -model elasticity-neo-hookean-current-ad-enzyme
//TESTARGS(name="Mooney-Rivlin current")                                                                -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/mooney-rivlin-current.yml
//TESTARGS(name="mixed neo-Hookean current")                                                            -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/mixed-neo-hookean-current-pcjacobi.yml
//TESTARGS(name="Linear poromechanics")                                                                 -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/poromechanics-linear-pcfieldsplit.yml
//TESTARGS(name="Poromechanics neo-Hookean current")                                                    -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/poromechanics-neo-hookean-current-pcfieldsplit.yml
//TESTARGS(name="Poromechanics neo-Hookean-Eipper current",only="serial,cgnsdiff")                      -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/poromechanics-neo-hookean-eipper-current-pcfieldsplit.yml -view_output_fields cgns:poromechanics-neo-hookean-eipper-current-pcfieldsplit_{ceed_resource}.cgns
//TESTARGS(name="Poromechanics neo-Hookean current Q1Q1 stable")                                        -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/poromechanics-neo-hookean-current-pstab-pcfieldsplit-lu.yml
//TESTARGS(name="Poromechanics neo-Hookean current isentropic ideal gas")                               -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/poromechanics-neo-hookean-eipper-gas-current-pcfieldsplit.yml
//TESTARGS(name="neo-Hookean current, scalar shock stabilization",only="serial,cgnsdiff",cgns_tol=1e-6) -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/neo-hookean-shock-scalar-current.yml -ts_max_time 1e-4 -view_output_fields cgns:neo-hookean-shock-scalar-current_{ceed_resource}.cgns -expected_strain_energy 9.628815577200e-05

// Other tests
//TESTARGS(name="schwarz-pendulum")                          -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/schwarz-pendulum.yml -ts_max_time 0.25 -ts_dt 0.05 -dm_plex_tps_extent 2,1,1 -dm_plex_tps_refine 0 -dm_plex_tps_layers 1 -strain_energy_atol 1e-12 -expected_strain_energy 3.462869242229e-08 -expected_strain_energy 2.187163610747e-08
//TESTARGS(name="schwarz-pendulum, enzyme",only="ad-enzyme") -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/schwarz-pendulum-enzyme.yml -ts_max_time 0.25 -ts_dt 0.05 -dm_plex_tps_extent 2,1,1 -dm_plex_tps_refine 0 -dm_plex_tps_layers 1 -strain_energy_atol 1e-12 -expected_strain_energy 3.462869242229e-08 -expected_strain_energy 2.187163610747e-08
//TESTARGS(name="Step-force uniaxial compression")           -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/step-force-uniaxial-compression.yml -ts_monitor_surface_force ascii:step-force-uniaxial-compression-face-force_{ceed_resource}.csv

//TESTARGS(name="neo-Hookean current, Eringen verification", csv_rtol = 1e-4, csv_ztol = 2e-6) -ceed {ceed_resource} -quiet -options_file examples/ymls/ex03/neo-hookean-current-eringen.yml -ts_max_time 0.1 -expected_strain_energy 1.012821471996e+02 -surface_force_faces 2 -ts_monitor_surface_force ascii:neo-hookean-current-eringen-face-force_{ceed_resource}.csv -ts_monitor_surface_force_interval 5
