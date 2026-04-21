/**
 * @file adapter.c
 * @brief Core adapter implementation
 */

#include <ratel-adapter/petsc-debug.h>
#include <ratel-adapter/ratel-adapter.h>
#include <ratel-impl.h>
#include <stdio.h>
#include <string.h>

// Adapter Struct
struct _p_RatelAdapter {
  RatelAdapterParameters params;
  DM dm;
  PetscInt dim;

  // Interface data 
  int n_interface_vertices;
  int *precice_vertex_ids;
  PetscInt *petsc_indices;
  PetscReal *vertex_coords;

  // Data buffers for mapping between PETSc Vec and preCICE buffers
  PetscReal *write_buffer;
  PetscReal *read_buffer;
  PetscSection section; // local section for indexing into vectors

  // Checkpointing  for rollback and initial condition writing (will get rid of some of these tsrollback does this internally)
  Vec checkpoint_solution;
  Vec checkpoint_V;
  Vec old_solution;
  Vec delta_U;
  PetscReal checkpoint_time;
  PetscInt checkpoint_step;
  PetscBool has_checkpoint;

  // MPI and Ratel info for logging and preCICE calls
  PetscReal current_time;
  PetscReal time_step;
  PetscBool is_initialized;
  PetscBool is_finalized;
  MPI_Comm comm;
  Ratel ratel;
  PetscInt rank;
};

// Log Macro for precice calls 
#define RatelAdapterLog(adapter, level, ...)                                   \
  do {                                                                         \
    if ((adapter)->ratel) {                                                    \
      PetscPrintf((adapter)->comm, __VA_ARGS__);                               \
      PetscPrintf((adapter)->comm, "\n");                                      \
    }                                                                          \
  } while (0)

// Define log levels if not available 
#ifndef RATEL_LOG_LEVEL_INFO
#define RATEL_LOG_LEVEL_INFO 0
#endif
#ifndef RATEL_LOG_LEVEL_DEBUG
#define RATEL_LOG_LEVEL_DEBUG 1
#endif

// Forward declarations of internal functions 
PetscErrorCode RatelAdapterExtractBoundaryVertices(
    Ratel ratel, DM dm, const char *label_name, PetscInt label_value, PetscInt dim,
    PetscInt *n_vertices, PetscReal **vertex_coords, PetscInt **petsc_indices);

PetscErrorCode RatelAdapterExtractBoundaryDOFs(Ratel ratel, DM dm, const char *label_name, PetscInt label_value, PetscInt dim, 
                                                       PetscInt *n_dofs, PetscReal **dof_coords, 
                                                      PetscInt **petsc_indices, PetscInt **local_points);



PetscErrorCode RatelAdapterVecToPrecice(PetscInt n_vertices, PetscInt dim,
                                        PetscInt *indices, Vec vec,
                                        PetscReal *buffer);

PetscErrorCode RatelAdapterPreciceToVec(PetscInt n_vertices, PetscInt dim,
                                        PetscInt *indices,
                                        const PetscReal *buffer, Vec vec);







PetscErrorCode RatelAdapterCreate(RatelAdapterParameters *params, MPI_Comm comm,
                                  Ratel ratel, RatelAdapter *adapter) {
  PetscFunctionBeginUser;

  if (!params || !adapter) {
    SETERRQ(comm, PETSC_ERR_ARG_NULL, "Null argument");
  }

  RatelAdapter a;
  PetscCall(PetscNew(&a));

  // Copy parameters 
  a->params = *params;
  a->comm = comm;
  a->ratel = ratel;

  // Auto-detect if we should write deltas based on the data name
  if (!a->params.is_delta && strstr(a->params.write_data_name, "Delta")) {
    a->params.is_delta = PETSC_TRUE;
    RatelAdapterLog(a, RATEL_LOG_LEVEL_INFO, "Auto-detected delta writing for data '%s'", a->params.write_data_name);
  }

  a->is_initialized = PETSC_FALSE;
  a->is_finalized = PETSC_FALSE;
  a->has_checkpoint = PETSC_FALSE;

  // Get MPI info
  PetscCallMPI(MPI_Comm_rank(comm, &a->rank));
  PetscCallMPI(MPI_Comm_size(comm, (int *)&a->params.size));
  a->params.rank = a->rank;

  // Create preCICE participant using C API passing our mpi comm
  precicec_createParticipant_withCommunicator(
      a->params.participant_name, a->params.config_file, a->rank,
      a->params.size,
      &comm);

  RatelAdapterLog(a, RATEL_LOG_LEVEL_INFO, "Created preCICE participant '%s'", a->params.participant_name);

  *adapter = a;
  PetscFunctionReturn(0);
}









PetscErrorCode RatelAdapterDestroy(RatelAdapter *adapter) {
  PetscFunctionBeginUser;

  if (!adapter || !*adapter) {
    PetscFunctionReturn(0);
  }

  RatelAdapter a = *adapter;

  if (!a->is_finalized) {
    precicec_finalize();
    a->is_finalized = PETSC_TRUE;
  }

  // Free arrays 
  PetscCall(PetscFree(a->precice_vertex_ids));
  PetscCall(PetscFree(a->petsc_indices));
  PetscCall(PetscFree(a->vertex_coords));
  PetscCall(PetscFree(a->write_buffer));
  PetscCall(PetscFree(a->read_buffer));

  // Free vectors 
  if (a->checkpoint_solution) {
    PetscCall(VecDestroy(&a->checkpoint_solution));
  }
  if (a->checkpoint_V) {
    PetscCall(VecDestroy(&a->checkpoint_V));
  }
  if (a->old_solution) {
    PetscCall(VecDestroy(&a->old_solution));
  }
  if (a->delta_U) {
    PetscCall(VecDestroy(&a->delta_U));
  }

  PetscCall(PetscFree(a));
  *adapter = NULL;

  PetscFunctionReturn(0);
}







PetscErrorCode RatelAdapterInitialize(RatelAdapter adapter, DM dm,
                                      Vec solution) {
  PetscFunctionBeginUser;

  if (!adapter || !dm) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Null argument");
  }

  if (adapter->is_initialized) {
    SETERRQ(adapter->comm, PETSC_ERR_ARG_WRONGSTATE,
            "Adapter already initialized");
  }

  adapter->dm = dm;

  // Get dm dimension and check against configured dimension  
  PetscCall(DMGetDimension(dm, &adapter->dim));
  if (adapter->dim != adapter->params.dim) {
    SETERRQ2(adapter->comm, PETSC_ERR_ARG_INCOMP,
             "DM dimension %D does not match configured dimension %D",
             adapter->dim, adapter->params.dim);
  }

  // Get section for DOF layout
  PetscCall(DMGetLocalSection(dm, &adapter->section));


  // Extract boundary vertices
  PetscInt n_vertices_petsc;
  PetscReal *coords;
  PetscInt *indices;

  // Plans to extract dofs or quad points in the future but for now linear elements only
  // nearest-projection in preCICE requires mesh connectivity and the api only seems to support linear meshes. 
  PetscCall(RatelAdapterExtractBoundaryVertices(
      adapter->ratel, dm, adapter->params.boundary_label_name,
      adapter->params.boundary_label_value, adapter->dim, &n_vertices_petsc, &coords,
      &indices));

  adapter->n_interface_vertices = (int)n_vertices_petsc;
  adapter->vertex_coords = coords;
  adapter->petsc_indices = indices;

  RatelAdapterLog(adapter, RATEL_LOG_LEVEL_INFO,
                  "Found %D interface vertices on rank %d (after filtering clamped nodes)", n_vertices_petsc,
                  adapter->rank);

// PetscCall(PetscDebugExportInterfacePoints(adapter->comm, adapter->n_interface_vertices, adapter->dim, adapter->vertex_coords, "interface_points_debug"));
// PetscCall(PetscDebugPrintMapping(adapter->comm, adapter->n_interface_vertices, adapter->dim, adapter->petsc_indices, adapter->vertex_coords)); <- This is a very nice debug function 

  // Allocate vertex IDs and buffers
  if (adapter->n_interface_vertices > 0) {
    PetscCall(PetscMalloc1(adapter->n_interface_vertices, &adapter->precice_vertex_ids));
    PetscCall(PetscMalloc1(adapter->n_interface_vertices * adapter->dim, &adapter->write_buffer));
    PetscCall(PetscMalloc1(adapter->n_interface_vertices * adapter->dim, &adapter->read_buffer));

    // Register vertices with preCICE 
    precicec_setMeshVertices(adapter->params.mesh_name, adapter->n_interface_vertices,
                             adapter->vertex_coords,
                             adapter->precice_vertex_ids);

    // If we were going to set mesh connectivity it would be here. 

    RatelAdapterLog(adapter, RATEL_LOG_LEVEL_INFO,
                    "Registered %d vertices with preCICE", adapter->n_interface_vertices);
  }


  // Create tracking vectors as LOCAL vectors 
  PetscCall(DMCreateLocalVector(dm, &adapter->delta_U));
  PetscCall(DMCreateLocalVector(dm, &adapter->old_solution));
  PetscCall(VecCopy(adapter->delta_U, adapter->old_solution)); // Initialize to 0

  // Write initial data if required
  PetscBool requires_init = PETSC_FALSE;
  PetscCall(RatelAdapterRequiresInitialData(adapter, &requires_init));

  // Don't worry about this, Solid solver for now will always go second. more information: https://precice.org/couple-your-code-initializing-coupling-data
  if (requires_init && adapter->n_interface_vertices > 0) {
    // Note: for initial write, we assume ICs are already in a local vector 
    // or we just use zeros. For DisplacementDelta, initial is 0.
    if (adapter->params.is_delta) {
      PetscCall(VecSet(adapter->delta_U, 0.0));
      PetscCall(RatelAdapterVecToPrecice((PetscInt)adapter->n_interface_vertices,
                                         adapter->dim, adapter->petsc_indices,
                                         adapter->delta_U, adapter->write_buffer));
    } else {
       // Pull initial solution to local vector for extraction
       Vec local_sol;
       PetscCall(DMGetLocalVector(dm, &local_sol));
       PetscCall(DMGlobalToLocal(dm, solution, INSERT_VALUES, local_sol));
       PetscCall(RatelAdapterVecToPrecice((PetscInt)adapter->n_interface_vertices,
                                         adapter->dim, adapter->petsc_indices,
                                         local_sol, adapter->write_buffer));
       PetscCall(DMRestoreLocalVector(dm, &local_sol));
    }

    precicec_writeData(adapter->params.mesh_name,
                       adapter->params.write_data_name, (int)adapter->n_interface_vertices,
                       adapter->precice_vertex_ids, adapter->write_buffer);

    RatelAdapterLog(adapter, RATEL_LOG_LEVEL_INFO,
                    "Wrote initial data to preCICE");
  }

  // Initialize preCICE
  precicec_initialize();
  adapter->is_initialized = PETSC_TRUE;

  RatelAdapterLog(adapter, RATEL_LOG_LEVEL_INFO,
                  "preCICE initialized successfully");

  PetscFunctionReturn(0);
  }




  // At the moment, since we use linear elements and our interface is soley defined on mesh vertices
  // we can get away with returning global vec which we substract from the I2 function. 
  // When we do dofs/quadrature points, this will have to be refactored
  PetscErrorCode RatelAdapterReadData(RatelAdapter adapter,
                                    PetscReal relative_read_time,
                                    Vec boundary_data) {
  PetscFunctionBeginUser;

  if (!adapter || !adapter->is_initialized) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
            "Adapter not initialized");
  }

  if (adapter->n_interface_vertices == 0) {
    PetscFunctionReturn(0);
  }

  // Read data from preCICE into internal buffer 
  precicec_readData(adapter->params.mesh_name, adapter->params.read_data_name,
                    (int)adapter->n_interface_vertices, adapter->precice_vertex_ids,
                    relative_read_time, adapter->read_buffer);


  // Convert to PETSc Local Vec
  Vec local_data;
  PetscCall(DMGetLocalVector(adapter->dm, &local_data));
  PetscCall(VecZeroEntries(local_data));
  PetscCall(RatelAdapterPreciceToVec((PetscInt)adapter->n_interface_vertices,
                                     adapter->dim, adapter->petsc_indices,
                                     adapter->read_buffer, local_data));

  // Scatter to Global Vec (summing values on shared nodes)
  PetscCall(VecZeroEntries(boundary_data));
  PetscCall(DMLocalToGlobal(adapter->dm, local_data, ADD_VALUES, boundary_data));
  PetscCall(DMRestoreLocalVector(adapter->dm, &local_data));

  PetscCall(PetscDebugVerifyZeroes(adapter->dm, boundary_data, adapter->params.boundary_label_name, adapter->params.boundary_label_value));

  PetscFunctionReturn(0);
  }




  PetscErrorCode RatelAdapterAdvance(RatelAdapter adapter, Vec solution,
                                   PetscReal dt, PetscReal *precice_dt) {
  PetscFunctionBeginUser;

  if (!adapter || !adapter->is_initialized) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
            "Adapter not initialized");
  }

  // Write data if we have interface vertices 
  if (adapter->n_interface_vertices > 0) {

    // Pull current solution to a local vector
    Vec local_sol;
    PetscCall(DMGetLocalVector(adapter->dm, &local_sol));
    PetscCall(DMGlobalToLocal(adapter->dm, solution, INSERT_VALUES, local_sol));

    if (adapter->params.is_delta) {
      // Compute Delta U = local_sol - old_solution (both are local)
      PetscCall(VecWAXPY(adapter->delta_U, -1.0, adapter->old_solution, local_sol));
      PetscCall(RatelAdapterVecToPrecice((PetscInt)adapter->n_interface_vertices,
                                         adapter->dim, adapter->petsc_indices,
                                         adapter->delta_U, adapter->write_buffer));
      
    } else {
      PetscCall(RatelAdapterVecToPrecice((PetscInt)adapter->n_interface_vertices,
                                         adapter->dim, adapter->petsc_indices,
                                         local_sol, adapter->write_buffer));
    }


    precicec_writeData(adapter->params.mesh_name,
                       adapter->params.write_data_name,
                       (int)adapter->n_interface_vertices,
                       adapter->precice_vertex_ids, adapter->write_buffer);

    PetscCall(DMRestoreLocalVector(adapter->dm, &local_sol));
  }

  /* Advance preCICE */
  precicec_advance(dt);
  *precice_dt = precicec_getMaxTimeStepSize();

  adapter->current_time += dt;

  PetscFunctionReturn(0);
  }

  PetscErrorCode RatelAdapterSaveCheckpointIfRequired(RatelAdapter adapter,
                                                    Vec solution,
                                                    Vec velocity,
                                                    PetscReal time,
                                                    PetscInt step,
                                                    PetscBool *saved) {
  PetscFunctionBeginUser;

  *saved = PETSC_FALSE;

  if (!adapter || !adapter->is_initialized) {
    PetscFunctionReturn(0);
  }

  if (precicec_requiresWritingCheckpoint()) {
    /* Create checkpoint vectors (as LOCAL vectors) if needed */
    if (!adapter->checkpoint_solution) {
      PetscCall(DMCreateLocalVector(adapter->dm, &adapter->checkpoint_solution));
    }
    if (velocity && !adapter->checkpoint_V) {
      PetscCall(DMCreateLocalVector(adapter->dm, &adapter->checkpoint_V));
    }

    /* Save solution (scatter from global to checkpoint) */
    PetscCall(DMGlobalToLocal(adapter->dm, solution, INSERT_VALUES, adapter->checkpoint_solution));
    if (velocity) {
      PetscCall(DMGlobalToLocal(adapter->dm, velocity, INSERT_VALUES, adapter->checkpoint_V));
    }

    adapter->checkpoint_time = time;
    adapter->checkpoint_step = step;
    adapter->has_checkpoint = PETSC_TRUE;

    // Update old_solution for delta calculation for the upcoming window
    // This only happens when precicec_requiresWritingCheckpoint() is true,
    // which is at the start of every time window.
    if (adapter->params.is_delta) {
        PetscCall(VecCopy(adapter->checkpoint_solution, adapter->old_solution));
    }

    *saved = PETSC_TRUE;

    RatelAdapterLog(adapter, RATEL_LOG_LEVEL_DEBUG,
                    "Saved checkpoint at time %g, step %D", time, step);
  }

  PetscFunctionReturn(0);
  }

  PetscErrorCode RatelAdapterReloadCheckpointIfRequired(RatelAdapter adapter,
                                                      Vec solution,
                                                      Vec velocity,
                                                      PetscReal *time,
                                                      PetscInt *step,
                                                      PetscBool *reloaded) {
  PetscFunctionBeginUser;

  *reloaded = PETSC_FALSE;

  if (!adapter || !adapter->is_initialized) {
    PetscFunctionReturn(0);
  }

  if (precicec_requiresReadingCheckpoint()) {
    if (!adapter->has_checkpoint) {
      SETERRQ(adapter->comm, PETSC_ERR_ARG_WRONGSTATE,
              "No checkpoint available to reload");
    }

    /* Restore solution and velocity (scatter from local checkpoint to global) */
    PetscCall(DMLocalToGlobal(adapter->dm, adapter->checkpoint_solution, INSERT_VALUES, solution));
    if (velocity && adapter->checkpoint_V) {
      PetscCall(DMLocalToGlobal(adapter->dm, adapter->checkpoint_V, INSERT_VALUES, velocity));
    }

    *time = adapter->checkpoint_time;
    *step = adapter->checkpoint_step;

    *reloaded = PETSC_TRUE;

    RatelAdapterLog(adapter, RATEL_LOG_LEVEL_DEBUG,
                    "Restored checkpoint to time %g, step %D", *time, *step);
  } else if (precicec_isTimeWindowComplete()) {
    // If the window is complete, update old_solution for the NEXT window
    if (adapter->params.is_delta) {
        // Here we need the current solution in a local vector 
        Vec local_sol;
        PetscCall(DMGetLocalVector(adapter->dm, &local_sol));
        PetscCall(DMGlobalToLocal(adapter->dm, solution, INSERT_VALUES, local_sol));
        PetscCall(VecCopy(local_sol, adapter->old_solution));
        PetscCall(DMRestoreLocalVector(adapter->dm, &local_sol));
    }
  }

  PetscFunctionReturn(0);
  }




PetscErrorCode RatelAdapterIsCouplingOngoing(RatelAdapter adapter,
                                             PetscBool *ongoing) {
  PetscFunctionBeginUser;

  *ongoing = PETSC_FALSE;

  if (!adapter || !adapter->is_initialized) {
    PetscFunctionReturn(0);
  }

  *ongoing = precicec_isCouplingOngoing() ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionReturn(0);
}





PetscErrorCode RatelAdapterRequiresInitialData(RatelAdapter adapter,
                                               PetscBool *required) {
  PetscFunctionBeginUser;

  *required = PETSC_FALSE;

  if (!adapter) {
    PetscFunctionReturn(0);
  }

  *required = precicec_requiresInitialData() ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionReturn(0);
}

PetscErrorCode RatelAdapterGetMaxTimeStepSize(RatelAdapter adapter,
                                              PetscReal *dt) {
  PetscFunctionBeginUser;

  *dt = 0.0;

  if (!adapter || !adapter->is_initialized) {
    PetscFunctionReturn(0);
  }

  *dt = precicec_getMaxTimeStepSize();

  PetscFunctionReturn(0);
}

PetscErrorCode RatelAdapterGetNumInterfaceVertices(RatelAdapter adapter,
                                                   PetscInt *n_vertices) {
  PetscFunctionBeginUser;

  *n_vertices = 0;

  if (!adapter) {
    PetscFunctionReturn(0);
  }

  *n_vertices = adapter->n_interface_vertices;

  PetscFunctionReturn(0);
}

const char *RatelAdapterGetVersion(void) {
  static char version[32];
  if (version[0] == '\0') {
    sprintf(version, "%d.%d.%d", RATEL_ADAPTER_VERSION_MAJOR,
            RATEL_ADAPTER_VERSION_MINOR, RATEL_ADAPTER_VERSION_PATCH);
  }
  return version;
}
