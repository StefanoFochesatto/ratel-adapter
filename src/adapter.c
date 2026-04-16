/**
 * @file adapter.c
 * @brief Core adapter implementation
 */

#include <ratel-adapter/petsc-debug.h>
#include <ratel-adapter/ratel-adapter.h>
#include <stdio.h>
#include <string.h>

/* Internal adapter structure */
struct _p_RatelAdapter {
  RatelAdapterParameters params;
  DM dm;
  PetscInt dim;

  /* Interface data */
  PetscInt n_interface_vertices;
  PetscInt *precice_vertex_ids;
  PetscInt *petsc_indices;
  PetscReal *vertex_coords;
  PetscInt *petsc_face_indices;

  /* Data buffers */
  PetscReal *write_buffer;
  PetscReal *read_buffer;
  PetscSection section;

  /* Checkpointing */
  Vec checkpoint_solution;
  PetscReal checkpoint_time;
  PetscInt checkpoint_step;
  PetscBool has_checkpoint;

  /* State */
  PetscReal current_time;
  PetscReal time_step;
  PetscBool is_initialized;
  PetscBool is_finalized;
  MPI_Comm comm;
  Ratel ratel;
  PetscInt rank;
};

/* Internal logging macro - simplified for now */
#define RatelAdapterLog(adapter, level, ...)                                   \
  do {                                                                         \
    if ((adapter)->ratel) {                                                    \
      PetscPrintf((adapter)->comm, __VA_ARGS__);                               \
      PetscPrintf((adapter)->comm, "\n");                                      \
    }                                                                          \
  } while (0)

/* Define log levels if not available */
#ifndef RATEL_LOG_LEVEL_INFO
#define RATEL_LOG_LEVEL_INFO 0
#endif
#ifndef RATEL_LOG_LEVEL_DEBUG
#define RATEL_LOG_LEVEL_DEBUG 1
#endif

/* Forward declarations of internal functions */
PetscErrorCode RatelAdapterExtractBoundaryVertices(
    DM dm, const char *label_name, PetscInt label_value, PetscInt dim,
    PetscInt *n_vertices, PetscReal **vertex_coords, PetscInt **petsc_indices);

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

  /* Copy parameters */
  a->params = *params;
  a->comm = comm;
  a->ratel = ratel;

  a->is_initialized = PETSC_FALSE;
  a->is_finalized = PETSC_FALSE;
  a->has_checkpoint = PETSC_FALSE;

  /* Get MPI info */
  PetscCallMPI(MPI_Comm_rank(comm, &a->rank));
  PetscCallMPI(MPI_Comm_size(comm, (int *)&a->params.size));
  a->params.rank = a->rank;

  /* Create preCICE participant using C API with explicit communicator */
  precicec_createParticipant_withCommunicator(
      a->params.participant_name, a->params.config_file, a->rank,
      a->params.size,
      &comm); // Pass pointer to MPI_Comm

  RatelAdapterLog(a, RATEL_LOG_LEVEL_INFO, "Created preCICE participant '%s'",
                  a->params.participant_name);

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

  /* Free arrays */
  PetscCall(PetscFree(a->precice_vertex_ids));
  PetscCall(PetscFree(a->petsc_indices));
  PetscCall(PetscFree(a->vertex_coords));
  PetscCall(PetscFree(a->write_buffer));
  PetscCall(PetscFree(a->read_buffer));

  /* Free checkpoint */
  if (a->checkpoint_solution) {
    PetscCall(VecDestroy(&a->checkpoint_solution));
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
  PetscCall(PetscDebugPrintLabels(dm));
  PetscCall(PetscDebugViewLabelAsText(dm, adapter->params.boundary_label_name,
                                      "boundary_debug.txt"));

  // Get dm dimension and check against configured dimenstion
  
  PetscCall(DMGetDimension(dm, &adapter->dim));
  if (adapter->dim != adapter->params.dim) {
    SETERRQ2(adapter->comm, PETSC_ERR_ARG_INCOMP,
             "DM dimension %D does not match configured dimension %D",
             adapter->dim, adapter->params.dim);
  }

  /* Get section for DOF layout */
  //PetscDebugViewCGNS(dm, "mesh_debug.cgns");  
  PetscCall(DMGetLocalSection(dm, &adapter->section));


  /* Extract boundary vertices */
  PetscInt n_vertices;
  PetscReal *coords;
  PetscInt *indices;
  PetscCall(RatelAdapterExtractBoundaryVertices(
      dm, adapter->params.boundary_label_name,
      adapter->params.boundary_label_value, adapter->dim, &n_vertices, &coords,
      &indices));

  adapter->n_interface_vertices = n_vertices;
  adapter->vertex_coords = coords;
  adapter->petsc_indices = indices;

  RatelAdapterLog(adapter, RATEL_LOG_LEVEL_INFO,
                  "Found %D interface vertices on rank %d", n_vertices,
                  adapter->rank);

  /* Allocate vertex IDs and buffers */
  if (n_vertices > 0) {
    PetscCall(PetscMalloc1(n_vertices, &adapter->precice_vertex_ids));
    PetscCall(PetscMalloc1(n_vertices * adapter->dim, &adapter->write_buffer));
    PetscCall(PetscMalloc1(n_vertices * adapter->dim, &adapter->read_buffer));

    /* Register vertices with preCICE */
    precicec_setMeshVertices(adapter->params.mesh_name, n_vertices,
                             adapter->vertex_coords,
                             adapter->precice_vertex_ids);

    RatelAdapterLog(adapter, RATEL_LOG_LEVEL_INFO,
                    "Registered %D vertices with preCICE", n_vertices);
  }

  /* Write initial data if required */
  PetscBool requires_init = PETSC_FALSE;
  PetscCall(RatelAdapterRequiresInitialData(adapter, &requires_init));

  if (requires_init && n_vertices > 0) {
    /* Extract displacements from solution */
    PetscCall(RatelAdapterVecToPrecice(adapter->n_interface_vertices,
                                       adapter->dim, adapter->petsc_indices,
                                       solution, adapter->write_buffer));

    precicec_writeData(adapter->params.mesh_name,
                       adapter->params.write_data_name, n_vertices,
                       adapter->precice_vertex_ids, adapter->write_buffer);

    RatelAdapterLog(adapter, RATEL_LOG_LEVEL_INFO,
                    "Wrote initial data to preCICE");
  }

  /* Initialize preCICE */
  precicec_initialize();

  adapter->is_initialized = PETSC_TRUE;

  RatelAdapterLog(adapter, RATEL_LOG_LEVEL_INFO,
                  "preCICE initialized successfully");

  PetscFunctionReturn(0);
}

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

  /* Read data from preCICE */
  precicec_readData(adapter->params.mesh_name, adapter->params.read_data_name,
                    adapter->n_interface_vertices, adapter->precice_vertex_ids,
                    relative_read_time, adapter->read_buffer);

  /* DEBUG: Print read buffer */
  PetscCall(PetscDebugPrintPreciceBuffer(adapter->comm, "Read Buffer (from preCICE)",
                                         adapter->n_interface_vertices, adapter->dim,
                                         adapter->read_buffer));

  /* Convert to PETSc Vec */
  PetscCall(RatelAdapterPreciceToVec(adapter->n_interface_vertices,
                                     adapter->dim, adapter->petsc_indices,
                                     adapter->read_buffer, boundary_data));

  PetscFunctionReturn(0);
}

PetscErrorCode RatelAdapterAdvance(RatelAdapter adapter, Vec solution,
                                   PetscReal dt, PetscReal *precice_dt) {
  PetscFunctionBeginUser;

  if (!adapter || !adapter->is_initialized) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
            "Adapter not initialized");
  }

  /* Write data if we have interface vertices */
  if (adapter->n_interface_vertices > 0) {
    PetscCall(RatelAdapterVecToPrecice(adapter->n_interface_vertices,
                                       adapter->dim, adapter->petsc_indices,
                                       solution, adapter->write_buffer));

    /* DEBUG: Print write buffer */
    PetscCall(PetscDebugPrintPreciceBuffer(adapter->comm, "Write Buffer (to preCICE)",
                                           adapter->n_interface_vertices, adapter->dim,
                                           adapter->write_buffer));

    precicec_writeData(adapter->params.mesh_name,
                       adapter->params.write_data_name,
                       adapter->n_interface_vertices,
                       adapter->precice_vertex_ids, adapter->write_buffer);
  }

  /* Advance preCICE */
  precicec_advance(dt);
  *precice_dt = precicec_getMaxTimeStepSize();

  adapter->current_time += dt;

  PetscFunctionReturn(0);
}

PetscErrorCode RatelAdapterSaveCheckpointIfRequired(RatelAdapter adapter,
                                                    Vec solution,
                                                    PetscReal time,
                                                    PetscInt step,
                                                    PetscBool *saved) {
  PetscFunctionBeginUser;

  *saved = PETSC_FALSE;

  if (!adapter || !adapter->is_initialized) {
    PetscFunctionReturn(0);
  }

  if (precicec_requiresWritingCheckpoint()) {
    /* Create checkpoint vector if needed */
    if (!adapter->checkpoint_solution) {
      PetscCall(VecDuplicate(solution, &adapter->checkpoint_solution));
    }

    /* Save solution */
    PetscCall(VecCopy(solution, adapter->checkpoint_solution));
    adapter->checkpoint_time = time;
    adapter->checkpoint_step = step;
    adapter->has_checkpoint = PETSC_TRUE;

    *saved = PETSC_TRUE;

    RatelAdapterLog(adapter, RATEL_LOG_LEVEL_DEBUG,
                    "Saved checkpoint at time %g, step %D", time, step);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode RatelAdapterReloadCheckpointIfRequired(RatelAdapter adapter,
                                                      Vec solution,
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

    /* Restore solution */
    PetscCall(VecCopy(adapter->checkpoint_solution, solution));
    *time = adapter->checkpoint_time;
    *step = adapter->checkpoint_step;

    *reloaded = PETSC_TRUE;

    RatelAdapterLog(adapter, RATEL_LOG_LEVEL_DEBUG,
                    "Restored checkpoint to time %g, step %D", *time, *step);
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
