/**
 * @file ratel-adapter.h
 * @brief Ratel-preCICE coupling adapter public API
 *
 * This adapter enables the Ratel solid mechanics library to couple with other
 * solvers through the preCICE library. It follows the "Adapter Class" pattern
 * used in the deal.II-preCICE adapter.
 *
 * @ingroup RatelAdapter
 */

#ifndef RATEL_ADAPTER_H
#define RATEL_ADAPTER_H

#include <petsc.h>
#include <precice/preciceC.h>
#include <ratel.h>

/* ============================================================================
 * Version Information
 * ============================================================================ */

/** @brief Major version number */
#define RATEL_ADAPTER_VERSION_MAJOR 0
/** @brief Minor version number */
#define RATEL_ADAPTER_VERSION_MINOR 1
/** @brief Patch version number */
#define RATEL_ADAPTER_VERSION_PATCH 0

/**
 * @brief Get adapter version string
 * @return Version string (e.g., "0.1.0")
 * @ingroup RatelAdapter
 */
const char *RatelAdapterGetVersion(void);

/* ============================================================================
 * Configuration Structure
 * ============================================================================ */

/**
 * @brief Configuration parameters for the Ratel-preCICE adapter
 *
 * Mirrors deal.II's PreciceAdapterConfiguration. All strings must be
 * null-terminated.
 *
 * @ingroup RatelAdapter
 */
typedef struct
{
  /** @brief preCICE participant name (as in precice-config.xml) */
  char participant_name[256];
  /** @brief Path to preCICE configuration file */
  char config_file[256];
  /** @brief Name of coupling mesh in preCICE */
  char mesh_name[256];
  /** @brief Name of coupling mesh in preCICE */
  char mesh_couple[256];
  /** @brief Name of data to read from preCICE (e.g., "Force") */
  char read_data_name[256];
  /** @brief Name of data to write to preCICE (e.g., "Displacement") */
  char write_data_name[256];

  /**
   * @brief DMLabel name identifying coupling boundary
   *
   * Typical values: "Face Sets", "marker", "boundary"
   */
  char boundary_label_name[64];
  /** @brief Label value identifying coupling interface faces */
  PetscInt boundary_label_value;

  /** @brief Spatial dimension (2 or 3) */
  PetscInt dim;

  /** @brief MPI rank (set automatically by RatelAdapterCreate) */
  PetscMPIInt rank;
  /** @brief MPI communicator size (set automatically) */
  PetscMPIInt size;

} RatelAdapterParameters;

/* ============================================================================
 * Main Adapter Type
 * ============================================================================ */

/**
 * @brief Opaque adapter handle
 *
 * The internal structure is defined in the implementation files.
 * Users should treat this as an opaque pointer.
 *
 * @ingroup RatelAdapter
 */
typedef struct _p_RatelAdapter *RatelAdapter;

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

/**
 * @brief Create a new Ratel-preCICE adapter
 *
 * Initializes the adapter context and creates the preCICE participant.
 * Does not yet register the coupling mesh - call RatelAdapterInitialize()
 * for that.
 *
 * Equivalent to deal.II's Adapter constructor.
 *
 * @param[in] params Configuration parameters
 * @param[in] comm MPI communicator (typically PETSC_COMM_WORLD)
 * @param[in] ratel Ratel context for logging (can be NULL)
 * @param[out] adapter New adapter instance
 *
 * @return PetscErrorCode - 0 on success, non-zero on failure
 *
 * Example:
 * @code
 * RatelAdapterParameters params;
 * strcpy(params.participant_name, "Ratel-Solid");
 * strcpy(params.config_file, "precice-config.xml");
 * // ... set other fields ...
 *
 * RatelAdapter adapter;
 * RatelAdapterCreate(&params, PETSC_COMM_WORLD, ratel, &adapter);
 * @endcode
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterCreate(RatelAdapterParameters *params, MPI_Comm comm, Ratel ratel,
                                  RatelAdapter *adapter);

/**
 * @brief Destroy adapter and free all resources
 *
 * Finalizes preCICE and frees all allocated memory.
 *
 * @param[in,out] adapter Adapter to destroy (set to NULL on output)
 * @return PetscErrorCode
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterDestroy(RatelAdapter *adapter);

/* ============================================================================
 * Initialization
 * ============================================================================ */

/**
 * @brief Initialize preCICE and register coupling mesh
 *
 * This function must be called after the DMPlex mesh is fully set up
 * (including coordinates) but before the time loop begins.
 *
 * Steps performed:
 * 1. Extract boundary vertices from DMPlex using specified label
 * 2. Register vertices with preCICE
 * 3. Write initial data if required
 * 4. Call precicec_initialize()
 *
 * Equivalent to deal.II's Adapter::initialize().
 *
 * @param[in,out] adapter Adapter context
 * @param[in] dm DMPlex mesh with coordinates (reference only, not copied)
 * @param[in] solution Initial solution vector (for initial data write)
 *
 * @return PetscErrorCode
 *
 * @note The adapter stores a reference to dm, not a copy. The DM must
 *       remain valid for the adapter's lifetime.
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterInitialize(RatelAdapter adapter, DM dm, Vec solution);

/* ============================================================================
 * Data Exchange
 * ============================================================================ */

/**
 * @brief Read coupling data from preCICE
 *
 * Reads data (typically forces) from preCICE at the specified time
 * and stores it in the provided vector. Assumes consistent nodal
 * forces (no integration required).
 *
 * Equivalent to deal.II's Adapter::read_data().
 *
 * @param[in] adapter Adapter context
 * @param[in] relative_read_time Time interpolation factor:
 *            0.0 = start of step, 1.0 = end of step
 * @param[out] boundary_data Vector to receive data (size: n_vertices * dim)
 *
 * @return PetscErrorCode
 *
 * @pre RatelAdapterInitialize() must have been called
 * @pre boundary_data must be properly sized for the coupling interface
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterReadData(RatelAdapter adapter, PetscReal relative_read_time,
                                    Vec boundary_data);

/**
 * @brief Write coupling data and advance preCICE
 *
 * Extracts displacement data from the solution vector, writes it to
 * preCICE, and advances the coupling. Returns the timestep size
 * suggested by preCICE.
 *
 * Equivalent to deal.II's Adapter::advance().
 *
 * @param[in] adapter Adapter context
 * @param[in] solution Solution vector containing displacements
 * @param[in] dt Timestep size used by the solver
 * @param[out] precice_dt Timestep size required by preCICE
 *             (may be smaller than dt for subcycling)
 *
 * @return PetscErrorCode
 *
 * Example:
 * @code
 * PetscReal precice_dt;
 * RatelAdapterAdvance(adapter, solution, dt, &precice_dt);
 * dt = PetscMin(dt, precice_dt);  // Respect preCICE limit
 * @endcode
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterAdvance(RatelAdapter adapter, Vec solution, PetscReal dt,
                                   PetscReal *precice_dt);

/* ============================================================================
 * Checkpointing (Implicit Coupling)
 * ============================================================================ */

/**
 * @brief Save current state if checkpointing is required
 *
 * For implicit coupling schemes, this saves the solution state
 * at the beginning of a timestep so it can be restored if the
 * coupling does not converge.
 *
 * Equivalent to deal.II's Adapter::save_current_state_if_required().
 *
 * @param[in,out] adapter Adapter context
 * @param[in] solution Current solution vector to save
 * @param[in] time Current simulation time
 * @param[in] step Current timestep number
 * @param[out] saved PETSC_TRUE if a checkpoint was created
 *
 * @return PetscErrorCode
 *
 * Example:
 * @code
 * PetscBool saved;
 * RatelAdapterSaveCheckpointIfRequired(adapter, U, time, step, &saved);
 * @endcode
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterSaveCheckpointIfRequired(RatelAdapter adapter, Vec solution,
                                                    PetscReal time, PetscInt step,
                                                    PetscBool *saved);

/**
 * @brief Restore state from checkpoint if required
 *
 * If preCICE requires a checkpoint reload (implicit coupling did not
 * converge), this restores the solution and time to their saved values.
 *
 * Equivalent to deal.II's Adapter::reload_old_state_if_required().
 *
 * @param[in,out] adapter Adapter context
 * @param[out] solution Solution vector to restore into
 * @param[out] time Time value to restore
 * @param[out] step Timestep number to restore
 * @param[out] reloaded PETSC_TRUE if state was restored
 *
 * @return PetscErrorCode
 *
 * Example:
 * @code
 * PetscBool reloaded;
 * RatelAdapterReloadCheckpointIfRequired(adapter, U, &time, &step, &reloaded);
 * if (reloaded) {
 *     TSSetTime(ts, time);  // Reset PETSc TS time
 *     continue;             // Retry timestep
 * }
 * @endcode
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterReloadCheckpointIfRequired(RatelAdapter adapter, Vec solution,
                                                      PetscReal *time, PetscInt *step,
                                                      PetscBool *reloaded);

/* ============================================================================
 * Query Functions
 * ============================================================================ */

/**
 * @brief Check if coupling simulation is ongoing
 *
 * @param[in] adapter Adapter context
 * @param[out] ongoing PETSC_TRUE if coupling should continue
 * @return PetscErrorCode
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterIsCouplingOngoing(RatelAdapter adapter, PetscBool *ongoing);

/**
 * @brief Check if initial data write is required
 *
 * preCICE may require initial data to be written before the time loop.
 *
 * @param[in] adapter Adapter context
 * @param[out] required PETSC_TRUE if initial data should be written
 * @return PetscErrorCode
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterRequiresInitialData(RatelAdapter adapter, PetscBool *required);

/**
 * @brief Get maximum timestep size from preCICE
 *
 * @param[in] adapter Adapter context
 * @param[out] dt Maximum allowed timestep size
 * @return PetscErrorCode
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterGetMaxTimeStepSize(RatelAdapter adapter, PetscReal *dt);

/**
 * @brief Get number of coupling interface vertices on this rank
 *
 * @param[in] adapter Adapter context
 * @param[out] n_vertices Number of interface vertices (local to this rank)
 * @return PetscErrorCode
 *
 * @ingroup RatelAdapter
 */
PetscErrorCode RatelAdapterGetNumInterfaceVertices(RatelAdapter adapter, PetscInt *n_vertices);

#endif /* RATEL_ADAPTER_H */
