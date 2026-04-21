/**
 * @file data_mapping.c
 * @brief Data format conversion between PETSc Vec and preCICE buffers
 */

#include <ratel-adapter/ratel-adapter.h>

/**
 * @brief Convert PETSc Vec to preCICE buffer format
 *
 * Extracts data at boundary vertices from the PETSc Vec and formats
 * it as [x0,y0,z0, x1,y1,z1, ...] for preCICE.
 *
 * @param[in]  adapter Adapter context
 * @param[in]  vec     PETSc vector (can be global or local)
 * @param[out] buffer  Output buffer (size: n_vertices * dim)
 *
 * @return PetscErrorCode
 */
PetscErrorCode RatelAdapterVecToPrecice(PetscInt n_vertices, PetscInt dim, PetscInt* indices,
                                        Vec vec, PetscReal* buffer) {
  PetscFunctionBeginUser;

  if (!indices || !vec || !buffer) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Null argument");
  }

  if (n_vertices == 0) {
    PetscFunctionReturn(0);
  }

  //Get array from Vec 
  const PetscScalar* array;
  PetscCall(VecGetArrayRead(vec, &array));

  // Copy data using petsc_indices
  for (PetscInt v = 0; v < n_vertices; v++) {
    for (PetscInt d = 0; d < dim; d++) {
      PetscInt idx = indices[v * dim + d];
      buffer[v * dim + d] = array[idx];
    }
  }

  PetscCall(VecRestoreArrayRead(vec, &array));

  PetscFunctionReturn(0);
}

/**
 * @brief Convert preCICE buffer to PETSc Vec
 *
 * Takes data in preCICE format [x0,y0,z0, x1,y1,z1, ...] and inserts
 * it into the PETSc Vec at boundary vertex locations.
 *
 * @param[in]  adapter Adapter context
 * @param[in]  buffer  Input buffer (size: n_vertices * dim)
 * @param[out] vec     PETSc vector to fill
 *
 * @return PetscErrorCode
 */
PetscErrorCode RatelAdapterPreciceToVec(PetscInt n_vertices, PetscInt dim, PetscInt* indices,
                                        const PetscReal* buffer, Vec vec) {
  PetscFunctionBeginUser;

  if (!indices || !buffer || !vec) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Null argument");
  }

  if (n_vertices == 0) {
    PetscFunctionReturn(0);
  }

  // Get array from Vec
  PetscScalar* array;
  PetscCall(VecGetArray(vec, &array));

  // Copy data using petsc_indices
  for (PetscInt v = 0; v < n_vertices; v++) {
    for (PetscInt d = 0; d < dim; d++) {
      PetscInt idx = indices[v * dim + d];
      array[idx] = buffer[v * dim + d];
    }
  }

  PetscCall(VecRestoreArray(vec, &array));

  PetscFunctionReturn(0);
}
