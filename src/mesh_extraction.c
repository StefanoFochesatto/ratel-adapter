/**
 * @file mesh_extraction.c
 * @brief DMPlex topology traversal for coupling interface extraction
 */

#include <ratel-adapter/ratel-adapter.h>

/**
 * @brief Extract boundary vertices from DMPlex
 *
 * Traverses DMPlex topology to find vertices on the coupling interface.
 * Only includes locally-owned vertices, that affect mesh topology i.e. second order meshes with mid-edge nodes will not include mid-edge nodes in the coupling interface. 
 * 
 * 1. Get DMLabel by name passed through adapter params
 * 2. Duplicate label and call DMPlexLabelComplete to propagate label to vertices
 * 3. Get stratum IS for the completed label and intersect with all vertices IS
 * 4. Filter for locally-owned vertices using DMPlexGetPointGlobal
 * 5. Extract coordinates from DM
 * 6. Build petsc_indices using PetscSection offsets
 *
 * @param[in]  adapter       Adapter context (for parameters)
 * @param[in]  dm            DMPlex mesh
 * @param[out] n_vertices    Number of interface vertices (local to rank)
 * @param[out] vertex_coords Vertex coordinates [x0,y0,z0, ...] (allocated)
 * @param[out] petsc_indices PETSc indices per component (allocated)
 *
 * @return PetscErrorCode
 */



PetscErrorCode RatelAdapterExtractBoundaryVertices( DM dm, const char *label_name, PetscInt label_value, PetscInt dim, PetscInt *n_vertices, PetscReal **vertex_coords, PetscInt **petsc_indices) {
  
  PetscFunctionBeginUser;
  

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  // Get DMLabel 
  DMLabel label;
  PetscCall(DMGetLabel(dm, label_name, &label));
  if (!label) {
    *n_vertices = 0;
    *vertex_coords = NULL;
    *petsc_indices = NULL;
    PetscFunctionReturn(0);
  }

  // Duplicate label to avoid modifying the original DM's label during completion 
  DMLabel temp_label;
  PetscCall(DMLabelDuplicate(label, &temp_label));
  // Propagate labels from faces to vertices/edges 
  PetscCall(DMPlexLabelComplete(dm, temp_label));


  // Create index set for all vertices in the mesh (this will include ghosts)
  PetscInt vStart, vEnd;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  IS all_vertices_is;
  PetscCall(ISCreateStride(PETSC_COMM_SELF, vEnd - vStart, vStart, 1,
                           &all_vertices_is));

  // Create index set for all dmpoints with our coupling label
  IS label_is;
  PetscCall(DMLabelGetStratumIS(temp_label, label_value, &label_is));

  // Intersect to get unique vertices on the boundary 
  IS vertex_is = NULL;
  if (label_is) {
    PetscCall(ISIntersect(label_is, all_vertices_is, &vertex_is));
    PetscCall(ISDestroy(&label_is));
  }
  PetscCall(ISDestroy(&all_vertices_is));
  PetscCall(DMLabelDestroy(&temp_label));

  // Empty Label value case: return with zero vertices
  if (!vertex_is) {
    *n_vertices = 0;
    *vertex_coords = NULL;
    *petsc_indices = NULL;
    PetscFunctionReturn(0);
  }

  // Get vertex indices from our IS
  PetscInt n_candidates;
  const PetscInt *candidates;
  PetscCall(ISGetLocalSize(vertex_is, &n_candidates));
  PetscCall(ISGetIndices(vertex_is, &candidates));

  /* Now we have to make a decision about how we are handling parallelism. 
  There are several options described here: https://precice.org/couple-your-code-distributed-meshes.html
  I think I would like to keep a single mesh. The question really is if on each process are we going to 
  tell preCICE about ghost vertices or not. If we do we will need to adjust ghost vertex values in 
  the read/write functions. */

  // We will filter to locally-owned vertices for now. 
  // Allocate storage
  PetscInt *local_vertices;
  PetscCall(PetscMalloc1(n_candidates, &local_vertices));
  PetscInt local_count = 0;

  // Filter for locally-owned vertices (no ghosts)
  for (PetscInt i = 0; i < n_candidates; i++) {
    // Get global offset
    PetscInt p = candidates[i];
    PetscInt global_offset;
    PetscCall(DMPlexGetPointGlobal(dm, p, &global_offset, NULL));
    // If global_offset is negative, this is a ghost vertex and we skip it.
    if (global_offset >= 0) {
      local_vertices[local_count++] = p;
    }
  }

  PetscCall(ISRestoreIndices(vertex_is, &candidates));
  PetscCall(ISDestroy(&vertex_is));
  *n_vertices = local_count;

  // If no local vertices, return empty arrays
  if (local_count == 0) {
    PetscCall(PetscFree(local_vertices));
    *vertex_coords = NULL;
    *petsc_indices = NULL;
    PetscFunctionReturn(0);
  }



  // Now we will extract coordinates and build our petsc_indices array to map preCICE data to petsc data.
  // Allocate storage
  PetscCall(PetscMalloc1(local_count * dim, vertex_coords));
  
  // Get coordinate section and local vec 
  PetscSection coord_section;
  Vec coord_vec;
  PetscCall(DMGetCoordinateSection(dm, &coord_section));
  PetscCall(DMGetCoordinatesLocal(dm, &coord_vec));

  // get pointer to coordinate array for reading
  const PetscScalar *coords;
  PetscCall(VecGetArrayRead(coord_vec, &coords));

  // loop through local vertices and extract coordinates
  for (PetscInt v = 0; v < local_count; v++) {
    PetscInt vertex = local_vertices[v];
    PetscInt offset;
    PetscCall(PetscSectionGetOffset(coord_section, vertex, &offset));

    for (PetscInt d = 0; d < dim; d++) {
      (*vertex_coords)[v * dim + d] = coords[offset + d];
    }
  }
  // free our read-only pointer to the coordinate array
  PetscCall(VecRestoreArrayRead(coord_vec, &coords));

  // We have now local_count, which is the number of owned vertices, 
  // the local_vertices array which maps from 0..local_count to the index in the DM
  // and the vertex_coords array which has the coordinates for those vertices, in the same layout as precice
  // x_1, y_1, z_1, x_2, y_2, z_2, ...


  // Build PETSc indices for DOF mapping 
  PetscSection section;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscMalloc1(local_count * dim, petsc_indices));

  for (PetscInt v = 0; v < local_count; v++) {
    PetscInt vertex = local_vertices[v];
    PetscInt offset;
    PetscCall(PetscSectionGetOffset(section, vertex, &offset));

    for (PetscInt d = 0; d < dim; d++) {
      (*petsc_indices)[v * dim + d] = offset + d;
    }
  }

  PetscCall(PetscFree(local_vertices));
  PetscFunctionReturn(0);
}
