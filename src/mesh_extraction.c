/**
 * @file mesh_extraction.c
 * @brief DMPlex topology traversal for coupling interface extraction
 */

#include <ratel-adapter/ratel-adapter.h>
#include <ratel-impl.h>

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



PetscErrorCode RatelAdapterExtractBoundaryVertices(Ratel ratel, DM dm, const char *label_name, PetscInt label_value, PetscInt dim, PetscInt *n_vertices, PetscReal **vertex_coords, PetscInt **petsc_indices) {
  
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
      // Check if this point is clamped in Ratel (exclude it from coupling)
      PetscBool is_clamped = PETSC_FALSE;
      if (ratel) {
        for (PetscInt f = 0; f < RATEL_MAX_FIELDS; f++) {
          for (PetscInt j = 0; j < ratel->bc_clamp_count[f]; j++) {
            PetscInt val;
            PetscCall(DMLabelGetValue(label, p, &val));
            if (val == ratel->bc_clamp_faces[f][j]) {
              is_clamped = PETSC_TRUE;
              break;
            }
          }
          if (is_clamped) break;
        }
      }
      
      if (!is_clamped) {
        local_vertices[local_count++] = p;
      }
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

// WIP function
PetscErrorCode RatelAdapterExtractBoundaryDOFs(Ratel ratel, DM dm, const char *label_name, PetscInt label_value, PetscInt dim, 
                                                       PetscInt *n_dofs, PetscReal **dof_coords, 
                                                      PetscInt **petsc_indices, PetscInt **local_points) {

  PetscFunctionBeginUser;
  
  // Get Label
  DMLabel label, temp_label;
  PetscCall(DMGetLabel(dm, label_name, &label));
  if (!label) {
    *n_dofs = 0;
    *dof_coords = NULL;
    *petsc_indices = NULL;
    PetscFunctionReturn(0);
  }

  // Duplicate and complete to ensure labels propagate from faces to edges/vertices
  PetscCall(DMLabelDuplicate(label, &temp_label));
  PetscCall(DMPlexLabelComplete(dm, temp_label));

  // Identify all coupling dofs.  (Corner, Midside, Face-center, etc.)
  PetscInt     pStart, pEnd;
  PetscSection section, global_section;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &global_section));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));

  // Allocate temporary storage for the maximum possible number of dm points
  PetscInt *local_dm_points;
  PetscCall(PetscMalloc1(pEnd - pStart, &local_dm_points));
  PetscInt dmpoint_local_count = 0;
  *n_dofs = 0;

  for (PetscInt p = pStart; p < pEnd; p++) {
    PetscInt dof, val, global_offset;

    // Filter dm point for coupling label
    PetscCall(DMLabelGetValue(temp_label, p, &val));
    if (val != label_value) continue;

    // Filter dm points for dofs
    PetscCall(PetscSectionGetDof(section, p, &dof));
    if (dof <= 0) continue;

    // Filter dm point for local ownership
    PetscCall(DMPlexGetPointGlobal(dm, p, &global_offset, NULL));
    if (global_offset < 0) continue;

    // Check if this point is clamped in Ratel (exclude it from coupling)
    PetscBool is_clamped = PETSC_FALSE;
    if (ratel) {
      for (PetscInt f = 0; f < RATEL_MAX_FIELDS; f++) {
        for (PetscInt j = 0; j < ratel->bc_clamp_count[f]; j++) {
          PetscInt val_clamp;
          PetscCall(DMLabelGetValue(label, p, &val_clamp));
          if (val_clamp == ratel->bc_clamp_faces[f][j]) {
            is_clamped = PETSC_TRUE;
            break;
          }
        }
        if (is_clamped) break;
      }
    }
    
    if (is_clamped) continue;

    *n_dofs += dof / dim; 
    local_dm_points[dmpoint_local_count++] = p;
  }
  // Cleanup temporary label
  PetscCall(DMLabelDestroy(&temp_label));

  // Check for empty label
  if (dmpoint_local_count == 0) {
    PetscCall(PetscFree(local_dm_points));
    *n_dofs = 0;
    *dof_coords = NULL;
    *petsc_indices = NULL;
    *local_points  = NULL; 
    PetscFunctionReturn(0);
  }

  // Now we will extract coordinates and build our petsc_indices array to map preCICE data to petsc data.
  // allocating memory
  PetscCall(PetscMalloc1(*n_dofs * dim, dof_coords));
  // Get coordinate section and local vec 
  PetscSection coord_section;
  Vec          coord_vec;
  const PetscScalar *coords;
  PetscCall(DMGetCoordinateSection(dm, &coord_section));
  PetscCall(DMGetCoordinatesLocal(dm, &coord_vec));
  PetscCall(VecGetArrayRead(coord_vec, &coords)); // get pointer to coordinate array for reading
  
  // loop through local vertices and extract coordinates
  PetscInt node_idx = 0; // Running counter for the flat preCICE array
  for (PetscInt v = 0; v < dmpoint_local_count; v++) {
    PetscInt p = local_dm_points[v];
    PetscInt dof, offset;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(PetscSectionGetOffset(coord_section, p, &offset));

    PetscInt dofs_per_dmpoint = dof / dim;
  
    for (PetscInt n = 0; n < dofs_per_dmpoint; n++) {
      for (PetscInt d = 0; d < dim; d++) {
        (*dof_coords)[node_idx * dim + d] = coords[offset + (n * dim) + d];
      }
      node_idx++;
    }
  }
  // free read-only pointer to the coordinate array
  PetscCall(VecRestoreArrayRead(coord_vec, &coords));

  // We have now n_dofs, which is the number of owned dofs, 
  // the local_dm_points array which maps from 0..dmpoint_local_count to the index in the DM
  // and the dof_coords array which has the coordinates for those dofs, in the same layout as precice
  // x_1, y_1, z_1, x_2, y_2, z_2, ...

  // Build PETSc indices for DOF mapping 
  PetscCall(PetscMalloc1(*n_dofs * dim, petsc_indices));

  node_idx = 0; // Reset counter
  for (PetscInt v = 0; v < dmpoint_local_count; v++) {
    PetscInt p = local_dm_points[v];
    PetscInt dof, offset;
    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(PetscSectionGetOffset(section, p, &offset));

    PetscInt nodes_on_this_point = dof / dim;

    for (PetscInt n = 0; n < nodes_on_this_point; n++) {
      for (PetscInt d = 0; d < dim; d++) {
        // Map every DOF for every node on this point
        (*petsc_indices)[node_idx * dim + d] = offset + (n * dim) + d;
      }
      node_idx++;
    }
  }

  *local_points = local_dm_points;
  // Final Cleanup
  PetscFunctionReturn(0);
}
