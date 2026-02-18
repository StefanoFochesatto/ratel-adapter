/**
 * @file mesh_extraction.c
 * @brief DMPlex topology traversal for coupling interface extraction
 */

#include <ratel-adapter/ratel-adapter.h>

/**
 * @brief Extract boundary vertices from DMPlex
 *
 * Traverses DMPlex topology to find vertices on the coupling interface.
 * Only includes locally-owned vertices (no ghosts).
 *
 * Algorithm:
 * 1. Get DMLabel by name
 * 2. Get stratum IS for label value
 * 3. For each face, get transitive closure to find vertices
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
PetscErrorCode RatelAdapterExtractBoundaryVertices(DM dm, const char* label_name,
                                                   PetscInt label_value, PetscInt dim,
                                                   PetscInt* n_vertices, PetscReal** vertex_coords,
                                                   PetscInt** petsc_indices) {
  PetscFunctionBeginUser;

  MPI_Comm comm;

  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  /* Get DMLabel */
  DMLabel label;
  PetscCall(DMGetLabel(dm, label_name, &label));
  if (!label) {
    SETERRQ1(comm, PETSC_ERR_ARG_WRONGSTATE, "Label '%s' not found in DM", label_name);
  }

  /* Get stratum (faces with matching label value) */
  IS face_is;
  PetscCall(DMLabelGetStratumIS(label, label_value, &face_is));

  if (!face_is) {
    /* No faces with this label value */
    *n_vertices = 0;
    *vertex_coords = NULL;
    *petsc_indices = NULL;
    PetscFunctionReturn(0);
  }

  /* Get face indices */
  PetscInt n_faces;
  const PetscInt* faces;
  PetscCall(ISGetLocalSize(face_is, &n_faces));
  PetscCall(ISGetIndices(face_is, &faces));

  /* Get vertex range */
  PetscInt v_start, v_end;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &v_start, &v_end));

  /* Temporary storage for vertex points */
  PetscInt* temp_vertices;
  PetscInt max_vertices = n_faces * 10; /* Estimate: 10 vertices per face */
  PetscCall(PetscMalloc1(max_vertices, &temp_vertices));
  PetscInt vertex_count = 0;

  /* Get section for DOF mapping */
  PetscSection section;
  PetscCall(DMGetLocalSection(dm, &section));

  /* Iterate over faces */
  for (PetscInt f = 0; f < n_faces; f++) {
    PetscInt face = faces[f];

    /* Get closure of face (includes face, edges, vertices) */
    PetscInt* closure = NULL;
    PetscInt closure_size;
    PetscCall(DMPlexGetTransitiveClosure(dm, face, PETSC_TRUE, &closure_size, &closure));

    /* Iterate closure to find vertices */
    for (PetscInt c = 0; c < closure_size * 2; c += 2) {
      PetscInt point = closure[c];

      /* Check if this is a vertex */
      if (point >= v_start && point < v_end) {
        /* Check if locally owned */
        PetscInt global_offset;
        PetscCall(DMPlexGetPointGlobal(dm, point, &global_offset, NULL));

        if (global_offset >= 0) {
          /* Locally owned vertex - add to list if not already present */
          PetscBool already_added = PETSC_FALSE;
          for (PetscInt v = 0; v < vertex_count; v++) {
            if (temp_vertices[v] == point) {
              already_added = PETSC_TRUE;
              break;
            }
          }

          if (!already_added) {
            if (vertex_count >= max_vertices) {
              /* Resize */
              max_vertices *= 2;
              PetscCall(PetscRealloc(max_vertices * sizeof(PetscInt), &temp_vertices));
            }
            temp_vertices[vertex_count++] = point;
          }
        }
      }
    }

    PetscCall(DMPlexRestoreTransitiveClosure(dm, face, PETSC_TRUE, &closure_size, &closure));
  }

  PetscCall(ISRestoreIndices(face_is, &faces));
  PetscCall(ISDestroy(&face_is));

  *n_vertices = vertex_count;

  if (vertex_count == 0) {
    PetscCall(PetscFree(temp_vertices));
    *vertex_coords = NULL;
    *petsc_indices = NULL;
    PetscFunctionReturn(0);
  }

  /* Extract coordinates */
  PetscCall(PetscMalloc1(vertex_count * dim, vertex_coords));

  /* Get coordinate section and vector */
  PetscSection coord_section;
  Vec coord_vec;
  PetscCall(DMGetCoordinateSection(dm, &coord_section));
  PetscCall(DMGetCoordinatesLocal(dm, &coord_vec));

  const PetscScalar* coords;
  PetscCall(VecGetArrayRead(coord_vec, &coords));

  for (PetscInt v = 0; v < vertex_count; v++) {
    PetscInt vertex = temp_vertices[v];
    PetscInt offset;
    PetscCall(PetscSectionGetOffset(coord_section, vertex, &offset));

    for (PetscInt d = 0; d < dim; d++) {
      (*vertex_coords)[v * dim + d] = coords[offset + d];
    }
  }

  PetscCall(VecRestoreArrayRead(coord_vec, &coords));

  /* Build PETSc indices */
  PetscCall(PetscMalloc1(vertex_count * dim, petsc_indices));

  for (PetscInt v = 0; v < vertex_count; v++) {
    PetscInt vertex = temp_vertices[v];
    PetscInt offset;
    PetscCall(PetscSectionGetOffset(section, vertex, &offset));

    for (PetscInt d = 0; d < dim; d++) {
      (*petsc_indices)[v * dim + d] = offset + d;
    }
  }

  PetscCall(PetscFree(temp_vertices));

  PetscFunctionReturn(0);
}
