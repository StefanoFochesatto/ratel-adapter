#ifndef PETSC_DEBUG_H
#define PETSC_DEBUG_H

#include <petscdm.h>
#include <petscdmplex.h>
#include <petscdmlabel.h>
#include <petscviewer.h>

/**
* @brief Exports every point ID in the DM, its type, and its value in a specific Label to a text file.
*
* @param dm         The DMPlex object
* @param label_name The name of the label (e.g., "Face Sets")
* @param filename   The output filename (e.g., "label_debug.txt")
*/
static inline PetscErrorCode PetscDebugViewLabelAsText(DM dm, const char *label_name, const char *filename) {
DMLabel         label;
PetscBool       hasLabel;
PetscInt        pStart, pEnd, dim;
PetscViewer     viewer;
IS              valueIS;
const PetscInt *values;
PetscInt        numValues;

PetscFunctionBegin;
PetscCall(DMHasLabel(dm, label_name, &hasLabel));
if (!hasLabel) {
PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "DEBUG: Label '%s' not found.\n", label_name));
PetscFunctionReturn(PETSC_SUCCESS);
}
PetscCall(DMGetLabel(dm, label_name, &label));
PetscCall(DMGetDimension(dm, &dim));
PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));

// Get all unique values in this label
PetscCall(DMLabelGetValueIS(label, &valueIS));
PetscCall(ISGetIndices(valueIS, &values));
PetscCall(ISGetSize(valueIS, &numValues));

PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)dm), filename, &viewer));
PetscCall(PetscViewerASCIIPrintf(viewer, "# DM Label Debug: %s\n", label_name));
PetscCall(PetscViewerASCIIPrintf(viewer, "# ID, Type, Label Values\n"));

for (PetscInt p = pStart; p < pEnd; ++p) {
PetscInt depth;
const char *type = "unknown";
PetscCall(DMPlexGetPointDepth(dm, p, &depth));

if (depth == 0) {
    type = "vertex";
} else if (depth == 1) {
    type = (dim == 1) ? "cell" : "edge";
} else if (depth == 2) {
    type = (dim == 2) ? "cell" : "face";
} else if (depth == 3) {
    type = "cell";
}

PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT ", %s", p, type));

// Check which values this point has
PetscBool first = PETSC_TRUE;
for (PetscInt v = 0; v < numValues; ++v) {
    PetscBool hasValue;
    PetscCall(DMLabelStratumHasPoint(label, values[v], p, &hasValue));
    if (hasValue) {
        if (first) {
            PetscCall(PetscViewerASCIIPrintf(viewer, ", %" PetscInt_FMT, values[v]));
            first = PETSC_FALSE;
        } else {
            PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT, values[v]));
        }
    }
}
if (first) {
    // Point has no values in this label
    PetscCall(PetscViewerASCIIPrintf(viewer, ", -1"));
}
PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
}

PetscCall(ISRestoreIndices(valueIS, &values));
PetscCall(ISDestroy(&valueIS));
PetscCall(PetscViewerDestroy(&viewer));
PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "DEBUG: Wrote Label Debug '%s' to %s\n", label_name, filename));
PetscFunctionReturn(PETSC_SUCCESS);
}


/**
* @brief Exports a DM and solution Vector to a CGNS file for ParaView, supporting time-stepping.
*
* @param dm         The DMPlex object
* @param U          The solution vector
* @param time       Current simulation time
* @param step       Current time step index
* @param filename   The output filename
*/
static inline PetscErrorCode PetscDebugSaveSolutionCGNS(DM dm, Vec U, PetscReal time, PetscInt step, const char *filename_base) {
PetscViewer viewer;
char        filename[PETSC_MAX_PATH_LEN];
PetscFunctionBegin;

// Generate a filename like mesh_debug.0005.cgns
// ParaView will automatically group these into a time series
PetscCall(PetscSNPrintf(filename, sizeof(filename), "%s.%04d.cgns", filename_base, (int)step));

PetscCall(DMSetOutputSequenceNumber(dm, step, time));
PetscCall(PetscViewerCGNSOpen(PetscObjectComm((PetscObject)dm), filename, FILE_MODE_WRITE, &viewer));
PetscCall(DMView(dm, viewer));
PetscCall(VecView(U, viewer));
PetscCall(PetscViewerDestroy(&viewer));
PetscFunctionReturn(PETSC_SUCCESS);
}

/**
* @brief Exports a DM to a CGNS file for ParaView.
*/
static inline PetscErrorCode PetscDebugViewCGNS(DM dm, const char *filename) {
PetscViewer viewer;
PetscFunctionBegin;
PetscCall(PetscViewerCGNSOpen(PetscObjectComm((PetscObject)dm), filename, FILE_MODE_WRITE, &viewer));
PetscCall(DMView(dm, viewer));
PetscCall(PetscViewerDestroy(&viewer));
PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "DEBUG: Wrote DM to %s\n", filename));
PetscFunctionReturn(PETSC_SUCCESS);
}

/**
* @brief Prints all DM labels and their unique values/strata.
*/
static inline PetscErrorCode PetscDebugPrintLabels(DM dm) {
PetscInt numLabels;
PetscFunctionBegin;
PetscCall(DMGetNumLabels(dm, &numLabels));
PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "--- DM Debug: %" PetscInt_FMT " Labels ---\n", numLabels));

for (PetscInt l = 0; l < numLabels; ++l) {
const char *labelName;
DMLabel     label;
IS          valueIS;
PetscInt    numValues;
const PetscInt *values;

PetscCall(DMGetLabelName(dm, l, &labelName));
PetscCall(DMGetLabelByNum(dm, l, &label));
PetscCall(DMLabelGetValueIS(label, &valueIS));
PetscCall(DMLabelGetNumValues(label, &numValues));
PetscCall(ISGetIndices(valueIS, &values));

PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "  Label [%" PetscInt_FMT "]: %s (%" PetscInt_FMT " values)\n", l, labelName, numValues));

for (PetscInt v = 0; v < numValues; ++v) {
    PetscInt size;
    PetscCall(DMLabelGetStratumSize(label, values[v], &size));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "    Value %" PetscInt_FMT ": %" PetscInt_FMT " points\n", values[v], size));
}

PetscCall(ISRestoreIndices(valueIS, &values));
PetscCall(ISDestroy(&valueIS));
}
PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "---------------------------\n"));
PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the contents of a preCICE raw double buffer
 *
 * @param comm       MPI communicator (usually PETSC_COMM_SELF since vertices are local)
 * @param name       A descriptive name for the output (e.g., "Read Buffer")
 * @param n_vertices Number of interface vertices
 * @param dim        Spatial dimension
 * @param buffer     The raw data array [x0, y0, z0, x1, y1, z1, ...]
 */
static inline PetscErrorCode PetscDebugPrintPreciceBuffer(MPI_Comm comm, const char *name, 
                                                          PetscInt n_vertices, PetscInt dim, 
                                                          const PetscReal *buffer) {
    PetscFunctionBegin;
    PetscCall(PetscPrintf(comm, "--- preCICE Buffer: %s ---\n", name));
    for (PetscInt i = 0; i < n_vertices; ++i) {
        PetscCall(PetscPrintf(comm, "  Vertex %" PetscInt_FMT ": [", i));
        for (PetscInt d = 0; d < dim; ++d) {
            PetscCall(PetscPrintf(comm, "%g", (double)buffer[i * dim + d]));
            if (d < dim - 1) {
                PetscCall(PetscPrintf(comm, ", "));
            }
        }
        PetscCall(PetscPrintf(comm, "]\n"));
    }
    PetscCall(PetscPrintf(comm, "---------------------------\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the values of a vector U on the vertices belonging to a specific label stratum.
 *
 * @param dm          The DMPlex object
 * @param U           The vector to read from (e.g., solution or force)
 * @param label_name  The name of the label (e.g., "Face Sets")
 * @param label_value The value in the label identifying the interface
 */
static inline PetscErrorCode PetscDebugPrintVectorOnInterface(DM dm, Vec U, const char *label_name, 
                                                              PetscInt label_value) {
    DMLabel         label;
    IS              stratumIS;
    const PetscInt *points;
    PetscInt        numPoints, dim;
    PetscSection    section;
    const PetscScalar *u_array;
    MPI_Comm        comm;
    PetscInt        vStart, vEnd;

    PetscFunctionBegin;
    PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(DMGetLabel(dm, label_name, &label));
    if (!label) {
        PetscCall(PetscPrintf(comm, "DEBUG: Label '%s' not found.\n", label_name));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Ensure label is complete on vertices
    DMLabel temp_label;
    PetscCall(DMLabelDuplicate(label, &temp_label));
    PetscCall(DMPlexLabelComplete(dm, temp_label));

    PetscCall(DMLabelGetStratumIS(temp_label, label_value, &stratumIS));
    if (!stratumIS) {
        PetscCall(PetscPrintf(comm, "DEBUG: Label '%s' value %" PetscInt_FMT " has no points.\n", label_name, label_value));
        PetscCall(DMLabelDestroy(&temp_label));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    PetscCall(ISGetIndices(stratumIS, &points));
    PetscCall(ISGetSize(stratumIS, &numPoints));
    PetscCall(VecGetArrayRead(U, &u_array));

    PetscCall(PetscPrintf(comm, "--- Vector Values on Interface: %s (Value %" PetscInt_FMT ") ---\n", label_name, label_value));

    for (PetscInt i = 0; i < numPoints; ++i) {
        PetscInt p = points[i];
        if (p < vStart || p >= vEnd) continue; // Not a vertex

        PetscInt global_offset;
        PetscCall(DMPlexGetPointGlobal(dm, p, &global_offset, NULL));
        if (global_offset < 0) continue; // Not locally owned

        PetscInt dof, off;
        PetscCall(PetscSectionGetDof(section, p, &dof));
        PetscCall(PetscSectionGetOffset(section, p, &off));

        if (dof > 0) {
            PetscCall(PetscPrintf(comm, "  Vertex %" PetscInt_FMT ": [", p));
            for (PetscInt d = 0; d < dof; ++d) {
                PetscCall(PetscPrintf(comm, "%g", (double)PetscRealPart(u_array[off + d])));
                if (d < dof - 1) {
                    PetscCall(PetscPrintf(comm, ", "));
                }
            }
            PetscCall(PetscPrintf(comm, "]\n"));
        }
    }

    PetscCall(VecRestoreArrayRead(U, &u_array));
    PetscCall(ISRestoreIndices(stratumIS, &points));
    PetscCall(ISDestroy(&stratumIS));
    PetscCall(DMLabelDestroy(&temp_label));
    PetscCall(PetscPrintf(comm, "---------------------------\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Exports the extracted interface coordinates to a simple CSV file.
 */
static inline PetscErrorCode PetscDebugExportInterfacePoints(MPI_Comm comm, PetscInt n_vertices, PetscInt dim, const PetscReal *coords, const char *filename) {
    PetscMPIInt rank;
    PetscFunctionBegin;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    
    char full_filename[PETSC_MAX_PATH_LEN];
    PetscCall(PetscSNPrintf(full_filename, sizeof(full_filename), "%s_rank%d.csv", filename, rank));
    
    FILE *fp = fopen(full_filename, "w");
    if (!fp) {
        SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Could not open file %s", full_filename);
    }
    
    fprintf(fp, "PointID,X,Y,Z\n");
    for (PetscInt i = 0; i < n_vertices; i++) {
        fprintf(fp, "%" PetscInt_FMT ",", i);
        for (PetscInt d = 0; d < dim; d++) {
            fprintf(fp, "%g", (double)coords[i * dim + d]);
            if (d < dim - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    PetscCall(PetscPrintf(comm, "DEBUG: Wrote interface points to %s\n", full_filename));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the mapping between preCICE flat buffer indices, PETSc local offsets, and coordinates.
 */
static inline PetscErrorCode PetscDebugPrintMapping(MPI_Comm comm, PetscInt n_vertices, PetscInt dim, const PetscInt *petsc_indices, const PetscReal *coords) {
    PetscFunctionBegin;
    PetscCall(PetscPrintf(comm, "--- preCICE to PETSc Mapping ---\n"));
    for (PetscInt i = 0; i < n_vertices; i++) {
        PetscCall(PetscPrintf(comm, "  preCICE Vertex %" PetscInt_FMT " -> PETSc Local Offsets [", i));
        for (PetscInt d = 0; d < dim; d++) {
            PetscCall(PetscPrintf(comm, "%" PetscInt_FMT, petsc_indices[i * dim + d]));
            if (d < dim - 1) PetscCall(PetscPrintf(comm, ", "));
        }
        PetscCall(PetscPrintf(comm, "] at Coords ["));
        for (PetscInt d = 0; d < dim; d++) {
            PetscCall(PetscPrintf(comm, "%g", (double)coords[i * dim + d]));
            if (d < dim - 1) PetscCall(PetscPrintf(comm, ", "));
        }
        PetscCall(PetscPrintf(comm, "]\n"));
    }
    PetscCall(PetscPrintf(comm, "--------------------------------\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Verifies that the global vector U is exactly 0.0 at all nodes NOT on the coupling interface.
 */
static inline PetscErrorCode PetscDebugVerifyZeroes(DM dm, Vec U, const char *label_name, PetscInt label_value) {
    DMLabel label, temp_label;
    IS stratumIS;
    MPI_Comm comm;
    Vec local_U;
    const PetscScalar *u_array;
    PetscInt pStart, pEnd, dim;
    PetscSection section;

    PetscFunctionBegin;
    PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMGetLabel(dm, label_name, &label));
    if (!label) PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(DMLabelDuplicate(label, &temp_label));
    PetscCall(DMPlexLabelComplete(dm, temp_label));

    PetscCall(DMLabelGetStratumIS(temp_label, label_value, &stratumIS));
    PetscCall(DMGetLocalVector(dm, &local_U));
    PetscCall(DMGlobalToLocal(dm, U, INSERT_VALUES, local_U));
    PetscCall(VecGetArrayRead(local_U, &u_array));
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));

    PetscInt non_zero_count = 0;
    for (PetscInt p = pStart; p < pEnd; ++p) {
        PetscInt dof, off, val = -1;
        PetscCall(PetscSectionGetDof(section, p, &dof));
        if (dof <= 0) continue;

        PetscCall(DMLabelGetValue(temp_label, p, &val));
        if (val == label_value) continue; // It is on the interface

        PetscCall(PetscSectionGetOffset(section, p, &off));
        for (PetscInt d = 0; d < dof; ++d) {
            if (PetscAbsScalar(u_array[off + d]) > 1e-14) {
                non_zero_count++;
                if (non_zero_count <= 5) {
                    PetscCall(PetscPrintf(comm, "DEBUG ERROR: Non-zero value %g found at non-interface point %" PetscInt_FMT " (dof %" PetscInt_FMT ")\n", (double)PetscAbsScalar(u_array[off + d]), p, d));
                }
            }
        }
    }

    if (non_zero_count == 0) {
        PetscCall(PetscPrintf(comm, "DEBUG: Verified %s boundary vector is exactly 0.0 on all non-interface nodes.\n", label_name));
    } else {
        PetscCall(PetscPrintf(comm, "DEBUG ERROR: Found %" PetscInt_FMT " non-zero DOF entries on non-interface nodes!\n", non_zero_count));
    }

    PetscCall(VecRestoreArrayRead(local_U, &u_array));
    PetscCall(DMRestoreLocalVector(dm, &local_U));
    if (stratumIS) PetscCall(ISDestroy(&stratumIS));
    PetscCall(DMLabelDestroy(&temp_label));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Traces delta displacements across implicit checkpoint rollbacks.
 */
static inline PetscErrorCode PetscDebugTraceDeltaDisplacements(MPI_Comm comm, PetscInt n_vertices, PetscInt dim, Vec local_sol, Vec old_solution, Vec delta_U, const PetscInt *petsc_indices, const PetscReal *precice_buffer) {
    const PetscScalar *cur_array, *old_array, *delta_array;
    PetscFunctionBegin;
    
    PetscCall(VecGetArrayRead(local_sol, &cur_array));
    PetscCall(VecGetArrayRead(old_solution, &old_array));
    PetscCall(VecGetArrayRead(delta_U, &delta_array));

    PetscCall(PetscPrintf(comm, "--- Delta Displacements Trace (First 5 nodes) ---\n"));
    PetscInt nodes_to_print = (n_vertices > 5) ? 5 : n_vertices;
    for (PetscInt i = 0; i < nodes_to_print; i++) {
        PetscCall(PetscPrintf(comm, "  Vertex %" PetscInt_FMT ":\n", i));
        for (PetscInt d = 0; d < dim; d++) {
            PetscInt idx = petsc_indices[i * dim + d];
            PetscReal cur = PetscRealPart(cur_array[idx]);
            PetscReal old = PetscRealPart(old_array[idx]);
            PetscReal delta = PetscRealPart(delta_array[idx]);
            PetscReal precice_val = precice_buffer[i * dim + d];
            PetscCall(PetscPrintf(comm, "    dim %" PetscInt_FMT " | Cur: %10.4e | Old: %10.4e | Delta(PETSc): %10.4e | preCICE Buffer: %10.4e\n", 
                                  d, (double)cur, (double)old, (double)delta, (double)precice_val));
        }
    }
    PetscCall(PetscPrintf(comm, "-------------------------------------------------\n"));

    PetscCall(VecRestoreArrayRead(local_sol, &cur_array));
    PetscCall(VecRestoreArrayRead(old_solution, &old_array));
    PetscCall(VecRestoreArrayRead(delta_U, &delta_array));

    PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* PETSC_DEBUG_H */