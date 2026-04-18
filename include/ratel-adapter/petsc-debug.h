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

#endif /* PETSC_DEBUG_H */