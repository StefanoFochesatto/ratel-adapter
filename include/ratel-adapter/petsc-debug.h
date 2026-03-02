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
#endif /* PETSC_DEBUG_H */