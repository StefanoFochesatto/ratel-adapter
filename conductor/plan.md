# Implementation Plan: Redundant Checkpointing Cleanup

## Objective
Remove redundant state tracking inside the preCICE adapter by delegating rollback responsibilities directly to the PETSc `TS` solver (for `ex03`) or the user application (for `ex02`). Additionally, optimize the `RatelAdapter` struct to only allocate the `delta_U` and `old_solution` vectors if `is_delta` is true, and properly update `old_solution` for calculating displacement increments.

## Key Files & Context
- `src/adapter.c` & `include/ratel-adapter/ratel-adapter.h`: Adapter implementation where state tracking will be stripped and replaced with lightweight requirement wrappers.
- `examples/ex02-quasistatic-precice.c`: Manual time-stepping loop that will now manually save its state.
- `examples/ex03-dynamic-precice.c`: Uses PETSc `TS`, which natively handles `TSRollBack()`.

## Implementation Steps

### Phase 1: Struct & Allocation Cleanup in `src/adapter.c`
1. **Remove Unused Struct Fields**: Remove `checkpoint_solution`, `checkpoint_V`, `checkpoint_time`, `checkpoint_step`, and `has_checkpoint` from `struct _p_RatelAdapter`.
2. **Conditional Allocation of Deltas**: In `RatelAdapterInitialize()`, ensure `delta_U` and `old_solution` are only allocated if `adapter->params.is_delta` is true. 
3. **Correct Initialization of `old_solution`**: When `is_delta` is true, properly initialize `old_solution` using the initial `solution` vector passed to `RatelAdapterInitialize()` instead of just copying uninitialized or zeroed memory.

### Phase 2: Function Replacement
1. **Remove the Old**: Delete `RatelAdapterSaveCheckpointIfRequired()` and `RatelAdapterReloadCheckpointIfRequired()` from `src/adapter.c` and `ratel-adapter.h`.
2. **Add the New**: Introduce two lightweight wrapper functions in `adapter.c` and `ratel-adapter.h`:
   - `PetscErrorCode RatelAdapterRequiresWritingCheckpoint(RatelAdapter adapter, PetscBool *requires);`
   - `PetscErrorCode RatelAdapterRequiresReadingCheckpoint(RatelAdapter adapter, PetscBool *requires);`

### Phase 3: Shift Delta Update Logic
1. **Relocate to Advance**: The logic that updates `old_solution` at the end of a time window was previously buried in the reload checkpoint function. Move this directly to `RatelAdapterAdvance()`.
2. **Implementation**: After calling `precicec_advance(dt)`, add a check for `precicec_isTimeWindowComplete()`. If true and `is_delta` is true, scatter the current global `solution` to a local vector and copy it to `adapter->old_solution` for the next time window.

### Phase 4: Example Updates
1. **`examples/ex03-dynamic-precice.c`**:
   - In `PreStepAdapter`, remove the call to `RatelAdapterSaveCheckpointIfRequired`.
   - In `PostStepAdapter`, replace the call to `RatelAdapterReloadCheckpointIfRequired` with `RatelAdapterRequiresReadingCheckpoint`. If reading is required, call `TSRollBack(ts)` and set the converged reason to iterating.
2. **`examples/ex02-quasistatic-precice.c`**:
   - Before the TS step, replace `RatelAdapterSaveCheckpointIfRequired` with `RatelAdapterRequiresWritingCheckpoint`. If required, explicitly save the global `U` and `time`/`step` into separate tracking variables managed in `main()`.
   - After `RatelAdapterAdvance()`, replace `RatelAdapterReloadCheckpointIfRequired` with `RatelAdapterRequiresReadingCheckpoint`. If required, manually restore `U` and `time`/`step` and `continue` to retry the step.

## Verification & Testing
- Build the examples and run existing tests (if any) to ensure no syntax errors.
- Confirm `delta_U` and `old_solution` are not allocated unless explicitly required by the options or data names.
- Verify `ex03-dynamic-precice.c` handles implicit coupling correctly using `TSRollBack()`.