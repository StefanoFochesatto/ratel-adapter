# Update preCICE Configuration for Aitken Acceleration

## Objective
Replace the current IQN-IMVJ acceleration method with Aitken acceleration in the `precice-config.xml` file.

## Key Files & Context
*   **File:** `precice-config.xml`
*   **Context:** The current implicit coupling scheme uses `IQN-IMVJ`. The user requested switching to `Aitken` acceleration, which is often more robust when the solution remains zero for initial periods, as it does not rely on building a history of previous time windows.

## Implementation Steps
1.  **Modify `precice-config.xml`:** Locate the `<coupling-scheme:serial-implicit>` block.
2.  **Remove `IQN-IMVJ`:** Delete the entire `<acceleration:IQN-IMVJ>` configuration block.
3.  **Insert `Aitken`:** Add the new `<acceleration:aitken>` block with the following configuration:
    *   Target the data exchanged from the second participant to the first: `<data name="DisplacementDelta" mesh="Solid-Mesh" />`
    *   Set the recommended initial relaxation for strongly coupled problems: `<initial-relaxation value="0.1" />`

## Verification & Testing
1.  **Configuration Check:** Ensure the XML syntax is correct and well-formed.
2.  **Execution Check:** The user can run the simulation using `run.sh` to verify that preCICE initializes correctly with the new Aitken acceleration method and does not produce configuration errors.