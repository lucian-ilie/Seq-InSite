## Performance Evaluation

This section outlines the inputs required and the format expected for evaluating the performance of the model.

### Inputs

The performance evaluation requires three primary inputs:

1. **Predictions Directory:**
   - **Description:** This directory should contain files, each corresponding to a specific protein. Each line in these files represents a prediction value ranging from 0 to 1.
   - **Path:** Specify the full path to the directory containing the prediction files.

2. **Actual Labels File:**
   - **Format:** The actual labels should be provided in a FASTA format file.
   - **Contents:**
     - **First Line:** Protein name.
     - **Second Line:** Amino acid sequence of the protein.
     - **Third Line:** Corresponding labels for each amino acid in the sequence.

3. **Model Name:**
   - **Purpose:** Enter the name of the model being evaluated. This name will be used for identification in the generated reports and output summaries.
