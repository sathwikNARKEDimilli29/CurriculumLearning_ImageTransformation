# This Repository is the code implementation for Research Study "Enhancing Network Intrusion Detection with a Stage-Wise Curriculum Learning Framework with Image Transformation and XAI"

## Steps to Run the Project

### 1. Organize the Dataset
- **Bifurcate Attacks**: 
  - Separate the attack traffic data and normal traffic data into two directories:
    - `Normal/`: Contains all normal traffic files.
    - `Attack/`: Contains all attack traffic files.
  - Ensure all `.csv` files are categorized accordingly.

### 2. Define Stages for Curriculum Learning
Define the Curriculum Learning Stages according to the attack types.

### 3. Install Required Dependencies
Run the following command to install all required dependencies:

```bash
pip install -r requirements.txt
