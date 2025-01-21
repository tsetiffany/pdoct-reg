# pdoct-reg
------ VERSION HISTORY ------

## VERSION 1.X ##  
v1.2 push 01/21/2025
- Python script now autoloads and registers all volumes with single run
- file error catching
- minor changes
- **MATLAB Scripts**
  - improved batchwise axial matching with interpolation
  - other minor changes

v1.1 push 12/04/2024
- **MATLAB Scripts**
  - improved registration via batch axial matching
  - added code for visualization
  - **new** `evaluateReg.m` for MS-SSIM, SNR and CNR quantification

v1.0 push 11/25/2024
- **Entrypoint**: `main.py`
- **2D Feature Matching** (Python)
- **3D Registration** (Python)
- **Environment Setup**:
  - Use the `environment.yml` file for a Conda virtual environment.
  - Alternatively, use `requirements.txt` for a local environment setup.
- **Output**: Python saves `.mat` files of fixed and registered volumes for further processing.
- **MATLAB Scripts**: Includes scripts for axial matching and multi-volume averaging.
  - `axialMatching.m`
  - `averageVolumes.m`
