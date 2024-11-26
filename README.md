# pdoct-reg
------ VERSION HISTORY ------

### ///// v1.0 push 11/25/2024 /////

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