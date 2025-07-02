# pdoct-reg
------ VERSION HISTORY ------
## VERSION 2.X ##  
v2.0.1 `main` branch push 06/30/2025
- `utils.py` bug fixes when loading in volumes, can now properly resize to desired shape
  - note that selecting a larger en-face size will result in better keypoint detection especially in UWFOV
  - `lightglue.py` parameter `filter_threshold` can also be tuned

v2.0 `main` branch push 06/06/2025
- Major changes:
  - `axialMatching.m`, `averageVolumes.m` are now functions to be called by new main function `mainPostProc.m`
  - `mainPostProc.m` should be run directly on Python outputs to minimize scripts in post-processing
  - `genDOPUComposite.m` for composite overlay visualization
  - `globalMCorr.py` now incorporated in main.py
  - `eyeliner.py` includes support for DOPU registration
- This version is to be used for 2025 journal submission

## VERSION 1.X ##  
v1.3.1 `OCTA` branch push 04/24/2025
- **MATLAB Scripts**
  - **new** submit `averageVolumes.m`, `rmovMotion.m` (remove OCTA motion white strip artifacts), `getSynthFixed` (generate motion-free template) and `generateOCTAenface.m` (extracts SVP/DVP based on pixel height, notch filtering, CLAHE) to new OCTA branch

v1.3 `main` push 04/23/2025
- Add DOPU registration to Python script
- **MATLAB Scripts**
  - **new** `extractSeg.m` for generating SVP and DVP images based on 3 segmentation lines (ILM, IPL, ONL)

v1.2 `main` push 01/21/2025
- Python script now autoloads and registers all volumes with single run
- file error catching
- minor changes
- **MATLAB Scripts**
  - improved batchwise axial matching with interpolation
  - other minor changes

v1.1 `main` push 12/04/2024
- **MATLAB Scripts**
  - improved registration via batch axial matching
  - added code for visualization
  - **new** `evaluateReg.m` for MS-SSIM, SNR and CNR quantification

v1.0 `main` push 11/25/2024
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
