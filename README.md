# ANN Reservoir Inflow Forecast Dashboard

Streamlit application for visualizing ANN-based inflow predictions for reservoirs in the Ohře river basin.

The app allows interactive exploration of:

- Real vs ensemble ANN predictions (time series)
- Model performance metrics across multiple runs
- Calibration vs validation comparison


---

## Features

### 1. Time Series View

For a selected reservoir and forecast horizon:

- Real observations (black line)
- Individual ANN runs (grey ensemble members)
- Ensemble mean (highlighted line)
- Optional uncertainty band (mean ± 2·std)
- Separate panels for:
  - Calibration
  - Validation

This allows visual inspection of:

- Spread between runs  
- Bias patterns  
- Stability of ensemble predictions  
- Generalization from calibration to validation  


---

### 2. Metrics View

For a selected metric (e.g., KGE, RMSE, NS, etc.):

- Boxplots across runs
- Separate panels for:
  - Calibration
  - Validation
- Optional shared y-axis for direct comparison

This allows evaluation of:

- Run-to-run variability  
- Robustness  
- Overfitting detection  
- Performance degradation from calibration to validation  


---

## Expected Folder Structure

The repository must follow this structure:

