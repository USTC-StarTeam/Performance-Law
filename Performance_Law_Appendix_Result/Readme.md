# README for Performance Law Fitting

This README provides an overview of the `performance_law_fitting.py` script, which is used to model and fit the "Performance Law" for sequential recommendation systems. The script uses a parametric model to fit performance metrics against varied hyperparameters and plots the results.

## Requirements

- Python 3.7 or later
- NumPy
- Matplotlib
- SciPy
- tqdm
- Collections
- heapq

## Overview

The script performs the following tasks:

1. **Data Preparation and Normalization**: Prepares data for fitting by normalizing row values and arranging data into structured arrays.
2. **Model Definition and Fitting**: Defines a model function and fits it to the data using SciPy's curve fitting function.
3. **3D Visualization**: Visualizes the fitted model and actual data in 3D plots.
4. **Result Evaluation**: Calculates and prints R-squared values to evaluate model fit.
5. **Hyperparameter Search**: Performs grid search for hyperparameters and logs the top results.
6. **Heatmap Visualization**: Normalizes and visualizes data using heatmaps.

## Usage

Make sure you have your results matrix ready to be pasted in the corresponding location in `prepare_data`.

### Execution

To run the script, simply execute:

```bash
python performance_law_fitting.py
```

The script will output plots and performance metrics to evaluate the model fitting.

### Function Details

- **normalize_row:** Normalizes a row of numeric data.
- **prepare_data(p2):** Prepares and structures data for fitting.
- **model_func(X, w1, ...):** Defines the model function to be fitted to the data. It models how hyperparameters affect recommendation performance.
- **total_main(pf1, pf2, p2):** Main function to fit the model to data and visualize results.
- **grid_search_and_fit(k):** Performs a grid search to find the best parameters that fit the model.
- **plot_heatmap(data):** Visualizes normalized data as a series of heatmaps.

### Plotting and Visualization

- The script generates 3D plots to show the fit between model predictions and actual data.
- Heatmaps are generated to show normalized performance data across different settings.

### Hyperparameters

- **pf1** and **pf2**: Multiplicative factors for parameters within the model function.
- **p2:** Scaling factor for data normalization.

## Notes

- Ensure the data is accurately represented in the desired format in `prepare_data`.
- The `model_func` parameters should be adjusted to reflect the nuances of your specific dataset and performance metrics.
- This script assumes the presence of a dataset and related utilities such as `plot_heatmap` and `only_paint`, ensure these functions are defined in your working environment.

## Troubleshooting

- Adjust optimization parameters in `curve_fit` if fitting issues occur.
- Watch out for optimization warnings and runtime errors, as they are logged in the output for each dataset.

This script provides a versatile tool for modeling and optimizing performance laws in recommender systems using a structured parametric approach. Adjust parameters and dataset specifics to tune the model for your unique needs.