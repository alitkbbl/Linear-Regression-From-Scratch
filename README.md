# 🚀 Professional Linear Regression (from Scratch) 

This repository contains a high-performance implementation of Linear Regression with L2 Regularization (Ridge), built entirely from scratch using Python and NumPy. It demonstrates the internal mechanics of Gradient Descent, feature scaling, and model evaluation without relying on high-level ML libraries for the core algorithm.

## ✨ Key Features

*   **Pure NumPy Core:** Hand-coded forward pass, backpropagation, and weight updates.
*   **L2 Regularization:** Prevents overfitting by penalizing large weights.
*   **Automatic Data Pipeline:** Includes handling missing values, feature scaling (standardization), and dataset splitting.
*   **Performance Metrics:** Implements the R-Squared ($R^2$) score for evaluation.
*   **Live Training Logs:** Real-time feedback on loss reduction during training.
*   **Visual Convergence:** Generates plots to visualize the optimization process.

## 📈 Example Results

After training for 1500 iterations, the model typically achieves:

- **Consistent Loss Reduction:** As seen in the generated convergence plot.
- **R2 Score:** Provides an accuracy measurement on the test set (usually around 0.60+ for this dataset with these hyper-parameters).
