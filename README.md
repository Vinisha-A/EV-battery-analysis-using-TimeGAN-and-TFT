# EV Battery State Prediction Using TimeGAN and Temporal Fusion Transformer (TFT)

A deep learning framework for electric vehicle battery health analytics that combines **TimeGAN-based synthetic time-series generation** with a **Temporal Fusion Transformer (TFT)** to predict State of Charge (SOC), State of Health (SOH), and Remaining Useful Life (RUL).

## Overview

Reliable battery monitoring is essential for electric vehicle safety, performance, and longevity. However, battery datasets can be limited, noisy, or imbalanced, while battery degradation often involves complex nonlinear patterns and temporal dependencies.

This project explores a hybrid approach that combines Time-Series Generative Adversarial Networks (TimeGAN) and Temporal Fusion Transformer (TFT) to support battery state prediction.

TimeGAN is used to generate synthetic multivariate time-series data intended to preserve temporal behavior and relationships between features. The augmented data is then used in the TFT-based prediction workflow for estimating important battery indicators.

The project also explores synthetic data quality through statistical similarity measures and t-SNE visualization.

## Key Objectives

- Generate synthetic battery time-series data using TimeGAN.
- Explore synthetic data quality using statistical comparisons and t-SNE visualization.
- Develop a TFT-based workflow for battery state prediction.
- Predict three battery indicators: SOC, SOH, and RUL.
- Investigate how synthetic data augmentation can support learning from limited battery data.

## Methodology

### 1. Synthetic Time-Series Generation with TimeGAN

TimeGAN is used to generate synthetic multivariate time-series samples that aim to preserve temporal patterns and relationships among battery features.

### 2. Synthetic Data Evaluation

The generated data is examined using statistical similarity measures and t-SNE visualization to compare its characteristics with the original data.

These evaluations help assess similarity; they do not, by themselves, establish that synthetic data is physically accurate or improves downstream predictions.

### 3. Battery State Prediction with TFT

The augmented dataset is used in a Temporal Fusion Transformer workflow. TFT combines attention mechanisms and gating components to model temporal dependencies and relationships relevant to forecasting.

The target battery indicators are:

- **State of Charge (SOC):** An estimate of the battery's remaining charge.
- **State of Health (SOH):** An indicator of battery condition relative to its reference state.
- **Remaining Useful Life (RUL):** An estimate of the battery's remaining useful operating life.

### 4. Model Evaluation

Prediction performance is assessed using Root Mean Squared Error (RMSE). The reported values should be interpreted in the context of the dataset, target scaling, prediction setup, and evaluation split used in the experiments.

## Reported Results

The project reported the following RMSE values:

| Prediction Target | RMSE |
|---|---:|
| State of Charge (SOC) | 0.4756 |
| State of Health (SOH) | 0.3750 |
| Remaining Useful Life (RUL) | 0.0401 |

**Note:** RMSE depends on the units and scaling of each target. These values should not be compared directly across targets unless their scales and evaluation procedures are considered. Refer to the notebook for the experimental details.

## Repository Contents

| Notebook | Description |
|---|---|
| `TimeGAN_Synthetic_data_generation.ipynb` | TimeGAN-based synthetic time-series generation and related analysis |
| `EV_Battery_Temporal_Fusion_Transformer.ipynb` | TFT-based battery state prediction workflow |

## Technology Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow/Keras or the framework used in the notebooks
- **Generative Modeling:** TimeGAN
- **Time-Series Forecasting:** Temporal Fusion Transformer (TFT)
- **Data Processing:** Pandas, NumPy
- **Visualization and Evaluation:** Matplotlib, t-SNE, statistical similarity measures
- **Development Environment:** Google Colab / Jupyter Notebook

*The exact libraries and versions should be confirmed against the notebook imports.*

## Getting Started

### Run in Google Colab

1. Open the desired notebook from the repository.
2. Select **Open in Colab**, or download the `.ipynb` file and upload it to Google Colab.
3. Review the notebook's dependency installation cells.
4. Configure access to the required dataset.
5. Run the notebook cells in order.

The notebooks may require a runtime with sufficient memory and compute resources, depending on the model configuration and dataset size.

## Research Context

This project is associated with the research work titled:

**"Synthetic to Smart: Enhancing EV Battery Insights Using TimeGAN and TFT."**

The proposed framework explores the use of synthetic time-series augmentation and temporal deep learning for battery state prediction.

## Limitations

- Synthetic data quality depends on the training data and TimeGAN configuration.
- Similarity between synthetic and original data does not guarantee improved predictive performance.
- Model results depend on dataset quality, preprocessing, target scaling, and evaluation methodology.
- The reported RMSE values should be interpreted alongside the corresponding experimental setup.
- Further validation on independent battery datasets would be needed to assess generalization to unseen operating conditions.

## Future Improvements

- Evaluate the models on additional battery datasets and operating conditions.
- Compare TFT performance with suitable baseline forecasting models.
- Investigate the effect of synthetic data augmentation on held-out test performance.
- Improve reproducibility through dependency versioning and documented dataset preparation.
- Include prediction-versus-actual plots and additional evaluation metrics.

## Author

**Vinisha A.**

B.E. Artificial Intelligence and Machine Learning | Deep Learning | Time-Series Modeling
