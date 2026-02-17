# ⚡ Energy Consumption Predictions with Bayesian LSTMs

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</p>

<p align="center">
  <b>A probabilistic deep learning approach for time series forecasting with uncertainty quantification</b>
</p>

---

## 📋 Overview

This project implements a **Bayesian LSTM** neural network for predicting household energy consumption with built-in uncertainty quantification. Unlike traditional point predictions, this approach provides confidence intervals, enabling better decision-making in real-world energy management scenarios.

### Key Features

- 🔮 **Probabilistic Predictions** - Get confidence bounds, not just point estimates
- 🧠 **Bayesian Deep Learning** - Monte Carlo Dropout for uncertainty estimation  
- 📊 **Interactive Visualizations** - Plotly-powered charts for analysis
- ⚡ **PyTorch Implementation** - Clean, modular, and extensible code

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Bayesian LSTM Model                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input (sequence_length × n_features)                      │
│            ↓                                                │
│   ┌─────────────────────┐                                   │
│   │   LSTM Encoder      │  128 hidden units, 2 layers       │
│   │   + MC Dropout      │  50% dropout (kept at inference)  │
│   └─────────────────────┘                                   │
│            ↓                                                │
│   ┌─────────────────────┐                                   │
│   │   LSTM Decoder      │  32 hidden units, 2 layers        │
│   │   + MC Dropout      │  50% dropout (kept at inference)  │
│   └─────────────────────┘                                   │
│            ↓                                                │
│   ┌─────────────────────┐                                   │
│   │   Fully Connected   │  Output: 1 (energy prediction)    │
│   └─────────────────────┘                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**UCI Machine Learning Repository - Appliances Energy Prediction**

| Property | Value |
|----------|-------|
| Samples | 19,735 |
| Features | Date, Day of Week, Hour, Energy Consumption |
| Frequency | 10 minutes (resampled to 1 hour) |
| Target | Log-transformed energy consumption (Wh) |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn plotly
```

### Running the Notebook

1. Clone this repository
2. Open `Energy_Consumption_Predictions_with_Bayesian_LSTMs_in_PyTorch.ipynb`
3. Run all cells sequentially

---

## 📈 Results

### Uncertainty Quantification

The model generates **99% confidence intervals** using Monte Carlo sampling:

- Run 100 forward passes with dropout enabled
- Compute mean and standard deviation of predictions
- Construct confidence bounds: μ ± 3σ

### Model Performance

| Metric | Description |
|--------|-------------|
| Loss Function | Mean Squared Error (MSE) |
| Training Split | 70% |
| Sequence Length | 10 time steps |
| Batch Size | 128 |
| Epochs | 150 |

---

## 🔬 Methodology

### Why Bayesian LSTMs?

Traditional neural networks provide **point estimates** without confidence measures. In critical applications like energy forecasting, understanding prediction uncertainty is crucial for:

- ⚠️ Risk assessment and safety margins
- 📅 Resource planning and scheduling  
- 💰 Cost optimization decisions

### Monte Carlo Dropout

Instead of expensive Bayesian inference, this implementation uses **MC Dropout**:

1. Keep dropout active during inference (not just training)
2. Run multiple forward passes through the network
3. Each pass samples a different sub-network
4. Aggregate predictions to estimate uncertainty

---

## 📁 Project Structure

```
probablistic reasoning/
├── README.md
├── Energy_Consumption_Predictions_with_Bayesian_LSTMs_in_PyTorch.ipynb
├── household_power_consumption.txt
└── [Generated HTML visualizations]
```

---

## 🛠️ Technical Details

### Hyperparameters

```python
hidden_size_1 = 128      # Encoder LSTM units
hidden_size_2 = 32       # Decoder LSTM units  
stacked_layers = 2       # LSTM layers per stage
dropout_probability = 0.5
learning_rate = 0.01
n_epochs = 150
batch_size = 128
sequence_length = 10
```

### Data Preprocessing

- **Temporal Features**: Day of week, hour of day
- **Log Transformation**: Applied to target variable
- **MinMax Scaling**: Features scaled to [0, 1]
- **Sliding Window**: 10-step sequences for LSTM input

---

## 📚 References

- [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) - Gal & Ghahramani, 2016
- [UCI ML Repository - Energy Dataset](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)

---
