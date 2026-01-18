# 📈 Toyota Stock Price Forecasting (LSTM)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abhinavmamidwar-toyota-stock-price-prediction-app-fhnezg.streamlit.app/)

## 🔗 Live Demo
Check out the interactive dashboard here: [https://abhinavmamidwar-toyota-stock-price-prediction-app-fhnezg.streamlit.app/](https://abhinavmamidwar-toyota-stock-price-prediction-app-fhnezg.streamlit.app/)

## 🚀 Project Overview
This project uses **Deep Learning (Long Short-Term Memory)** to predict the closing prices of Toyota Motor Corporation (TM). Stock markets are volatile, but by using a time-series architecture, this model captures long-term dependencies to provide highly accurate daily forecasts.

**Key Achievement:** The model achieved a **98.23% overall accuracy** (1.77% MAPE) on unseen test data.

[Image of a professional stock market prediction dashboard with next-day price forecast and trend indicators]

## 🧠 Model Intelligence
* **Architecture:** Stacked LSTM (50 units each) with Dropout layers (20%) to prevent overfitting.
* **Optimizer:** Adam (Adaptive Moment Estimation) for faster convergence.
* **Feature Engineering:** 60-day sliding window (using the past 2 months of price data to predict the next day).
* **Data Leakage Prevention:** Scaler parameters were fitted strictly on training data and saved as `scaler.gz` for consistent inference.

[Image of LSTM model architecture diagram showing input layer, hidden layers, and output layer]

## 📊 Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Accuracy** | **98.23%** |
| **MAPE** (Mean Absolute Percentage Error) | **1.77%** |
| **RMSE** (Root Mean Squared Error) | **3.99** |

## 🛠️ Tech Stack
* **Backend:** Python, TensorFlow, Keras, NumPy, Pandas
* **Frontend:** Streamlit (Custom Dark Theme)
* **Preprocessing:** Scikit-Learn (MinMaxScaler)
* **Version Control:** Git

## 📂 Project Structure
* `app.py`: The high-contrast Streamlit dashboard for real-time visualization.
* `model.py`: The training pipeline and model persistence logic.
* `toyota_model.h5`: The pre-trained LSTM neural network.
* `scaler.gz`: Saved scaling parameters for production-ready inference.
* `requirements.txt`: Environment dependencies.

## ⚙️ Setup & Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/Abhinavmamidwar/Toyota-Stock-Price-Prediction.git](https://github.com/Abhinavmamidwar/Toyota-Stock-Price-Prediction.git)

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Run the Dashboard:
    ```bash
    streamlit run app.py

Author: Abhinav Mamidwar