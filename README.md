# 📈 Stock Price Prediction Using Deep Learning (LSTM)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)

A deep learning-based application that predicts future stock closing prices using a Long Short-Term Memory (LSTM) neural network and provides an interactive Flask web application for training, visualization, and forecasting.

---

## Key Features

- Predicts future stock closing prices using an LSTM neural network
- Interactive Flask web application for training, prediction, and visualization
- Uses historical stock market data from Yahoo Finance
- Supports Apple (AAPL), Microsoft (MSFT), and Tesla (TSLA)
- Visualizes predicted vs. actual stock prices
- Evaluates model performance using RMSE, MAE, MAPE, and R² Score

---

## Overview

This project predicts future stock closing prices using a Long Short-Term Memory (LSTM) neural network. Historical stock market data is collected using the Yahoo Finance API, preprocessed into sequential time-series data, and used to train a deep learning model capable of learning temporal patterns in stock prices.

The trained model is integrated into a Flask web application where users can select a stock, train the model, visualize predictions, and evaluate forecasting performance.

---

## How It Works

The application follows this workflow:

1. **Data Collection**
   - Downloads historical stock market data using Yahoo Finance.

2. **Data Preprocessing**
   - Cleans missing values.
   - Normalizes stock prices.
   - Creates sequential time-series data for LSTM training.

3. **Model Training**
   - Trains an LSTM neural network using TensorFlow and Keras.

4. **Prediction**
   - Predicts future stock closing prices.

5. **Evaluation**
   - Computes RMSE, MAE, MAPE, and R² Score.

---

## Dataset

Historical stock price data is collected using the Yahoo Finance API (`yfinance`).

Supported stocks include:

- Apple (AAPL)
- Microsoft (MSFT)
- Tesla (TSLA)

Historical closing prices are transformed into sequential time-series data for training the LSTM model.

---

## Technologies Used

- **Python** – Programming language
- **TensorFlow & Keras** – Deep learning framework
- **Flask** – Web application framework
- **Pandas** – Data manipulation
- **NumPy** – Numerical computing
- **Scikit-learn** – Data preprocessing and evaluation
- **Matplotlib** – Data visualization
- **Yahoo Finance (yFinance)** – Historical stock market data

---

## Model Architecture

The stock price prediction model is built using a Long Short-Term Memory (LSTM) neural network with the following components:

- **Input Layer** – Receives sequential historical stock price data.
- **LSTM Layers** – Learn temporal dependencies and long-term patterns in stock prices.
- **Dropout Layer** – Helps reduce overfitting during training.
- **Dense Output Layer** – Predicts the next stock closing price.

### Training Configuration

- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Evaluation Metrics:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)
  - R² Score

---

## Project Structure

```text
Stock-Price-Prediction-LSTM/
│
├── demo/
│   ├── homepage-demo.mp4
│   └── results-demo.mp4
│
├── history/                      # Training history for each stock
│   ├── AAPL_training_history.csv
│   ├── MSFT_training_history.csv
│   └── TSLA_training_history.csv
│
├── models/                       # Trained LSTM models
│   ├── AAPL_lstm_model.keras
│   ├── MSFT_lstm_model.keras
│   └── TSLA_lstm_model.keras
│
├── predictions/                  # Generated prediction outputs
│   ├── AAPL_predictions.csv
│   ├── MSFT_predictions.csv
│   └── TSLA_predictions.csv
│
├── results/                      # Model evaluation metrics
│   ├── AAPL_results.txt
│   ├── MSFT_results.txt
│   └── TSLA_results.txt
│
├── scalers/                      # Saved feature scalers
│   ├── AAPL_scaler.pkl
│   ├── MSFT_scaler.pkl
│   └── TSLA_scaler.pkl
│
├── static/
│   ├── images/
│   └── style.css
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── error.html
│
├── app.py                        # Flask web application
├── model.py                      # LSTM model architecture
├── predict.py                    # Prediction logic
├── train_model.py                # Model training script
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── .gitignore
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/ayaronrios/Stock-Price-Prediction-LSTM.git
```

### Navigate to the project

```bash
cd Stock-Price-Prediction-LSTM
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:8000
```

---

## Application Demo

The Flask application provides an intuitive interface for training the LSTM model, visualizing stock price predictions, and evaluating forecasting performance.

### Homepage

<img width="930" height="1634" alt="Home Page" 
src="https://github.com/user-attachments/assets/a2c622f5-97a4-4607-be16-667102bd5083"/>

The homepage allows users to select a stock, configure model parameters, train the LSTM model, and generate stock price predictions.

### Prediction Results

<img width="930" height="3424" alt="Prediction Results" 
src="https://github.com/user-attachments/assets/e48957cc-66f3-411d-adc6-7d646f02ebf2"/>

The results page displays the predicted stock prices along with performance metrics and visualizations, enabling users to evaluate the model's forecasting accuracy.

### Demo Videos

- ▶️ **Home Page Demo** – [homepage-demo.mp4](demo/homepage-demo.mp4)

- ▶️ **Prediction Results Demo** – [results-demo.mp4](demo/results-demo.mp4)

---

## 📈 Evaluation Metrics

The model is evaluated using the following regression metrics:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R² Score

---

## Results

The trained LSTM model successfully learns temporal patterns from historical stock price data to forecast future closing prices for the selected stocks. Model performance is demonstrated below using one of the evaluated test cases (Tesla).

### Model Performance (Tesla Test Case)

- **Predicted Closing Price:** $436.17
- **Actual Closing Price:** $449.72
- **RMSE:** 13.82
- **MAE:** 10.40
- **MAPE:** 3.60%
- **R² Score:** 0.978

The application also generates prediction graphs and training loss curves to help visualize forecasting accuracy and model convergence.

---

## Future Improvements

- Support additional stocks
- Multi-step future price prediction
- Real-time stock data integration
- Hyperparameter tuning
- Interactive charts using Plotly
- Compare LSTM with GRU and Transformer models
- Cloud deployment

---

## 👩‍💻 Author

**Akshaya Karanam**

GitHub: [@ayaronrios](https://github.com/ayaronrios)

---

## ⚠️ Disclaimer

This project is developed for educational purposes. The stock price predictions are based on historical market data and should **not** be considered financial or investment advice.

---

## Acknowledgments

- TensorFlow and Keras communities for deep learning tools
- Yahoo Finance (yFinance) for providing historical stock market data
- Open-source Python data science community

---

## 📄 License

This project is licensed under the MIT License.
