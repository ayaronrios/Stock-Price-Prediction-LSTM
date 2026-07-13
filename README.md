# 📈 Stock Price Prediction using Deep Learning (LSTM) and Python

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)

This project presents a web-based stock price prediction system that uses a Long Short-Term Memory (LSTM) neural network to forecast future stock closing prices from historical market data. The application provides an intuitive Flask-based interface for training the model, generating predictions, visualizing results, and evaluating model performance using standard regression metrics.

---

## 🎥 Demo Videos

Watch the application in action through the following demonstrations:

- ▶️ **Home Page Demo** – [homepage-demo.mp4](demo/homepage-demo.mp4)
- ▶️ **Prediction Results Demo** – [results-demo.mp4](demo/results-demo.mp4)

---

## 📷 Screenshots

### Home Page

<img width="930" height="1634" alt="Home Page" 
  src="https://github.com/user-attachments/assets/a2c622f5-97a4-4607-be16-667102bd5083" />


### Prediction Results

<img width="930" height="3424" alt="Prediction Results" 
  src="https://github.com/user-attachments/assets/e48957cc-66f3-411d-adc6-7d646f02ebf2" />

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| Model | Long Short-Term Memory (LSTM) Neural Network |
| Framework | TensorFlow & Flask |
| Dataset | Historical Stock Market Data |
| Supported Stocks | Apple (AAPL), Microsoft (MSFT), Tesla (TSLA) |
| Evaluation | RMSE, MAE, MAPE, R² Score |
| Outputs | Prediction Graphs, Training Loss Curves and Performance Metrics |

---

## 🛠️ Technologies Used

- Python
- Flask
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- yFinance

---

## 📂 Project Structure

```text
Stock-Price-Prediction-LSTM/
│
├── demo/
│   ├── homepage-demo.mp4
│   └── results-demo.mp4
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
├── app.py
├── train_model.py
├── model.py
├── predict.py
├── requirements.txt
├── README.md
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

Open your browser and visit:

```text
http://127.0.0.1:8000
```

---

## 📊 Workflow

1. Select a stock (Apple, Microsoft, or Tesla).
2. Train the LSTM model using historical market data.
3. Generate stock price predictions.
4. Evaluate model performance.
5. Display prediction charts and training loss graphs.

---

## 📈 Evaluation Metrics

The model performance is evaluated using:

- RMSE
- MAE
- MAPE
- R² Score

---

## 📊 Results

The developed LSTM model successfully predicts future stock closing prices using historical market data. The application evaluates model performance using RMSE, MAE, MAPE, and R² Score while providing prediction graphs and training loss curves to visualize forecasting accuracy and model convergence.

---

## 🔮 Future Enhancements

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

B.Tech – Computer Science and Engineering (Artificial Intelligence & Machine Learning)

GitHub: https://github.com/ayaronrios

---

## ⚠️ Disclaimer

This project is developed for educational purposes. The stock price predictions are based on historical market data and should **not** be considered financial or investment advice.

---

## 📄 License

This project is licensed under the MIT License.
