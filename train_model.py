#previously it defaulted to apple even after user choosing



# Download selected stock market data
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# UPDATE: Import datetime for recording experiment time
from datetime import datetime




stocks = {
    "1": ("Apple", "AAPL"),
    "2": ("Microsoft", "MSFT"),
    "3": ("Tesla", "TSLA"),
    "AAPL": ("Apple", "AAPL"),
    "MSFT": ("Microsoft", "MSFT"),
    "TSLA": ("Tesla", "TSLA")
}



def train_predict(ticker):
    ticker = ticker.upper()

    if ticker in stocks:
        company, ticker = stocks[ticker]
    else:
        raise ValueError("Invalid ticker")


    print(f"\nSelected: {company} ({ticker})")

    df = yf.download(
        ticker,
        start="2015-01-01",
        end="2026-01-01",
        auto_adjust=False
    )


    #check if download failed
    if df.empty:
        raise ValueError(f"No stock data found for {ticker}")
     
    print(df.head())
    #Keep only needed column:
    #We’ll predict Close price

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    #Now the model has an extra feature that captures day-to-day percentage changes.
    df['Daily_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)







    #DATA VISUALIZATION


    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], color='blue')
    # UPDATE: More descriptive graph title
    plt.title(f"{company} Historical Closing Prices (2015–2025)")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (USD)")
    plt.grid(True)
    plt.savefig(f"static/images/{ticker}_history.png")
    plt.close()






    #DATA PREPROCESSING
    from sklearn.preprocessing import MinMaxScaler


    # Split the dataframe into train and test
    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Scale using ONLY the training data
    scaler = MinMaxScaler(feature_range=(0,1))
    #Training → fit_transform
    train_scaled = scaler.fit_transform(train_df)
    #Testing → transform
    test_scaled = scaler.transform(test_df)

    scaled_data = np.vstack((train_scaled, test_scaled))







    # CREATE TRAINING DATASET (using all features)

    X = []
    y = []

    if ticker == "MSFT":
        WINDOW = 120
        epochs = 200
    elif ticker == "AAPL":
        WINDOW = 60
        epochs = 150
    else: #TSLA
        WINDOW = 60
        epochs = 100

    for i in range(WINDOW, len(scaled_data)):
        X.append(scaled_data[i-WINDOW:i])
        y.append(scaled_data[i, 3])

    X = np.array(X)
    y = np.array(y)

    #TRAIN/TEST SPLIT
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]








    #BUILD LSTM MODEL(DEEP LEARNING)
    from model import build_model
    model = build_model((X.shape[1], X.shape[2]))
    
    print("\nModel Architecture")
    model.summary()

    #EARLY STOPPING
    from tensorflow.keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )





    #TRAIN MODEL
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stop]
    )
    #(You can increase epochs later for better accuracy)
    print(f"Training stopped after {len(history.history['loss'])} epochs.")

    # UPDATE: Save training history for future analysis
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f"history/{ticker}_training_history.csv", index=False)





    #TRAINING LOSS GRAPH
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # UPDATE: Better title for model training graph
    plt.title(f"{company} LSTM Model Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.grid(True)
    plt.savefig(f"static/images/{ticker}_loss.png")
    plt.close()




    #PREDICTIONS
    predictions = model.predict(X_test)

    # Create dummy arrays with the same number of columns as the scaled data
    pred_dummy = np.zeros((len(predictions), scaled_data.shape[1]))
    pred_dummy[:, 3] = predictions[:, 0]

    actual_dummy = np.zeros((len(y_test), scaled_data.shape[1]))
    actual_dummy[:, 3] = y_test

    # Convert only the Close column back to original prices
    predictions = scaler.inverse_transform(pred_dummy)[:, 3]
    y_test_actual = scaler.inverse_transform(actual_dummy)[:, 3]

    # Save actual vs predicted prices to a CSV file
    results_df = pd.DataFrame({
        "Actual Price": y_test_actual,
        "Predicted Price": predictions
    })

    results_df.to_csv(f"predictions/{ticker}_predictions.csv", index=False)




    #VISUALIZATION OF PREDICTIONS
    plt.figure(figsize=(12,6))
    plt.plot(y_test_actual, color="blue", label="Actual Price")
    plt.plot(predictions, color="red", label="Predicted Price")
    # UPDATE: Better title for prediction graph
    plt.title(f"{company} Actual vs Predicted Closing Prices (LSTM)")
    plt.xlabel("Trading Days")
    plt.ylabel("Closing Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"static/images/{ticker}_prediction.png")
    plt.close()





    #MODEL EVALUATION
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    mae = mean_absolute_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)
    # UPDATE: Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100



    # Display results
    print("\n========== RESULTS ==========")
    print(f"Company : {company}")
    print(f"Ticker  : {ticker}")
    print(f"RMSE    : {rmse:.2f}")
    print(f"MAE     : {mae:.2f}")
    print(f"MAPE    : {mape:.2f}%")
    print(f"R²      : {r2:.3f}")
    print("=============================")
    # UPDATE: Display the latest predicted and actual closing prices
    print(f"\nLatest predicted closing price: ${predictions[-1]:.2f}")
    print(f"Latest actual closing price:    ${y_test_actual[-1]:.2f}")




    # Save results to a text file
    with open(f"results/{ticker}_results.txt", "w") as f:
        f.write(f"Date: {datetime.now()}\n") # UPDATE: Save the date and time of model execution
        f.write(f"Company: {company}\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")  # UPDATE
        f.write(f"R²: {r2:.3f}\n")
        # UPDATE: Save latest prediction
        f.write(f"Latest Predicted Price: ${predictions[-1]:.2f}\n")
        f.write(f"Latest Actual Price: ${y_test_actual[-1]:.2f}\n")
    





    #SAVE THE MODEL
    model.save(f"models/{ticker}_lstm_model.keras")

    #SAVE THE SCALER

    joblib.dump(scaler, f"scalers/{ticker}_scaler.pkl")





    # UPDATE: Notify the user that all project files have been saved successfully
    print("\n===================================")
    print("Project completed successfully!")
    print(f"Model saved as: models/{ticker}_lstm_model.keras")
    print(f"Scaler saved as: scalers/{ticker}_scaler.pkl")
    print(f"Results saved as: results/{ticker}_results.txt")
    print(f"Prediction data saved as: predictions/{ticker}_predictions.csv")
    print(f"Prediction graph saved as: static/images/{ticker}_prediction.png")
    print(f"Training loss graph saved as: static/images/{ticker}_loss.png")
    print(f"Training history saved as: history/{ticker}_training_history.csv")
    print("===================================")

    return {
        "company": company,
        "ticker": ticker,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "predicted": predictions[-1],
        "actual": y_test_actual[-1]
    }

    
if __name__ == "__main__":

    print("Choose a stock:")
    print("1. Apple (AAPL)")
    print("2. Microsoft (MSFT)")
    print("3. Tesla (TSLA)")

    while True:
        choice = input("Enter your choice: ").strip().upper()

        if choice in stocks:
            break

        print("Invalid choice. Try again.")

    train_predict(choice)  
    