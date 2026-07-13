# BUILD LSTM MODEL(DEEP LEARNING)
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Build the LSTM model
def build_model(input_shape):
    model = Sequential()

    # Define the input shape
    model.add(Input(shape=input_shape))

    # First LSTM layer
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    # Second LSTM layer
    model.add(LSTM(32))
    model.add(Dropout(0.2))

    # Output layers
    model.add(Dense(16))
    model.add(Dense(1))

    # Compile the model
    # Reduce the learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="mean_squared_error"
    )

    return model