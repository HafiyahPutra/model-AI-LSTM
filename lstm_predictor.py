import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential

# Load model yang sudah disimpan
model = load_model("model.h5")

def predict_price():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    close_data = df[['close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_data)

    seq_length = 10
    last_sequence = scaled[-seq_length:]
    last_sequence = last_sequence.reshape((1, seq_length, 1))

    prediction = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    return round(float(predicted_price), 2)
