from fastapi import FastAPI
from lstm_predictor import predict_bitcoin_price

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Selamat datang di API Prediksi Harga Bitcoin dengan LSTM"}

@app.get("/predict")
def predict():
    predicted_price = predict_bitcoin_price()
    return {"predicted_price": predicted_price}
