from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lstm_model import predict_price

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain React-mu kalau perlu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def get_prediction():
    price = predict_price()
    return {"predicted_price": price}
