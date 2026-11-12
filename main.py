from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# 1. Mengambil model dan daftar fitur dari Hugging Face
REPO_ID = "pipin07/adidas-operating-profit-rf"

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="model_rf_adidas.joblib"
)

columns_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="kolom_fitur.joblib"
)

model = joblib.load(model_path)
kolom = joblib.load(columns_path)

# 2. Membuat aplikasi API
app = FastAPI(
    title="Adidas Operating Profit API",
    description="API untuk memprediksi Operating Profit transaksi Adidas."
)

# Mengizinkan website mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Menentukan data yang boleh dikirim pengguna
class Transaction(BaseModel):
    total_sales: float
    units_sold: int
    price_per_unit: float

    retailer: Literal[
        "Amazon", "Foot Locker", "Kohl's",
        "Sports Direct", "Walmart", "West Gear"
    ]

    region: Literal[
        "Midwest", "Northeast", "South",
        "Southeast", "West"
    ]

    product: Literal[
        "Men's Apparel",
        "Men's Athletic Footwear",
        "Men's Street Footwear",
        "Women's Apparel",
        "Women's Athletic Footwear",
        "Women's Street Footwear"
    ]

    sales_method: Literal["In-store", "Online", "Outlet"]


# 4. Halaman pemeriksaan API
@app.get("/")
def home():
    return {
        "message": "Adidas Operating Profit API is running!"
    }


# 5. Menjalankan prediksi
@app.post("/predict")
def predict(data: Transaction):

    # Membuat satu baris dengan 19 kolom
    X = pd.DataFrame(
        0,
        index=[0],
        columns=kolom
    )

    # Memasukkan angka dari pengguna
    X["Total Sales"] = data.total_sales
    X["Units Sold"] = data.units_sold
    X["Price per Unit"] = data.price_per_unit

    # Mengisi kategori Retailer
    if data.retailer != "Amazon":
        X[f"Retailer_{data.retailer}"] = 1

    # Mengisi kategori Region
    if data.region != "Midwest":
        X[f"Region_{data.region}"] = 1

    # Mengisi kategori Product
    if data.product != "Men's Apparel":
        X[f"Product_{data.product}"] = 1

    # Mengisi kategori Sales Method
    if data.sales_method != "In-store":
        X[f"Sales Method_{data.sales_method}"] = 1

    # Meminta Random Forest membuat prediksi
    prediction = model.predict(X)[0]

    return {
        "predicted_operating_profit": round(float(prediction), 2),
        "currency": "USD",
        "model": "Random Forest Regressor"
    }
