import joblib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Any, Dict

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Loan Prediction (RF)", layout="centered")

MODEL_PATH = "logistic_regression_model.pkl"
ENCODER_PATH = "scaler_model.pkl"
DEFAULT_UNKNOWN = 0.0  # samakan dengan training kamu (kalau mau aman: -1)

# =========================
# LOAD ARTIFACTS (cached)
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)  # dict: {col_name: mapping_dict}
    return model, encoders

def encode_column(df: pd.DataFrame, column: str, mapping: Dict[str, float], default: float):
    df[column] = (
        df[column]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(mapping)
        .fillna(default)
        .astype(float)
    )
    return df

def prepare_input(user_features: Dict[str, Any], encoders: Dict[str, Dict[str, float]]):
    df = pd.DataFrame([user_features])

    # Pastikan kolom-kolom encoder ada
    for col, mapping in encoders.items():
        if col not in df.columns:
            df[col] = default = DEFAULT_UNKNOWN
        df = encode_column(df, col, mapping, DEFAULT_UNKNOWN)

    return df

# =========================
# UI
# =========================
st.title("Loan Approval Prediction (Random Forest)")
st.caption("Streamlit app: load encoder (.pkl) + model RF (.pkl). Tidak fit ulang.")

# Load
try:
    model, encoders = load_artifacts()
except Exception as e:
    st.error(f"Gagal load model/encoder: {e}")
    st.stop()

st.subheader("Input Features")

# Buat input form berbasis mapping yang ada (kolom kategorikal yang dimapping)
# Angka/numerik lain bisa kamu tambah manual sesuai dataset (contoh totalDTI, age, dll)
with st.form("predict_form"):
    user_features: Dict[str, Any] = {}

    # contoh numeric fields (silakan tambahkan sesuai training kamu)
    user_features["age"] = st.number_input("age", min_value=0, max_value=120, value=35)
    user_features["totalDTI"] = st.number_input("totalDTI", min_value=0.0, max_value=10.0, value=0.34)

    # categorical mapped fields: pilih dari key mapping
    # (supaya tidak ada unseen category, kecuali kamu isi manual)
    for col, mapping in encoders.items():
        options = sorted(list(mapping.keys()))
        # tampilkan option asli (string) untuk dipilih
        user_features[col] = st.selectbox(col, options=options, index=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        X = prepare_input(user_features, encoders)

        pred = model.predict(X)[0]
        label = "Approved" if int(pred) == 1 else "Disapproved"

        proba = None
        if hasattr(model, "predict_proba"):
            # asumsi binary: kolom ke-1 adalah prob positif
            proba = float(model.predict_proba(X)[0][1])

        st.success(f"Prediction: **{label}**")
        if proba is not None:
            st.info(f"Probability Approved: **{proba:.4f}**")

        with st.expander("Show processed input (after encoding)"):
            st.dataframe(X)

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()
st.caption("Tip: kalau takut unseen category di dunia nyata, ubah DEFAULT_UNKNOWN menjadi -1 dan retrain model.")
