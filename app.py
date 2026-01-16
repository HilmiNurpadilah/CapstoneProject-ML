# app.py
# Streamlit app: Prediksi Keterlambatan Pengiriman E-Commerce
# Jalankan: streamlit run app.py
#
# Pastikan file model ada di folder yang sama (default):
#   best_model_shipping_delay.joblib
# atau ubah MODEL_PATH di bawah.

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Prediksi Keterlambatan Pengiriman",
    page_icon="🚚",
    layout="wide",
)

MODEL_PATH = "model/best_model_shipping_delay.joblib"

# -----------------------------
# CSS (estetik + UX)
# -----------------------------
st.markdown(
    """
<style>
/* Page spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Header card */
.hero {
  background: linear-gradient(135deg, rgba(35, 99, 255, 0.12), rgba(0, 204, 150, 0.10));
  border: 1px solid rgba(120,120,120,0.18);
  border-radius: 18px;
  padding: 18px 18px;
}

/* Soft cards */
.card {
  border: 1px solid rgba(120,120,120,0.18);
  border-radius: 16px;
  padding: 16px;
  background: rgba(255,255,255,0.03);
}

/* Small help text */
.muted { opacity: 0.75; font-size: 0.92rem; }

/* Big KPI */
.kpi {
  font-size: 2.0rem;
  font-weight: 750;
  line-height: 1.1;
}
.kpi-sub { opacity: 0.8; font-size: 0.95rem; }

/* Button spacing */
.stButton>button { border-radius: 12px; padding: 0.6rem 1rem; }

/* Improve table look */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Badge */
.badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  border: 1px solid rgba(120,120,120,0.25);
  background: rgba(120,120,120,0.10);
  font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Utilities
# -----------------------------
FEATURE_COLS = [
    "Warehouse_block",
    "Mode_of_Shipment",
    "Customer_care_calls",
    "Customer_rating",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Product_importance",
    "Gender",
    "Discount_offered",
    "Weight_in_gms",
]

CATEGORICAL_OPTIONS = {
    "Warehouse_block": ["A", "B", "C", "D", "E", "F"],
    "Mode_of_Shipment": ["Flight", "Ship", "Road"],
    "Product_importance": ["low", "medium", "high"],
    "Gender": ["F", "M"],
}

NUMERIC_HINTS = {
    "Customer_care_calls": (0, 10, 4),
    "Customer_rating": (1, 5, 3),
    "Cost_of_the_Product": (1, 1000, 200),
    "Prior_purchases": (0, 50, 3),
    "Discount_offered": (0, 100, 10),
    "Weight_in_gms": (1, 50000, 3000),
}


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def make_single_row_input(
    warehouse_block: str,
    mode_of_shipment: str,
    customer_care_calls: int,
    customer_rating: int,
    cost_of_product: int,
    prior_purchases: int,
    product_importance: str,
    gender: str,
    discount_offered: int,
    weight_in_gms: int,
) -> pd.DataFrame:
    row = {
        "Warehouse_block": warehouse_block,
        "Mode_of_Shipment": mode_of_shipment,
        "Customer_care_calls": customer_care_calls,
        "Customer_rating": customer_rating,
        "Cost_of_the_Product": cost_of_product,
        "Prior_purchases": prior_purchases,
        "Product_importance": product_importance,
        "Gender": gender,
        "Discount_offered": discount_offered,
        "Weight_in_gms": weight_in_gms,
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


def predict_with_threshold(model, X: pd.DataFrame, threshold: float = 0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba


def nice_label(pred_int: int) -> str:
    return "Terlambat" if pred_int == 1 else "Tepat Waktu"


# -----------------------------
# Sidebar: Model + Settings
# -----------------------------
st.sidebar.markdown("### ⚙️ Pengaturan")

uploaded_model = st.sidebar.file_uploader(
    "Upload model (.joblib) (opsional)",
    type=["joblib"],
    help="Kalau kamu mau, upload model hasil training. Kalau tidak, app akan memakai file default di folder yang sama.",
)

threshold = st.sidebar.slider(
    "Ambang Prediksi (threshold)",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Semakin besar threshold → model lebih 'ketat' menyatakan TERLAMBAT (precision naik, recall turun).",
)

show_debug = st.sidebar.toggle("Tampilkan detail input", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
<div class="muted">
<b>Catatan:</b><br/>
Target model: <span class="badge">Reached.on.Time_Y.N</span><br/>
0 = Tepat Waktu, 1 = Terlambat
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Load model
# -----------------------------
model = None
model_source = None

try:
    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        model_source = "Uploaded model"
    else:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            model_source = f"Default: {MODEL_PATH}"
        else:
            model_source = None
except Exception as e:
    st.sidebar.error(f"Gagal load model: {e}")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="hero">
  <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:14px; flex-wrap:wrap;">
    <div>
      <div style="font-size:0.95rem; opacity:0.85;">🚚 Capstone Project • Prediksi Keterlambatan Pengiriman</div>
      <div style="font-size:1.55rem; font-weight:800; margin-top:4px;">Perbandingan Logistic Regression vs XGBoost</div>
      <div class="muted" style="margin-top:6px;">
        Masukkan fitur pengiriman untuk memprediksi apakah pengiriman <b>berpotensi terlambat</b>.
        Kamu juga bisa upload CSV untuk prediksi batch.
      </div>
    </div>
    <div>
      <div class="badge">Model: {}</div>
    </div>
  </div>
</div>
""".format(model_source if model_source else "Belum ditemukan"),
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "Model belum bisa dipakai. Pastikan file model ada (best_model_shipping_delay.joblib) "
        "atau upload model .joblib dari sidebar."
    )
    st.stop()

st.write("")

# -----------------------------
# Tabs
# -----------------------------
tab_single, tab_batch, tab_about = st.tabs(["🔮 Prediksi Cepat", "📦 Prediksi Batch (CSV)", "ℹ️ Tentang & Tips"])

# -----------------------------
# Single prediction
# -----------------------------
with tab_single:
    c1, c2 = st.columns([1.05, 0.95], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧾 Input Data Pengiriman")
        st.caption("Isi data berikut. Setelah itu klik **Prediksi**.")

        with st.form("single_form", clear_on_submit=False):
            colA, colB = st.columns(2)

            with colA:
                warehouse_block = st.selectbox(
                    "Warehouse_block",
                    CATEGORICAL_OPTIONS["Warehouse_block"],
                    index=0,
                    help="Kode blok gudang (A–F).",
                )
                mode_of_shipment = st.selectbox(
                    "Mode_of_Shipment",
                    CATEGORICAL_OPTIONS["Mode_of_Shipment"],
                    index=0,
                    help="Metode pengiriman.",
                )
                product_importance = st.selectbox(
                    "Product_importance",
                    CATEGORICAL_OPTIONS["Product_importance"],
                    index=1,
                    help="Tingkat kepentingan produk.",
                )
                gender = st.selectbox(
                    "Gender",
                    CATEGORICAL_OPTIONS["Gender"],
                    index=0,
                    help="Gender pelanggan (sesuai dataset).",
                )

            with colB:
                customer_care_calls = st.number_input(
                    "Customer_care_calls",
                    min_value=NUMERIC_HINTS["Customer_care_calls"][0],
                    max_value=NUMERIC_HINTS["Customer_care_calls"][1],
                    value=NUMERIC_HINTS["Customer_care_calls"][2],
                    step=1,
                    help="Jumlah panggilan ke customer care.",
                )
                customer_rating = st.number_input(
                    "Customer_rating",
                    min_value=NUMERIC_HINTS["Customer_rating"][0],
                    max_value=NUMERIC_HINTS["Customer_rating"][1],
                    value=NUMERIC_HINTS["Customer_rating"][2],
                    step=1,
                    help="Rating pelanggan (1–5).",
                )
                cost_of_product = st.number_input(
                    "Cost_of_the_Product",
                    min_value=NUMERIC_HINTS["Cost_of_the_Product"][0],
                    max_value=NUMERIC_HINTS["Cost_of_the_Product"][1],
                    value=NUMERIC_HINTS["Cost_of_the_Product"][2],
                    step=1,
                    help="Harga produk (sesuaikan skala dataset).",
                )
                prior_purchases = st.number_input(
                    "Prior_purchases",
                    min_value=NUMERIC_HINTS["Prior_purchases"][0],
                    max_value=NUMERIC_HINTS["Prior_purchases"][1],
                    value=NUMERIC_HINTS["Prior_purchases"][2],
                    step=1,
                    help="Jumlah pembelian sebelumnya.",
                )
                discount_offered = st.number_input(
                    "Discount_offered",
                    min_value=NUMERIC_HINTS["Discount_offered"][0],
                    max_value=NUMERIC_HINTS["Discount_offered"][1],
                    value=NUMERIC_HINTS["Discount_offered"][2],
                    step=1,
                    help="Diskon yang diberikan (0–100).",
                )
                weight_in_gms = st.number_input(
                    "Weight_in_gms",
                    min_value=NUMERIC_HINTS["Weight_in_gms"][0],
                    max_value=NUMERIC_HINTS["Weight_in_gms"][1],
                    value=NUMERIC_HINTS["Weight_in_gms"][2],
                    step=50,
                    help="Berat produk dalam gram.",
                )

            submit = st.form_submit_button("🚀 Prediksi Sekarang")

        st.markdown("</div>", unsafe_allow_html=True)

        X_one = make_single_row_input(
            warehouse_block=warehouse_block,
            mode_of_shipment=mode_of_shipment,
            customer_care_calls=int(customer_care_calls),
            customer_rating=int(customer_rating),
            cost_of_product=int(cost_of_product),
            prior_purchases=int(prior_purchases),
            product_importance=product_importance,
            gender=gender,
            discount_offered=int(discount_offered),
            weight_in_gms=int(weight_in_gms),
        )

        if show_debug:
            st.markdown("#### 🔎 Data yang dikirim ke model")
            st.dataframe(X_one, use_container_width=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Hasil Prediksi")
        st.caption("Hasil muncul setelah kamu klik **Prediksi Sekarang**.")

        if "last_pred" not in st.session_state:
            st.session_state.last_pred = None

        if submit:
            pred, proba = predict_with_threshold(model, X_one, threshold=threshold)
            pred_int = int(pred[0])
            p = float(proba[0])

            st.session_state.last_pred = (pred_int, p)

        if st.session_state.last_pred is None:
            st.info("Silakan isi input di kiri, lalu klik **Prediksi Sekarang**.")
        else:
            pred_int, p = st.session_state.last_pred
            label = nice_label(pred_int)

            # KPI
            st.markdown(
                f"""
<div style="display:flex; gap:14px; flex-wrap:wrap;">
  <div style="flex:1; min-width:240px;">
    <div class="kpi">{label}</div>
    <div class="kpi-sub">Dengan threshold <b>{threshold:.2f}</b></div>
  </div>
  <div style="flex:1; min-width:240px;">
    <div class="kpi">{p*100:.1f}%</div>
    <div class="kpi-sub">Probabilitas <b>Terlambat</b></div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.write("")
            st.progress(min(max(p, 0.0), 1.0))

            if pred_int == 1:
                st.warning(
                    "⚠️ Pengiriman diprediksi **TERLAMBAT**. Pertimbangkan tindakan: prioritaskan proses, cek kapasitas gudang, atau ubah metode pengiriman."
                )
            else:
                st.success(
                    "✅ Pengiriman diprediksi **TEPAT WAKTU**. Tetap pantau faktor risiko seperti diskon tinggi/berat besar."
                )

            st.markdown("---")
            st.markdown("#### 🧠 Interpretasi singkat")
            st.write(
                "Probabilitas menunjukkan seberapa besar model yakin pengiriman akan terlambat. "
                "Jika kamu ingin model lebih hati-hati menyatakan terlambat (lebih presisi), naikkan threshold."
            )

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Batch prediction
# -----------------------------
with tab_batch:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📦 Prediksi Batch dari CSV")
    st.caption(
        "Upload file CSV yang berisi kolom fitur. Minimal harus punya kolom: "
        + ", ".join(FEATURE_COLS)
    )

    up = st.file_uploader("Upload CSV", type=["csv"])

    if up is not None:
        try:
            df_up = pd.read_csv(up)
            st.write("Preview data:")
            st.dataframe(df_up.head(10), use_container_width=True)

            missing = [c for c in FEATURE_COLS if c not in df_up.columns]
            if missing:
                st.error(
                    "CSV kamu belum lengkap. Kolom yang kurang: " + ", ".join(missing)
                )
            else:
                Xb = df_up[FEATURE_COLS].copy()

                # Prediksi
                pred_b, proba_b = predict_with_threshold(model, Xb, threshold=threshold)

                out = df_up.copy()
                out["pred_label"] = [nice_label(int(v)) for v in pred_b]
                out["proba_late"] = np.round(proba_b, 6)

                # Ringkasan
                late_count = int((pred_b == 1).sum())
                ontime_count = int((pred_b == 0).sum())
                st.markdown("#### Ringkasan")
                colx, coly, colz = st.columns(3)
                colx.metric("Total baris", len(out))
                coly.metric("Diprediksi Terlambat", late_count)
                colz.metric("Diprediksi Tepat Waktu", ontime_count)

                st.write("Hasil (dengan kolom prediksi):")
                st.dataframe(out.head(50), use_container_width=True)

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download hasil prediksi (CSV)",
                    data=csv_bytes,
                    file_name="prediksi_keterlambatan_pengiriman.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Gagal memproses CSV: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# About & Tips
# -----------------------------
with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.write(
        "Aplikasi ini memuat model machine learning (hasil training notebook) untuk memprediksi "
        "keterlambatan pengiriman e-commerce. Kamu bisa melakukan prediksi satu data (form) atau batch (CSV)."
    )

    st.markdown("#### 🧩 Format CSV Batch (kolom wajib)")
    st.code(", ".join(FEATURE_COLS))

    st.markdown("</div>", unsafe_allow_html=True)
