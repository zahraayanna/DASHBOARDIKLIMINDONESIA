import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ================== PAGE CONFIG =======================
st.set_page_config(page_title="🌧️ Analisis & Prediksi Iklim", layout="wide")

# ================== CUSTOM CSS (TAMPILAN BARU) =======================
st.markdown("""
    <style>
    /* Background utama dan sidebar */
    .main {
        background: #F7F5F2 !important;
    }
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E0DED8;
    }

    /* Judul utama */
    .main-title-wrap {
        background: linear-gradient(120deg, #0F766E, #22C55E);
        padding: 18px 26px;
        border-radius: 18px;
        color: white;
        text-align: left;
        box-shadow: 0 8px 18px rgba(0,0,0,0.08);
        margin-bottom: 8px;
    }
    .main-title-wrap h1 {
        font-size: 30px;
        margin: 0;
        font-weight: 800;
    }
    .main-title-wrap p {
        font-size: 14px;
        margin: 4px 0 0 0;
        opacity: 0.95;
    }

    /* Subjudul bagian */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #134E4A;
        margin-top: 10px;
        margin-bottom: 2px;
    }

    /* Card statistik baru */
    .metric-card {
        background-color: #FFFFFF;
        padding: 16px 18px;
        border-radius: 16px;
        box-shadow: 0px 4px 14px rgba(15,118,110,0.10);
        border: 1px solid rgba(15,118,110,0.18);
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-icon {
        font-size: 20px;
        margin-bottom: 2px;
    }

    /* Label selectbox di konten */
    .stSelectbox label {
        font-weight: 600;
        color: #0F172A;
    }

    /* Sedikit rapikan header sidebar */
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #0F766E;
    }
    </style>
""", unsafe_allow_html=True)

# ================== KONFIGURASI DASAR =======================
DEFAULT_REGION = "Papua Barat"
DEFAULT_FILE = "PAPUABARAT2.xlsx"

# ================== FUNGSI LOAD DATA =======================
@st.cache_data
def load_default_data():
    """Memuat data default Papua Barat dari file lokal."""
    df = pd.read_excel(DEFAULT_FILE, sheet_name="Data Harian - Table")
    df = df.loc[:, ~df.columns.duplicated()]
    if "kecepatan_angin" in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month
    return df

def load_uploaded_data(file):
    """Memuat data dari file yang di-upload user."""
    df = pd.read_excel(file, sheet_name="Data Harian - Table")
    df = df.loc[:, ~df.columns.duplicated()]
    if "kecepatan_angin" in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month
    return df

# ================== SIDEBAR: PILIH DATA =======================
st.sidebar.header("📂 Data Iklim")

uploaded_file = st.sidebar.file_uploader(
    "Upload data iklim wilayah lain (.xlsx)", type=["xlsx"]
)

if uploaded_file is not None:
    # pakai data upload
    df = load_uploaded_data(uploaded_file)
    # tebak nama wilayah dari nama file, tapi bisa diedit
    inferred_region = uploaded_file.name.replace(".xlsx", "").replace("_", " ").title()
    region_name = st.sidebar.text_input(
        "Nama wilayah untuk ditampilkan", value=inferred_region
    )
    source_info = f"Data dari file upload: **{uploaded_file.name}**"
else:
    # pakai data default Papua Barat
    df = load_default_data()
    region_name = DEFAULT_REGION
    source_info = f"Data default dari file lokal: **{DEFAULT_FILE}**"

# ================== TITLE (DINAMIS SESUAI WILAYAH) =======================
st.markdown(f"""
<div class="main-title-wrap">
    <h1>🌧️ Analisis & Prediksi Iklim {region_name}</h1>
    <p>Dashboard interaktif untuk menampilkan data historis dan prediksi jangka panjang
    berbagai variabel iklim di wilayah {region_name}.</p>
</div>
""", unsafe_allow_html=True)

st.caption(source_info)

# ================== SIDEBAR FILTER =======================
st.sidebar.header("🔍 Filter Data")

selected_year = st.sidebar.multiselect(
    "Pilih Tahun", sorted(df["Tahun"].unique()), default=df["Tahun"].unique()
)

selected_month = st.sidebar.multiselect(
    "Pilih Bulan", range(1, 13), default=range(1, 13)
)

df = df[df["Tahun"].isin(selected_year)]
df = df[df["Bulan"].isin(selected_month)]

# ================== VARIABLES =======================
possible_vars = ["Tn", "Tx", "Tavg", "kelembaban", "curah_hujan",
                 "matahari", "FF_X", "DDD_X"]
available_vars = [v for v in possible_vars if v in df.columns]

label = {
    "Tn": "Suhu Minimum (°C)",
    "Tx": "Suhu Maksimum (°C)",
    "Tavg": "Suhu Rata-rata (°C)",
    "kelembaban": "Kelembaban (%)",
    "curah_hujan": "Curah Hujan (mm)",
    "matahari": "Durasi Matahari (jam)",
    "FF_X": "Kecepatan Angin (m/s)",
    "DDD_X": "Arah Angin (°)"
}

if len(available_vars) == 0:
    st.error(
        "Tidak ada variabel iklim yang dikenali di dataset ini.\n\n"
        "Pastikan minimal ada salah satu dari kolom: "
        + ", ".join(possible_vars)
    )
    st.stop()

# ================== AGGREGASI BULANAN =======================
agg_dict = {v: "mean" for v in available_vars}
if "curah_hujan" in available_vars:
    agg_dict["curah_hujan"] = "sum"

monthly = df.groupby(["Tahun", "Bulan"]).agg(agg_dict).reset_index()

# ================== TRAIN MODEL =======================
models = {}
metrics = {}

for v in available_vars:
    X = monthly[["Tahun", "Bulan"]]
    y = monthly[v]

    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=180, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xts)

    models[v] = model
    metrics[v] = (mean_squared_error(yts, pred) ** 0.5, r2_score(yts, pred))

# ================== CARD STATISTIK =======================
st.markdown('<div class="section-title">📊 Ringkasan Data</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📏</div>
        <div class="metric-label">Data Historis</div>
        <div class="metric-value">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📅</div>
        <div class="metric-label">Rentang Tahun</div>
        <div class="metric-value">{df['Tahun'].min()} - {df['Tahun'].max()}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🌡️</div>
        <div class="metric-label">Variabel Iklim</div>
        <div class="metric-value">{len(available_vars)}</div>
    </div>
    """, unsafe_allow_html=True)

# ================== GRAFIK HISTORIS =======================
st.markdown('<div class="section-title">📈 Tren Data Historis</div>', unsafe_allow_html=True)

var_plot = st.selectbox("Pilih Variabel", [label[v] for v in available_vars])
key = [k for k, v in label.items() if v == var_plot][0]

monthly["Tanggal"] = pd.to_datetime(
    monthly["Tahun"].astype(str) + "-" +
    monthly["Bulan"].astype(str) + "-01"
)

fig1 = px.line(
    monthly,
    x="Tanggal",
    y=key,
    markers=True,
    title=f"{var_plot} di {region_name}",
    template="plotly_white",
    color_discrete_sequence=["#0F766E"]  # hijau teal
)

st.plotly_chart(fig1, use_container_width=True)

# ================== PREDIKSI =======================
future = pd.DataFrame(
    [(y, m) for y in range(2025, 2076) for m in range(1, 13)],
    columns=["Tahun", "Bulan"]
)

for v in available_vars:
    future[f"Pred_{v}"] = models[v].predict(future[["Tahun", "Bulan"]])

st.markdown('<div class="section-title">🔮 Prediksi 2025–2075</div>', unsafe_allow_html=True)

var_pred = st.selectbox("Pilih Variabel Prediksi", [label[v] for v in available_vars])
key2 = [k for k, v in label.items() if v == var_pred][0]

future["Tanggal"] = pd.to_datetime(
    future["Tahun"].astype(str) + "-" +
    future["Bulan"].astype(str) + "-01"
)

fig2 = px.line(
    future,
    x="Tanggal",
    y=f"Pred_{key2}",
    title=f"Prediksi {var_pred} di {region_name}",
    template="plotly_white",
    color_discrete_sequence=["#22C55E"]  # hijau lebih terang
)

st.plotly_chart(fig2, use_container_width=True)

# ================== DOWNLOAD =======================
csv = future.to_csv(index=False).encode("utf8")

# bikin nama file sesuai nama wilayah (lowercase + underscore)
file_slug = region_name.lower().replace(" ", "_")
download_name = f"prediksi_{file_slug}.csv"

st.download_button(
    "📥 Download Dataset Prediksi",
    data=csv,
    file_name=download_name,
    mime="text/csv"
)
