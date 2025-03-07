import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi Seaborn
sns.set(style="whitegrid")

# Membaca data
days_df = pd.read_csv("day_cleaned.csv")
hours_df = pd.read_csv("hour_cleaned.csv")

# Konversi tanggal ke format datetime
days_df["tanggal"] = pd.to_datetime(days_df["tanggal"])
hours_df["tanggal"] = pd.to_datetime(hours_df["tanggal"])

# Sidebar - Menampilkan Judul & Gambar
st.sidebar.title("🚴‍♂️ Bike Sharing Dashboard")
st.sidebar.image("https://miro.medium.com/v2/resize:fit:2000/0*TZ0bsPAR7gGvOoEu", use_column_width=True)

# Sidebar - Rentang Tanggal
st.sidebar.header("Filter Rentang Waktu")
st.sidebar.caption("Pilih rentang waktu untuk melihat data penyewaan sepeda.")

# Mengambil nilai minimum dan maksimum tanggal dari dataset
min_date = days_df["tanggal"].min()
max_date = days_df["tanggal"].max()

# Membuat input rentang tanggal
start_date, end_date = st.sidebar.date_input(
    "Pilih Rentang Waktu",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Filter data berdasarkan tanggal yang dipilih
filtered_days_df = days_df[
    (days_df["tanggal"] >= pd.Timestamp(start_date)) & 
    (days_df["tanggal"] <= pd.Timestamp(end_date))
]

filtered_hours_df = hours_df[
    (hours_df["tanggal"] >= pd.Timestamp(start_date)) & 
    (hours_df["tanggal"] <= pd.Timestamp(end_date))
]

# Menampilkan rentang waktu yang dipilih di halaman utama
st.write(f"**Rentang Waktu Dipilih:** {start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}")

st.sidebar.markdown("© 2025 by JOSUA SIANTURI")

st.markdown(
    """
    <style>
    .metric-container {
        display: flex;
        justify-content: space-around;
        text-align: center;
        background-color: #181818;
        padding: 15px;
        border-radius: 10px;
    }
    .metric-box {
        color: white;
        font-size: 18px;
        font-weight: bold;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

total_penyewa_sepeda = filtered_days_df["total_penyewaan_sepeda"].sum()
total__penyewa_kasual = filtered_days_df["penyewa_kasual"].sum()
total_penyewa_terdaftar = filtered_days_df["penyewa_terdaftar"].sum()

st.markdown(
    f"""
    <div class="metric-container">
        <div class="metric-box">
            <div>Total Penyewa Sepeda</div>
            <div class="metric-value">{total_penyewa_sepeda:,}</div>
        </div>
        <div class="metric-box">
            <div>Total Penyewa Kasual</div>
            <div class="metric-value">{total__penyewa_kasual:,}</div>
        </div>
        <div class="metric-box">
            <div>Total Penyewa Register</div>
            <div class="metric-value">{total_penyewa_terdaftar:,}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.header("Analisis Jam Penyewaan Sepeda")

hourly_rentals = filtered_hours_df.groupby("jam")["total_penyewaan_sepeda"].sum().reset_index()
top_busy_hours = hourly_rentals.sort_values(by="total_penyewaan_sepeda", ascending=False).head(5)
least_busy_hours = hourly_rentals.sort_values(by="total_penyewaan_sepeda", ascending=True).head(5)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

sns.barplot(x="jam", y="total_penyewaan_sepeda", data=top_busy_hours, palette="Blues_r", ax=ax[0])
ax[0].set_title("Jam Sibuk - Penyewaan Sepeda Tertinggi", fontsize=16, fontweight='bold')

for bar in ax[0].patches:
    ax[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12, fontweight='bold')

sns.barplot(x="jam", y="total_penyewaan_sepeda", data=least_busy_hours, palette="Reds_r", ax=ax[1])
ax[1].set_title("Jam Sepi - Penyewaan Sepeda Terendah", fontsize=16, fontweight='bold')

for bar in ax[1].patches:
    ax[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12, fontweight='bold')

st.pyplot(fig)

st.header("Pola Penyewaan Sepeda Berdasarkan Musim")

season_rentals = filtered_days_df.groupby("musim")["total_penyewaan_sepeda"].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="musim", y="total_penyewaan_sepeda", data=season_rentals, palette="coolwarm", ax=ax)

for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12, fontweight='bold')

st.pyplot(fig)

st.header("Clustering Penyewaan Sepeda")

st.write("""
**Apa itu Clustering?**  
Clustering membantu mengelompokkan data berdasarkan pola yang mirip.  
Di sini, kita akan mengelompokkan data penyewaan sepeda berdasarkan jumlah penyewa(penyewa_terdaftar atau penyewa_kasual) dan faktor cuaca(suhu, kelembaban atau kecepatan_angin).
""")

# Pilih fitur untuk clustering
feature_options = ["penyewa_terdaftar", "penyewa_kasual", "suhu", "kelembaban", "kecepatan_angin"]
feature_x = st.selectbox("Pilih fitur untuk sumbu X:", feature_options, index=0)
feature_y = st.selectbox("Pilih fitur untuk sumbu Y:", feature_options, index=1)

# Persiapan data clustering
features = filtered_days_df[[feature_x, feature_y]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Menentukan jumlah cluster optimal (Elbow Method)
wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Visualisasi Elbow Method
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 6), wcss, marker="o", linestyle="--", color="b")
ax.set_title("Metode Elbow untuk Menentukan Jumlah Cluster", fontsize=14, fontweight="bold")
ax.set_xlabel("Jumlah Cluster")
ax.set_ylabel("WCSS")
ax.grid(True)
st.pyplot(fig)

# Menentukan jumlah cluster (default: 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
filtered_days_df["Cluster"] = kmeans.fit_predict(scaled_features)

# Visualisasi Clustering menggunakan Seaborn
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x=filtered_days_df[feature_x],
    y=filtered_days_df[feature_y],
    hue=filtered_days_df["Cluster"],
    palette="Set1",
    s=100,
    ax=ax
)
ax.set_title(f"Hasil Clustering Berdasarkan {feature_x} & {feature_y}", fontsize=14, fontweight="bold")
ax.set_xlabel(feature_x.capitalize())
ax.set_ylabel(feature_y.capitalize())
ax.legend(title="Cluster")
ax.grid(True)
st.pyplot(fig)

st.subheader("Interpretasi Hasil Clustering")

st.write("""
🔹 **Cluster 0**: Penyewaan rendah, kemungkinan terjadi saat suhu dingin atau kelembaban tinggi.  
🔹 **Cluster 1**: Penyewaan tinggi, dominasi oleh penyewa terdaftar. Biasanya terjadi saat suhu optimal.  
🔹 **Cluster 2**: Penyewaan sedang, bervariasi tergantung faktor cuaca seperti angin dan kelembaban.  
""")

st.write("**Kesimpulan**: Clustering ini membantu mengidentifikasi pola penyewaan berdasarkan cuaca dan jenis pengguna sepeda.")

st.header("Pola Penyewaan Berdasarkan Kondisi Cuaca")

weather_rentals = filtered_days_df.groupby("cuaca")["total_penyewaan_sepeda"].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="cuaca", y="total_penyewaan_sepeda", data=weather_rentals, palette="viridis", ax=ax)

for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12, fontweight='bold')

st.pyplot(fig)

st.header("Dinamika Penyewaan Sepeda: Pengguna Terdaftar vs Kasual Sepanjang Waktu")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(filtered_days_df['tanggal'], filtered_days_df['penyewa_terdaftar'], label="Penyewa Terdaftar", color='g', alpha=0.7)
ax.plot(filtered_days_df['tanggal'], filtered_days_df['penyewa_kasual'], label="Penyewa Kasual", color='r', alpha=0.7)

ax.legend()
st.pyplot(fig)
