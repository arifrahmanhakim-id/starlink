import streamlit as st
import pandas as pd
import re
import numpy as np
import plotly.express as px
import os
from textblob import TextBlob

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="Analisis Sentimen Ulasan Starlink",
    page_icon="🚀"
)

# ============================================================================
# FUNGSI-FUNGSI UTILITY
# ============================================================================

def clean_text(text):
    """Membersihkan teks ulasan"""
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = text.strip()
    return text

def predict_sentiment_textblob(text):
    """
    Prediksi sentimen menggunakan TextBlob (tidak memerlukan PyTorch)
    Polarity: -1 (negatif) hingga 1 (positif)
    """
    try:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return "Netral"
        
        blob = TextBlob(cleaned_text)
        polarity = blob.sentiment.polarity
        
        # Mapping polarity ke kategori sentimen
        if polarity > 0.1:
            return "Positif"
        elif polarity < -0.1:
            return "Negatif"
        else:
            return "Netral"
    
    except Exception as e:
        st.warning(f"Error dalam prediksi: {e}")
        return "Netral"

@st.cache_data
def load_data():
    """Memuat data CSV"""
    try:
        possible_paths = [
            'starlink_ulasan_dengan_sentimen.csv',
            './starlink_ulasan_dengan_sentimen.csv',
            '../starlink_ulasan_dengan_sentimen.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path, sep=';', encoding='utf-8')
                return df
        
        st.warning("File CSV tidak ditemukan. Menggunakan data demo.")
        return create_demo_data()
    
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return create_demo_data()

def create_demo_data():
    """Membuat data demo untuk demonstrasi"""
    demo_data = {
        'nama': ['User 1', 'User 2', 'User 3', 'User 4', 'User 5', 'User 6', 'User 7', 'User 8'],
        'ulasan': [
            'Layanan sangat baik dan cepat, saya sangat puas',
            'Jaringan sering putus-putus dan lambat',
            'Cukup memuaskan untuk penggunaan sehari-hari',
            'Sinyal lemot dan mahal, tidak sebanding',
            'Sangat membantu untuk daerah terpencil',
            'OK saja, tidak terlalu buruk',
            'Jaringan stabil dan cepat, sangat merekomendasikan',
            'Aplikasinya sulit digunakan dan sering error'
        ],
        'rating': [5, 2, 4, 1, 5, 3, 5, 2],
        'sentimen': ['Positif', 'Negatif', 'Netral', 'Negatif', 'Positif', 'Netral', 'Positif', 'Negatif']
    }
    return pd.DataFrame(demo_data)

# ============================================================================
# LOAD DATA
# ============================================================================
df_data = load_data()

# ============================================================================
# TAMPILAN UTAMA - HEADER
# ============================================================================

st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🚀 Dashboard Analisis Sentimen Ulasan Starlink</h1>
        <p style='font-size: 16px; color: gray;'>
            Analisis sentimen ulasan pengguna Starlink dari Google Play Store (Jan-Apr 2026)
        </p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - INFORMASI
# ============================================================================

with st.sidebar:
    st.header("ℹ️ Informasi Aplikasi")
    st.info("""
    Dashboard Analisis Sentimen Starlink menggunakan:
    - **Model**: TextBlob (Lightweight, Tanpa PyTorch)
    - **Sumber Data**: Google Play Store
    - **Periode**: Januari - April 2026
    - **Total Ulasan**: {} ulasan
    """.format(len(df_data)))
    
    st.divider()
    
    st.subheader("📊 Statistik Dataset")
    if not df_data.empty and 'sentimen' in df_data.columns:
        sentiment_counts = df_data['sentimen'].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Ulasan", len(df_data))
        with col2:
            st.metric("⭐ Positif", sentiment_counts.get('Positif', 0))
        with col3:
            st.metric("👎 Negatif", sentiment_counts.get('Negatif', 0))
        
        st.metric("⚪ Netral", sentiment_counts.get('Netral', 0))

# ============================================================================
# SECTION 1 - PREDIKSI SENTIMEN TUNGGAL
# ============================================================================

st.header("📝 Prediksi Sentimen Ulasan Tunggal")

col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "Masukkan ulasan Starlink di sini:",
        height=120,
        placeholder="Contoh: Layanan Starlink sangat cepat dan stabil, saya sangat puas!",
        value=""
    )

with col2:
    st.write("")
    st.write("")
    predict_button = st.button("🔍 Prediksi", use_container_width=True, type="primary")

if predict_button:
    if user_input.strip():
        with st.spinner("Menganalisis sentimen..."):
            sentiment = predict_sentiment_textblob(user_input)
        
        # Tampilkan hasil dengan styling
        if sentiment == "Positif":
            st.success(f"✅ **Sentimen: {sentiment}**", icon="😊")
            st.write("**Interpretasi**: Ulasan ini menunjukkan pandangan positif terhadap Starlink.")
        elif sentiment == "Negatif":
            st.error(f"❌ **Sentimen: {sentiment}**", icon="😠")
            st.write("**Interpretasi**: Ulasan ini menunjukkan keluhan atau pandangan negatif.")
        else:
            st.info(f"⚪ **Sentimen: {sentiment}**", icon="😐")
            st.write("**Interpretasi**: Ulasan ini netral atau berisi opini campuran.")
        
        # Tampilkan teks yang dibersihkan
        with st.expander("📖 Lihat teks yang dibersihkan"):
            st.text(clean_text(user_input))
    else:
        st.warning("⚠️ Mohon masukkan ulasan terlebih dahulu")

st.divider()

# ============================================================================
# SECTION 2 - VISUALISASI DATA HISTORIS
# ============================================================================

st.header("📊 Visualisasi Distribusi Sentimen")

if not df_data.empty and 'sentimen' in df_data.columns:
    # Hitung distribusi sentimen
    sentiment_counts = df_data['sentimen'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentimen', 'Jumlah Ulasan']
    
    # Urutkan agar konsisten
    sentiment_order = {'Positif': 0, 'Netral': 1, 'Negatif': 2}
    sentiment_counts['order'] = sentiment_counts['Sentimen'].map(sentiment_order)
    sentiment_counts = sentiment_counts.sort_values('order').drop('order', axis=1)
    
    # Definisikan warna
    color_map = {
        'Positif': '#2ecc71',
        'Negatif': '#e74c3c',
        'Netral': '#95a5a6'
    }
    
    # Buat visualisasi
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie Chart
        fig_pie = px.pie(
            sentiment_counts,
            values='Jumlah Ulasan',
            names='Sentimen',
            title='Distribusi Sentimen (Pie Chart)',
            color='Sentimen',
            color_discrete_map=color_map,
            hole=0.4
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar Chart
        fig_bar = px.bar(
            sentiment_counts,
            x='Sentimen',
            y='Jumlah Ulasan',
            title='Distribusi Sentimen (Bar Chart)',
            color='Sentimen',
            color_discrete_map=color_map,
            text='Jumlah Ulasan'
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Tabel ringkasan
    st.subheader("📋 Ringkasan Data")
    display_df = sentiment_counts.copy()
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Sentimen': st.column_config.TextColumn("Kategori Sentimen", width="medium"),
            'Jumlah Ulasan': st.column_config.NumberColumn("Jumlah", width="medium")
        }
    )
    
    # Statistik tambahan
    st.subheader("📈 Analisis Sentimen")
    col1, col2, col3 = st.columns(3)
    
    total = len(df_data)
    
    with col1:
        positif_count = sentiment_counts[sentiment_counts['Sentimen'] == 'Positif']['Jumlah Ulasan'].sum()
        positif_pct = (positif_count / total * 100) if total > 0 else 0
        st.metric("✅ Positif", f"{positif_count}", f"{positif_pct:.1f}%")
    
    with col2:
        netral_count = sentiment_counts[sentiment_counts['Sentimen'] == 'Netral']['Jumlah Ulasan'].sum()
        netral_pct = (netral_count / total * 100) if total > 0 else 0
        st.metric("⚪ Netral", f"{netral_count}", f"{netral_pct:.1f}%")
    
    with col3:
        negatif_count = sentiment_counts[sentiment_counts['Sentimen'] == 'Negatif']['Jumlah Ulasan'].sum()
        negatif_pct = (negatif_count / total * 100) if total > 0 else 0
        st.metric("❌ Negatif", f"{negatif_count}", f"{negatif_pct:.1f}%")

else:
    st.warning("❌ Data sentimen tidak tersedia untuk visualisasi")

st.divider()

# ============================================================================
# SECTION 3 - DATA TABLE
# ============================================================================

st.header("📑 Tabel Data Ulasan")

if not df_data.empty:
    # Limit tampilan
    max_rows = st.slider("Jumlah baris yang ditampilkan:", 5, min(100, len(df_data)), 20)
    
    # Filter sentimen (opsional)
    if 'sentimen' in df_data.columns:
        sentimen_filter = st.multiselect(
            "Filter berdasarkan sentimen:",
            options=df_data['sentimen'].unique(),
            default=df_data['sentimen'].unique()
        )
        df_filtered = df_data[df_data['sentimen'].isin(sentimen_filter)].head(max_rows)
    else:
        df_filtered = df_data.head(max_rows)
    
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)
else:
    st.info("Tidak ada data untuk ditampilkan")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
        <p>Dashboard Analisis Sentimen Starlink | Powered by Streamlit</p>
        <p>Data: Google Play Store Reviews (Jan - Apr 2026)</p>
        <p style='font-size: 10px;'>Sentiment Analysis using TextBlob Library</p>
    </div>
""", unsafe_allow_html=True)
