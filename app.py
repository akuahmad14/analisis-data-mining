import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, classification_report
)

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Smartphone AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CONSTANTS & CSS
# =========================
DANGER  = "#f87171"
SUCCESS = "#34d399"
PRIMARY = "#60a5fa"
ACCENT  = "#a78bfa"

def load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e5e7eb", family="Poppins"),
    margin=dict(l=10, r=10, t=50, b=10),
)

FEATURES = [
    'daily_screen_time_hours',
    'social_media_hours',
    'gaming_hours',
    'work_study_hours',
    'sleep_hours',
    'notifications_per_day',
    'app_opens_per_day',
    'weekend_screen_time',
]

FEATURE_LABELS = {
    'daily_screen_time_hours' : 'Screen Time Harian (jam)',
    'social_media_hours'      : 'Media Sosial (jam)',
    'gaming_hours'            : 'Gaming (jam)',
    'work_study_hours'        : 'Kerja / Belajar (jam)',
    'sleep_hours'             : 'Jam Tidur (jam)',
    'notifications_per_day'   : 'Notifikasi / Hari',
    'app_opens_per_day'       : 'Buka Aplikasi / Hari',
    'weekend_screen_time'     : 'Screen Time Weekend (jam)',
}

# =========================
# HELPERS
# =========================
def hero(icon, title, subtitle):
    st.markdown(f"""
    <div class="app-hero">
        <div class="icon-badge">{icon}</div>
        <div><h1>{title}</h1><p>{subtitle}</p></div>
    </div>""", unsafe_allow_html=True)

def section_title(text):
    st.markdown(f"""
    <div class="section-title"><div class="bar"></div><h3>{text}</h3></div>
    """, unsafe_allow_html=True)

def kpi_card(icon, label, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>""", unsafe_allow_html=True)

def chart_wrap(fig, height=None):
    if height:
        fig.update_layout(height=height)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# LOAD DATA & TRAIN
# =========================
@st.cache_data
def load_and_train():
    df = pd.read_csv("dataset/Smartphone_Usage_And_Addiction_Analysis.csv")
    df = df.dropna(subset=['addicted_label'])

    X = df[FEATURES]
    y = df['addicted_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mdl = GaussianNB()
    mdl.fit(X_train, y_train)

    y_pred = mdl.predict(X_test)

    return df, X, y, X_train, X_test, y_train, y_test, mdl, y_pred

df, X, y, X_train, X_test, y_train, y_test, model, y_pred = load_and_train()

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

# =========================
# SIDEBAR
# =========================
if "menu" not in st.session_state:
    st.session_state.menu = "Dashboard"

st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="logo">📱</div>
    <div>
        <div class="title">Smartphone AI</div>
        <div class="subtitle">Addiction Analytics Suite</div>
    </div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div class="sidebar-stats">
    <div class="sidebar-stat-chip">
        <div class="v">{acc:.0%}</div>
        <div class="l">Accuracy</div>
    </div>
    <div class="sidebar-stat-chip">
        <div class="v">{len(df):,}</div>
        <div class="l">Records</div>
    </div>
</div>""", unsafe_allow_html=True)

NAV_ITEMS = [
    ("Dashboard",             "📊"),
    ("Dataset Explorer",      "🗂️"),
    ("Data Processing",       "⚙️"),
    ("Prior Probability",     "📐"),
    ("Naive Bayes Classifier","🤖"),
    ("Visualisasi Distribusi","📈"),
    ("Analisis Karakteristik","🔎"),
    ("Hasil Rekomendasi",     "🔍"),
    ("Download Hasil",        "📄"),
]

st.sidebar.markdown('<div class="sidebar-nav-label">Menu</div>', unsafe_allow_html=True)
for key, icon in NAV_ITEMS:
    is_active = st.session_state.menu == key
    if st.sidebar.button(
        f"{icon}   {key}",
        key=f"nav_{key}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.menu = key
        st.rerun()

st.sidebar.markdown("""
<div class="sidebar-footer">
    Made with ❤️ using Streamlit & Scikit-learn<br>
    © 2026 Smartphone AI
</div>""", unsafe_allow_html=True)

menu = st.session_state.menu

# ==================================================
# 1. DASHBOARD
# ==================================================
if menu == "Dashboard":
    hero("📊", "SmartPhone Addiction Dashboard", "Ringkasan model & data secara sekilas")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("🎯", "Accuracy",  f"{acc:.0%}")
    with c2: kpi_card("🗂️", "Total Data", f"{len(df):,}")
    with c3: kpi_card("🧬", "Fitur",     f"{len(FEATURES)}")
    with c4:
        high_pct = (y == 1).sum() / len(y) * 100
        kpi_card("⚠️", "High Risk", f"{high_pct:.1f}%")

    section_title("AI Insight")
    high = int((y == 1).sum()); low = int((y == 0).sum())
    pct  = high / (high + low) * 100
    msg  = (f"🚨 Mayoritas pengguna ({pct:.1f}%) menunjukkan pola penggunaan tinggi yang mengarah "
            f"pada potensi kecanduan digital." if high > low else
            f"✅ Sebagian besar pengguna ({100-pct:.1f}%) masih berada pada kategori penggunaan smartphone yang aman.")
    st.markdown(f'<div class="insight-card">{msg}</div>', unsafe_allow_html=True)

    section_title("Distribusi Kecanduan Smartphone")
    col1, col2 = st.columns([1.3, 1])

    counts = df['addicted_label'].value_counts().rename({0: "Low Risk", 1: "High Risk"})

    with col1:
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                     color_discrete_map={"Low Risk": SUCCESS, "High Risk": DANGER},
                     labels={"x": "Kategori", "y": "Jumlah"}, text=counts.values)
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="Jumlah per Kategori")
        chart_wrap(fig)

    with col2:
        fig2 = px.pie(names=counts.index, values=counts.values, color=counts.index,
                      color_discrete_map={"Low Risk": SUCCESS, "High Risk": DANGER}, hole=0.55)
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(**PLOTLY_LAYOUT, title="Proporsi", showlegend=False)
        chart_wrap(fig2)

    section_title("Rata-rata Fitur per Kategori")
    avg = df.groupby('addicted_label')[FEATURES].mean().rename(index={0:"Low Risk",1:"High Risk"})
    avg_m = avg.reset_index().melt(id_vars='addicted_label', var_name='Fitur', value_name='Nilai')
    avg_m['Fitur'] = avg_m['Fitur'].map(FEATURE_LABELS)
    fig3 = px.bar(avg_m, x="Fitur", y="Nilai", color="addicted_label", barmode="group",
                  color_discrete_map={"Low Risk": SUCCESS, "High Risk": DANGER})
    fig3.update_layout(**PLOTLY_LAYOUT, legend_title_text="", xaxis_tickangle=-25)
    chart_wrap(fig3)

# ==================================================
# 2. DATASET EXPLORER
# ==================================================
elif menu == "Dataset Explorer":
    hero("🗂️", "Dataset Explorer", "Eksplorasi isi dataset secara interaktif")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("📋", "Total Baris",   f"{len(df):,}")
    with c2: kpi_card("🧬", "Total Kolom",   f"{len(df.columns)}")
    with c3: kpi_card("❌", "Missing Values", f"{df.isnull().sum().sum()}")
    with c4: kpi_card("🏷️", "Kelas Target",  "2 (0 & 1)")

    section_title("Preview Data")
    n = st.slider("Jumlah baris yang ditampilkan", 5, 100, 10)
    st.dataframe(df[FEATURES + ['addicted_label']].head(n), use_container_width=True)

    section_title("Statistik Deskriptif")
    desc = df[FEATURES].describe().T.rename(columns=str)
    desc.index = [FEATURE_LABELS[f] for f in desc.index]
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

    section_title("Missing Values per Kolom")
    mv = df[FEATURES + ['addicted_label']].isnull().sum().reset_index()
    mv.columns = ["Kolom", "Jumlah Missing"]
    mv["Kolom"] = mv["Kolom"].map({**FEATURE_LABELS, 'addicted_label': 'addicted_label'})
    fig = px.bar(mv, x="Kolom", y="Jumlah Missing",
                 color="Jumlah Missing", color_continuous_scale="Reds")
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, xaxis_tickangle=-25)
    chart_wrap(fig)

    section_title("Korelasi Antar Fitur")
    corr = df[FEATURES].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         x=[FEATURE_LABELS[f] for f in FEATURES],
                         y=[FEATURE_LABELS[f] for f in FEATURES])
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=500)
    chart_wrap(fig_corr, height=500)

# ==================================================
# 3. DATA PROCESSING
# ==================================================
elif menu == "Data Processing":
    hero("⚙️", "Data Processing", "Tahapan preprocessing sebelum data dilatih ke model")

    section_title("Alur Data Processing")
    st.markdown("""
    <div class="insight-card">
    Berikut tahapan yang dilakukan sebelum data masuk ke model Naive Bayes:<br><br>
    <b>1. Load Dataset</b> → Baca file CSV<br>
    <b>2. Drop Missing Values</b> → Hapus baris yang kolom <code>addicted_label</code>-nya kosong<br>
    <b>3. Seleksi Fitur</b> → Hanya ambil 8 fitur numerik yang relevan<br>
    <b>4. Split Data</b> → 80% Training, 20% Testing (random_state=42)<br>
    <b>5. Training</b> → Model Gaussian Naive Bayes dilatih dengan data train
    </div>
    """, unsafe_allow_html=True)

    section_title("Sebelum & Sesudah Drop Missing Values")
    col1, col2 = st.columns(2)
    df_raw_full = pd.read_csv("dataset/Smartphone_Usage_And_Addiction_Analysis.csv")
    with col1:
        kpi_card("📋", "Sebelum (total baris)", f"{len(df_raw_full):,}")
    with col2:
        kpi_card("✅", "Sesudah (baris valid)", f"{len(df):,}")

    section_title("Fitur yang Digunakan")
    feat_df = pd.DataFrame({
        "No": range(1, len(FEATURES)+1),
        "Nama Kolom": FEATURES,
        "Label": [FEATURE_LABELS[f] for f in FEATURES],
        "Tipe Data": ["Numerik (float)"] * 6 + ["Numerik (int)"] * 2,
    })
    st.dataframe(feat_df, hide_index=True, use_container_width=True)

    section_title("Pembagian Data Train & Test")
    total = len(df)
    n_train = len(X_train)
    n_test  = len(X_test)
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("📦", "Total Data", f"{total:,}")
    with c2: kpi_card("🏋️", "Data Train (80%)", f"{n_train:,}")
    with c3: kpi_card("🧪", "Data Test (20%)", f"{n_test:,}")

    fig_split = px.pie(
        names=["Train (80%)", "Test (20%)"],
        values=[n_train, n_test],
        color_discrete_sequence=[PRIMARY, ACCENT],
        hole=0.5,
    )
    fig_split.update_layout(**PLOTLY_LAYOUT, showlegend=True)
    chart_wrap(fig_split)

    section_title("Distribusi Label pada Train & Test")
    train_dist = y_train.value_counts().rename({0:"Low Risk",1:"High Risk"}).reset_index()
    train_dist.columns = ["Kelas","Jumlah"]
    train_dist["Set"] = "Train"
    test_dist  = y_test.value_counts().rename({0:"Low Risk",1:"High Risk"}).reset_index()
    test_dist.columns = ["Kelas","Jumlah"]
    test_dist["Set"] = "Test"
    dist_df = pd.concat([train_dist, test_dist])
    fig_dist = px.bar(dist_df, x="Set", y="Jumlah", color="Kelas", barmode="group",
                      color_discrete_map={"Low Risk": SUCCESS, "High Risk": DANGER})
    fig_dist.update_layout(**PLOTLY_LAYOUT)
    chart_wrap(fig_dist)

# ==================================================
# 4. PRIOR PROBABILITY (Bukti Algoritma)
# ==================================================
elif menu == "Prior Probability":
    hero("📐", "Prior Probability", "Pembuktian matematis cara kerja Gaussian Naive Bayes")

    section_title("Apa itu Prior Probability?")
    st.markdown("""
    <div class="insight-card">
    <b>Prior Probability</b> adalah probabilitas awal sebuah kelas <i>sebelum</i> melihat data fitur apapun.
    Naive Bayes menghitungnya dari proporsi kelas di data training.<br><br>
    <b>Rumus:</b> P(Kelas) = Jumlah data kelas tersebut / Total data training
    </div>
    """, unsafe_allow_html=True)

    n_high  = int((y_train == 1).sum())
    n_low   = int((y_train == 0).sum())
    n_total = len(y_train)
    p_high  = n_high / n_total
    p_low   = n_low  / n_total

    section_title("Hasil Prior Probability dari Data Training")
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("📊", "Total Data Train", f"{n_total:,}")
    with c2: kpi_card("⚠️", "P(High Risk)", f"{p_high:.4f} ({p_high*100:.1f}%)")
    with c3: kpi_card("✅", "P(Low Risk)",  f"{p_low:.4f} ({p_low*100:.1f}%)")

    fig_prior = px.bar(
        x=["P(High Risk)", "P(Low Risk)"],
        y=[p_high, p_low],
        color=["High Risk", "Low Risk"],
        color_discrete_map={"High Risk": DANGER, "Low Risk": SUCCESS},
        text=[f"{p_high:.4f}", f"{p_low:.4f}"],
    )
    fig_prior.update_traces(textposition="outside")
    fig_prior.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="Prior Probability per Kelas")
    chart_wrap(fig_prior)

    section_title("Mean & Variance per Fitur (Likelihood Parameter)")
    st.markdown("""
    <div class="insight-card">
    Gaussian NB menyimpan <b>mean (θ)</b> dan <b>variance (σ²)</b> setiap fitur untuk tiap kelas.
    Nilai inilah yang dipakai untuk menghitung likelihood P(fitur | kelas) menggunakan rumus distribusi normal.
    </div>
    """, unsafe_allow_html=True)

    # model.theta_ = mean per class, model.var_ = variance per class
    mean_df = pd.DataFrame(
        model.theta_,
        columns=[FEATURE_LABELS[f] for f in FEATURES],
        index=["Low Risk (0)", "High Risk (1)"]
    ).T
    var_df  = pd.DataFrame(
        model.var_,
        columns=[FEATURE_LABELS[f] for f in FEATURES],
        index=["Low Risk (0)", "High Risk (1)"]
    ).T

    col1, col2 = st.columns(2)
    with col1:
        section_title("Mean (θ) per Fitur")
        st.dataframe(mean_df.style.format("{:.3f}"), use_container_width=True)
    with col2:
        section_title("Variance (σ²) per Fitur")
        st.dataframe(var_df.style.format("{:.3f}"), use_container_width=True)

    section_title("🧮 Contoh Perhitungan Manual (1 Sampel Data)")
    st.markdown("""
    <div class="insight-card">
    Di bawah ini adalah simulasi perhitungan Naive Bayes secara manual menggunakan 1 baris data dari data test.
    Kamu bisa lihat step-by-step bagaimana model sampai ke keputusan akhir.
    </div>
    """, unsafe_allow_html=True)

    sample = X_test.iloc[0]
    true_label  = int(y_test.iloc[0])
    pred_label  = int(model.predict([sample])[0])

    st.markdown("**Data sampel yang dihitung:**")
    sample_df = pd.DataFrame({
        "Fitur": [FEATURE_LABELS[f] for f in FEATURES],
        "Nilai": [f"{sample[f]:.2f}" for f in FEATURES]
    })
    st.dataframe(sample_df, hide_index=True, use_container_width=True)

    # manual gaussian pdf
    def gaussian_pdf(x, mean, var):
        eps = 1e-9
        coef = 1.0 / np.sqrt(2 * np.pi * (var + eps))
        exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
        return coef * exponent

    rows = []
    for cls_idx, cls_name in enumerate(["Low Risk (0)", "High Risk (1)"]):
        for i, f in enumerate(FEATURES):
            x    = sample[f]
            mean = model.theta_[cls_idx][i]
            var  = model.var_[cls_idx][i]
            pdf  = gaussian_pdf(x, mean, var)
            rows.append({
                "Kelas": cls_name,
                "Fitur": FEATURE_LABELS[f],
                "Nilai (x)": round(x, 3),
                "Mean (θ)": round(mean, 3),
                "Var (σ²)": round(var, 3),
                "P(x|kelas)": f"{pdf:.6f}",
            })

    calc_df = pd.DataFrame(rows)
    st.dataframe(calc_df, hide_index=True, use_container_width=True)

    # posterior
    log_prob = model.predict_log_proba([sample])[0]
    prob     = model.predict_proba([sample])[0]

    section_title("Posterior Probability")
    c1, c2 = st.columns(2)
    with c1:
        kpi_card("✅", "P(Low Risk | data)", f"{prob[0]*100:.4f}%")
    with c2:
        kpi_card("⚠️", "P(High Risk | data)", f"{prob[1]*100:.4f}%")

    final_cls = "High Risk ⚠️" if pred_label == 1 else "Low Risk ✅"
    st.markdown(f"""
    <div class="insight-card" style="margin-top:14px">
    ➡️ Karena <b>P({'High Risk' if pred_label==1 else 'Low Risk'})</b> lebih besar, model memilih kelas:
    <b style="color:{'#f87171' if pred_label==1 else '#34d399'}; font-size:18px"> {final_cls}</b>
    <br>Label asli: <b>{'High Risk' if true_label==1 else 'Low Risk'}</b>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Prediksi: <b>{'High Risk' if pred_label==1 else 'Low Risk'}</b>
    &nbsp;&nbsp;→&nbsp;&nbsp;
    {'✅ Benar' if pred_label==true_label else '❌ Salah'}
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# 5. NAIVE BAYES CLASSIFIER
# ==================================================
elif menu == "Naive Bayes Classifier":
    hero("🤖", "Naive Bayes Classifier", "Evaluasi performa model secara lengkap")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("🎯", "Accuracy",  f"{acc:.4f}")
    with c2: kpi_card("🔬", "Precision", f"{precision:.4f}")
    with c3: kpi_card("📡", "Recall",    f"{recall:.4f}")
    with c4: kpi_card("⚖️", "F1-Score",  f"{f1:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        section_title("Confusion Matrix")
        cm     = confusion_matrix(y_test, y_pred)
        labels = ["Low Risk", "High Risk"]
        fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True,
                           color_continuous_scale="Blues",
                           labels=dict(x="Prediksi", y="Aktual"))
        fig_cm.update_layout(**PLOTLY_LAYOUT)
        chart_wrap(fig_cm)

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        <div class="insight-card">
        <b>TP (True Positive)</b> = {tp} → diprediksi High Risk, aslinya High Risk ✅<br>
        <b>TN (True Negative)</b> = {tn} → diprediksi Low Risk, aslinya Low Risk ✅<br>
        <b>FP (False Positive)</b> = {fp} → diprediksi High Risk, aslinya Low Risk ❌<br>
        <b>FN (False Negative)</b> = {fn} → diprediksi Low Risk, aslinya High Risk ❌
        </div>
        """, unsafe_allow_html=True)

    with col2:
        section_title("Feature Influence")
        importance = np.abs(model.theta_[1] - model.theta_[0])
        order      = np.argsort(importance)
        fig_fi = px.bar(
            x=importance[order],
            y=[FEATURE_LABELS[f] for f in np.array(FEATURES)[order]],
            orientation="h",
            color=importance[order],
            color_continuous_scale="Blues",
            labels={"x": "Selisih Mean (Influence)", "y": ""},
        )
        fig_fi.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        chart_wrap(fig_fi)

    section_title("Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Low Risk","High Risk"], output_dict=True)
    rep_df = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
    st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)

    section_title("Tabel Hasil Prediksi vs Aktual (50 data pertama)")
    result_tbl = X_test.copy().reset_index(drop=True)
    result_tbl.columns = [FEATURE_LABELS[f] for f in FEATURES]
    result_tbl["Aktual"]   = ["High Risk" if v==1 else "Low Risk" for v in y_test.values[:len(result_tbl)]]
    result_tbl["Prediksi"] = ["High Risk" if v==1 else "Low Risk" for v in y_pred[:len(result_tbl)]]
    result_tbl["Status"]   = ["✅ Benar" if a==p else "❌ Salah"
                               for a,p in zip(result_tbl["Aktual"], result_tbl["Prediksi"])]
    st.dataframe(result_tbl.head(50), use_container_width=True)

# ==================================================
# 6. VISUALISASI DISTRIBUSI
# ==================================================
elif menu == "Visualisasi Distribusi":
    hero("📈", "Visualisasi Distribusi Fitur", "Sebaran nilai tiap fitur berdasarkan kategori risiko")

    df_vis = df[FEATURES + ['addicted_label']].copy()
    df_vis['Kategori'] = df_vis['addicted_label'].map({0:"Low Risk",1:"High Risk"})

    section_title("Pilih Fitur untuk Dieksplorasi")
    selected = st.selectbox(
        "Fitur",
        FEATURES,
        format_func=lambda f: FEATURE_LABELS[f]
    )
    label = FEATURE_LABELS[selected]

    col1, col2 = st.columns(2)

    with col1:
        section_title(f"Histogram — {label}")
        fig_hist = px.histogram(
            df_vis, x=selected, color="Kategori", barmode="overlay", nbins=40,
            color_discrete_map={"Low Risk": SUCCESS, "High Risk": DANGER},
            labels={selected: label},
            opacity=0.75,
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT)
        chart_wrap(fig_hist)

    with col2:
        section_title(f"Boxplot — {label}")
        fig_box = px.box(
            df_vis, x="Kategori", y=selected, color="Kategori",
            color_discrete_map={"Low Risk": SUCCESS, "High Risk": DANGER},
            labels={selected: label},
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        chart_wrap(fig_box)

    section_title("Distribusi Semua Fitur (Violin Plot)")
    df_melt = df_vis.melt(
        id_vars=['Kategori'],
        value_vars=FEATURES,
        var_name='Fitur',
        value_name='Nilai'
    )
    df_melt['Fitur'] = df_melt['Fitur'].map(FEATURE_LABELS)
    fig_violin = px.violin(
        df_melt, x="Fitur", y="Nilai", color="Kategori", box=True,
        color_discrete_map={"Low Risk": SUCCESS, "High Risk": DANGER},
    )
    fig_violin.update_layout(**PLOTLY_LAYOUT, xaxis_tickangle=-25)
    chart_wrap(fig_violin, height=480)

    section_title("Scatter Plot antar Fitur")
    c1, c2 = st.columns(2)
    feat_opts = {FEATURE_LABELS[f]: f for f in FEATURES}
    with c1:
        fx = st.selectbox("Sumbu X", list(feat_opts.keys()), index=0)
    with c2:
        fy = st.selectbox("Sumbu Y", list(feat_opts.keys()), index=1)
    fig_sc = px.scatter(
        df_vis, x=feat_opts[fx], y=feat_opts[fy], color="Kategori",
        color_discrete_map={"Low Risk": SUCCESS, "High Risk": DANGER},
        labels={feat_opts[fx]: fx, feat_opts[fy]: fy},
        opacity=0.6,
    )
    fig_sc.update_layout(**PLOTLY_LAYOUT)
    chart_wrap(fig_sc)

# ==================================================
# 7. ANALISIS KARAKTERISTIK
# ==================================================
elif menu == "Analisis Karakteristik":
    hero("🔎", "Analisis Karakteristik", "Perbandingan karakteristik pengguna Low Risk vs High Risk")

    df_grp = df.groupby('addicted_label')[FEATURES].mean().rename(index={0:"Low Risk",1:"High Risk"})

    section_title("Rata-rata Nilai Fitur per Kategori")
    display_df = df_grp.copy()
    display_df.columns = [FEATURE_LABELS[f] for f in display_df.columns]
    st.dataframe(display_df.style.format("{:.2f}"), use_container_width=True)

    section_title("Radar Chart — Profil Rata-rata per Kelas")
    cats = [FEATURE_LABELS[f] for f in FEATURES]
    # normalize 0-1 for radar
    mn   = df[FEATURES].min()
    mx   = df[FEATURES].max()
    norm = (df_grp - mn) / (mx - mn + 1e-9)

    def hex_to_rgba(hex_color, alpha=0.15):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig_radar = go.Figure()
    colors = {"Low Risk": SUCCESS, "High Risk": DANGER}
    for cls in ["Low Risk", "High Risk"]:
        vals = norm.loc[cls].tolist()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill='toself',
            name=cls,
            line_color=colors[cls],
            fillcolor=hex_to_rgba(colors[cls], 0.15),
        ))
    fig_radar.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], color="#6b7280"),
            angularaxis=dict(color="#6b7280"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True, height=450,
    )
    chart_wrap(fig_radar, height=450)

    section_title("Selisih Mean (High Risk − Low Risk)")
    diff   = df_grp.loc["High Risk"] - df_grp.loc["Low Risk"]
    diff_df= pd.DataFrame({"Fitur": [FEATURE_LABELS[f] for f in FEATURES], "Selisih": diff.values})
    diff_df["Warna"] = ["High Risk lebih tinggi" if v>0 else "Low Risk lebih tinggi" for v in diff_df["Selisih"]]
    fig_diff = px.bar(diff_df, x="Fitur", y="Selisih", color="Warna",
                      color_discrete_map={
                          "High Risk lebih tinggi": DANGER,
                          "Low Risk lebih tinggi": SUCCESS,
                      })
    fig_diff.update_layout(**PLOTLY_LAYOUT, showlegend=True, xaxis_tickangle=-25)
    chart_wrap(fig_diff)

    section_title("Insight Karakteristik")
    for f in FEATURES:
        hr_val = df_grp.loc["High Risk", f]
        lr_val = df_grp.loc["Low Risk", f]
        diff_v = hr_val - lr_val
        arah   = "lebih tinggi ⬆️" if diff_v > 0 else "lebih rendah ⬇️"
        st.markdown(f"- **{FEATURE_LABELS[f]}**: High Risk rata-rata **{hr_val:.1f}** vs Low Risk **{lr_val:.1f}** → {arah} sebesar **{abs(diff_v):.1f}**")

# ==================================================
# 8. HASIL REKOMENDASI (PREDICT)
# ==================================================
elif menu == "Hasil Rekomendasi":
    hero("🔍", "Hasil Rekomendasi", "Input data pengguna → prediksi risiko + rekomendasi")
 
    # ── tab: Manual vs Upload ──
    tab1, tab2 = st.tabs(["👤 Prediksi Manual (1 orang)", "📂 Prediksi Massal (Upload Excel)"])
 
    # =============================================
    # TAB 1 — MANUAL
    # =============================================
    with tab1:
        with st.form("predict_form"):
            section_title("⏱️ Pola Penggunaan Harian")
            u1, u2, u3, u4 = st.columns(4)
            screen = u1.slider("Screen Time Harian (jam)", 0.0, 24.0, 5.0)
            social = u2.slider("Media Sosial (jam)",       0.0, 24.0, 3.0)
            gaming = u3.slider("Gaming (jam)",             0.0, 24.0, 2.0)
            work   = u4.slider("Kerja / Belajar (jam)",    0.0, 24.0, 6.0)
 
            u5, u6, u7, u8 = st.columns(4)
            sleep   = u5.slider("Jam Tidur",                 0.0, 24.0, 7.0)
            weekend = u6.slider("Screen Time Weekend (jam)", 0.0, 24.0, 6.0)
            notif   = u7.number_input("Notifikasi / Hari",   0, 1000, 50)
            opens   = u8.number_input("Buka Aplikasi / Hari",0, 1000, 60)
 
            submitted = st.form_submit_button("🚀 Predict Sekarang", type="primary")
 
        if submitted:
            inp  = np.array([[screen, social, gaming, work, sleep, notif, opens, weekend]])
            pred = int(model.predict(inp)[0])
            prob = model.predict_proba(inp)[0]
 
            st.session_state.last_pred = {
                "inputs": {
                    "Screen Time Harian (jam)":  screen,
                    "Media Sosial (jam)":         social,
                    "Gaming (jam)":               gaming,
                    "Kerja / Belajar (jam)":      work,
                    "Jam Tidur":                  sleep,
                    "Screen Time Weekend (jam)":  weekend,
                    "Notifikasi / Hari":          notif,
                    "Buka Aplikasi / Hari":       opens,
                },
                "pred":      pred,
                "prob_low":  float(prob[0]) * 100,
                "prob_high": float(prob[1]) * 100,
                "timestamp": datetime.now(),
            }
 
        if "last_pred" in st.session_state:
            res  = st.session_state.last_pred
            pred = res["pred"]
 
            banner_cls = "result-high" if pred == 1 else "result-low"
            banner_txt = ("⚠️ High Risk Addiction Detected" if pred == 1
                          else "✅ Low Risk — Penggunaan Tergolong Aman")
            st.markdown(f'<div class="result-banner {banner_cls}">{banner_txt}</div>',
                        unsafe_allow_html=True)
 
            g1, g2 = st.columns([1, 1.4])
            with g1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res["prob_high"],
                    number={"suffix": "%", "font": {"color": "#f8fafc"}},
                    gauge={
                        "axis": {"range": [0,100], "tickcolor": "#6b7280"},
                        "bar":  {"color": DANGER if pred==1 else SUCCESS},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0,50],  "color": "rgba(52,211,153,0.18)"},
                            {"range": [50,100],"color": "rgba(248,113,113,0.18)"},
                        ],
                    },
                    title={"text": "High Risk Probability", "font": {"color": "#9ca3af","size":13}},
                ))
                fig_gauge.update_layout(**PLOTLY_LAYOUT, height=280)
                chart_wrap(fig_gauge)
 
            with g2:
                section_title("Probability Breakdown")
                st.markdown(f"**Low Risk:** {res['prob_low']:.1f}%")
                st.progress(res["prob_low"] / 100)
                st.markdown(f"**High Risk:** {res['prob_high']:.1f}%")
                st.progress(res["prob_high"] / 100)
 
                section_title("Data yang Dimasukkan")
                in_df = pd.DataFrame({"Field": res["inputs"].keys(),
                                      "Nilai": res["inputs"].values()})
                st.dataframe(in_df, hide_index=True, use_container_width=True)
 
            section_title("💡 Rekomendasi")
            recs_high = [
                ("📵", "Batasi screen time harian maksimal 4 jam di luar jam kerja/belajar."),
                ("🌙", "Usahakan tidur minimal 7–8 jam per malam; hindari HP 1 jam sebelum tidur."),
                ("📲", "Kurangi notifikasi aktif — matikan notif yang tidak penting."),
                ("⚽", "Ganti waktu main HP dengan aktivitas fisik atau hobi offline."),
                ("🧘", "Pertimbangkan digital detox minimal 1 hari per minggu."),
            ]
            recs_low = [
                ("✅", "Pola penggunaan smartphonemu sudah tergolong sehat — pertahankan!"),
                ("⏰", "Tetap jaga screen time harian agar tidak merangkak naik."),
                ("😴", "Konsistensi tidur yang baik adalah kunci — jaga jam tidurmu."),
                ("🔕", "Sesekali coba mode DND (Do Not Disturb) saat fokus belajar/kerja."),
            ]
            for icon, text in (recs_high if pred == 1 else recs_low):
                st.markdown(f'''<div class="insight-card" style="margin-bottom:8px">
                {icon} {text}</div>''', unsafe_allow_html=True)
 
    # =============================================
    # TAB 2 — BATCH / UPLOAD EXCEL
    # =============================================
    with tab2:
        section_title("📂 Upload File Excel")
 
        # Template download
        st.markdown("""
        <div class="insight-card">
        Upload file <b>Excel (.xlsx)</b> dengan kolom sesuai format di bawah.
        Setiap baris = 1 orang. Klik tombol <b>Download Template</b> jika belum punya formatnya.
        </div>
        """, unsafe_allow_html=True)
 
        # Buat template Excel
        template_df = pd.DataFrame(columns=FEATURES)
        template_df.loc[0] = [5.0, 3.0, 2.0, 6.0, 7.0, 50, 60, 6.0]   # contoh baris
        template_df.loc[1] = [10.0, 6.0, 4.0, 3.0, 5.0, 120, 150, 12.0]
 
        buf_tmpl = io.BytesIO()
        template_df.to_excel(buf_tmpl, index=False)
        buf_tmpl.seek(0)
 
        st.download_button(
            "📥 Download Template Excel",
            data=buf_tmpl,
            file_name="template_prediksi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
 
        uploaded = st.file_uploader("Upload file Excel kamu di sini", type=["xlsx","xls"])
 
        if uploaded is not None:
            try:
                up_df = pd.read_excel(uploaded)
 
                # Validasi kolom
                missing_cols = [f for f in FEATURES if f not in up_df.columns]
                if missing_cols:
                    st.error(f"❌ Kolom berikut tidak ditemukan di file: {missing_cols}")
                    st.info("Pastikan nama kolom sama persis dengan template. Gunakan tombol Download Template di atas.")
                else:
                    batch_df = up_df[FEATURES].copy().fillna(0)
 
                    # Prediksi semua baris
                    preds = model.predict(batch_df)
                    probs = model.predict_proba(batch_df)
 
                    batch_df["Prediksi"]        = ["High Risk ⚠️" if p==1 else "Low Risk ✅" for p in preds]
                    batch_df["Prob. Low Risk"]  = [f"{p[0]*100:.1f}%" for p in probs]
                    batch_df["Prob. High Risk"] = [f"{p[1]*100:.1f}%" for p in probs]
                    batch_df["Rekomendasi"]     = [
                        "Segera kurangi screen time & perbaiki pola tidur."
                        if p==1 else "Pertahankan pola penggunaan yang sehat."
                        for p in preds
                    ]
                    batch_df.columns = (
                        [FEATURE_LABELS[f] for f in FEATURES] +
                        ["Prediksi", "Prob. Low Risk", "Prob. High Risk", "Rekomendasi"]
                    )
                    batch_df.insert(0, "No", range(1, len(batch_df)+1))
 
                    # Ringkasan
                    n_high = int(sum(preds == 1))
                    n_low  = int(sum(preds == 0))
                    n_tot  = len(preds)
                    section_title("📊 Ringkasan Hasil")
                    c1, c2, c3 = st.columns(3)
                    with c1: kpi_card("👥", "Total Data", f"{n_tot}")
                    with c2: kpi_card("⚠️", "High Risk", f"{n_high} orang ({n_high/n_tot*100:.1f}%)")
                    with c3: kpi_card("✅", "Low Risk",  f"{n_low} orang ({n_low/n_tot*100:.1f}%)")
 
                    # Pie ringkasan
                    fig_b = px.pie(
                        names=["High Risk","Low Risk"], values=[n_high, n_low],
                        color_discrete_sequence=[DANGER, SUCCESS], hole=0.5,
                    )
                    fig_b.update_layout(**PLOTLY_LAYOUT, showlegend=True)
                    chart_wrap(fig_b)
 
                    section_title("📋 Tabel Hasil Prediksi")
                    st.dataframe(batch_df, use_container_width=True, hide_index=True)
 
                    # Download hasil Excel
                    buf_out = io.BytesIO()
                    batch_df.to_excel(buf_out, index=False)
                    buf_out.seek(0)
 
                    st.download_button(
                        "📥 Download Hasil Prediksi (.xlsx)",
                        data=buf_out,
                        file_name=f"hasil_prediksi_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )
 
                    st.session_state.last_batch = batch_df
 
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")
 

# ==================================================
# 9. DOWNLOAD HASIL
# ==================================================
elif menu == "Download Hasil":
    hero("📄", "Download Hasil", "Unduh laporan hasil prediksi terakhir dalam format PDF")

    if "last_pred" not in st.session_state:
        st.markdown("""
        <div class="insight-card">
        ⚠️ Belum ada hasil prediksi. Silakan ke halaman <b>Hasil Rekomendasi</b> terlebih dahulu,
        isi data dan klik <b>Predict Sekarang</b>, lalu kembali ke sini untuk download.
        </div>
        """, unsafe_allow_html=True)
    else:
        res  = st.session_state.last_pred
        pred = res["pred"]

        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("🎯", "Hasil Prediksi", "High Risk ⚠️" if pred==1 else "Low Risk ✅")
        with c2: kpi_card("📊", "Prob. High Risk", f"{res['prob_high']:.1f}%")
        with c3: kpi_card("📅", "Waktu Prediksi",  res["timestamp"].strftime("%d %b %Y %H:%M"))

        section_title("Preview Data yang Akan Di-download")
        in_df = pd.DataFrame({"Field": res["inputs"].keys(), "Nilai": res["inputs"].values()})
        st.dataframe(in_df, hide_index=True, use_container_width=True)

        def create_pdf(res):
            buffer = io.BytesIO()
            c      = canvas.Canvas(buffer, pagesize=letter)
            W, H   = letter

            # header bar
            c.setFillColorRGB(0.07, 0.09, 0.16)
            c.rect(0, H-90, W, 90, fill=1, stroke=0)
            c.setFillColorRGB(1,1,1)
            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, H-45, "Smartphone Addiction - Prediction Report")
            c.setFont("Helvetica", 10)
            c.drawString(50, H-65, f"Generated: {res['timestamp'].strftime('%d %B %Y, %H:%M')}")
            c.drawString(50, H-78, f"Model: Gaussian Naive Bayes  |  Accuracy: {acc:.2%}")

            # result section
            y = H - 130
            c.setFillColorRGB(0,0,0)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y, "Hasil Prediksi")
            y -= 22
            c.setFont("Helvetica", 12)
            risk_text = "High Risk Addiction" if res["pred"]==1 else "Low Risk (Aman)"
            c.drawString(50, y, f"Hasil       : {risk_text}");           y -= 18
            c.drawString(50, y, f"Prob. Low Risk  : {res['prob_low']:.2f}%");  y -= 18
            c.drawString(50, y, f"Prob. High Risk : {res['prob_high']:.2f}%"); y -= 30

            # input data
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y, "Data yang Dimasukkan"); y -= 22
            c.setFont("Helvetica", 11)
            for field, val in res["inputs"].items():
                c.drawString(50, y, f"{field}: {val}")
                y -= 18
                if y < 80:
                    c.showPage(); y = H - 60

            # recommendations
            y -= 10
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y, "Rekomendasi"); y -= 22
            c.setFont("Helvetica", 11)
            recs = (
                [
                    "Batasi screen time harian maksimal 4 jam di luar jam kerja/belajar.",
                    "Usahakan tidur minimal 7-8 jam per malam; hindari HP 1 jam sebelum tidur.",
                    "Kurangi notifikasi aktif — matikan notif yang tidak penting.",
                    "Ganti waktu main HP dengan aktivitas fisik atau hobi offline.",
                    "Pertimbangkan digital detox minimal 1 hari per minggu.",
                ] if res["pred"]==1 else [
                    "Pola penggunaan smartphonemu sudah tergolong sehat — pertahankan!",
                    "Tetap jaga screen time harian agar tidak merangkak naik.",
                    "Konsistensi tidur yang baik adalah kunci — jaga jam tidurmu.",
                    "Sesekali coba mode DND saat fokus belajar/kerja.",
                ]
            )
            for i, r in enumerate(recs, 1):
                c.drawString(50, y, f"{i}. {r}"); y -= 18
                if y < 80:
                    c.showPage(); y = H - 60

            c.save()
            buffer.seek(0)
            return buffer

        pdf = create_pdf(res)
        st.download_button(
            "📥 Download PDF Report",
            data=pdf,
            file_name=f"prediction_{res['timestamp'].strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            type="primary",
        )