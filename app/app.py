"""
LSTM + Attention 零售需求预测 — Streamlit Web App
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──
st.set_page_config(
    page_title="零售需求预测 | LSTM + Attention",
    page_icon="📊",
    layout="wide"
)

# ── Model architecture (must match training) ──
LOOKBACK = 12
HORIZON = 4
FEATURE_COLS = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

class LSTMAttentionForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, horizon=4, num_quantiles=3):
        super().__init__()
        self.horizon = horizon
        self.num_quantiles = num_quantiles
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4,
                                                dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * num_quantiles)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)
        last_hidden = attn_out[:, -1, :]
        attn_score = torch.mean(attn_out, dim=-1, keepdim=True)
        attn_weight = torch.softmax(attn_score, dim=1)
        weighted_hidden = torch.sum(attn_out * attn_weight, dim=1)
        combined = (last_hidden + weighted_hidden) / 2
        out = self.fc(combined)
        return out.view(-1, self.horizon, self.num_quantiles), attn_weights


# ── Paths: works on both local dev and Streamlit Cloud ──
import os, urllib.request
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)  # go up from app/ to repo root
DATA_PATH = os.path.join(REPO_DIR, 'data', 'Walmart.csv')
MODEL_DIR = os.path.join(REPO_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_lstm_attention.pt')
MODEL_URL = "https://github.com/lingtao-meng/lstm-attention-demand-forecasting/releases/download/v1.0/best_lstm_attention.pt"

# ── Load model ──
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = LSTMAttentionForecaster(input_dim=len(FEATURE_COLS))

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    else:
        os.makedirs(MODEL_DIR, exist_ok=True)
        st.info("⏳ 首次运行，正在下载模型文件（约3.3MB）...")
        progress = st.progress(0)
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                progress.progress(min(block_num * block_size / total_size, 1.0))
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=report_progress)
        progress.progress(1.0)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        st.success("✅ 模型下载完成！")

    model.to(device)
    model.eval()
    return model

# ── Load data ──
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    df['Store'] = df['Store'].astype(int)
    return df

# ── Main UI ──
st.title("📊 零售需求预测 — LSTM + Attention")
st.markdown("基于 Walmart 45家门店数据，使用深度学习模型预测未来4周销售额")
st.markdown("---")

# Load
with st.spinner("加载模型和数据..."):
    model = load_model()
    df = load_data()

# Sidebar
st.sidebar.header("⚙️ 参数设置")
store = st.sidebar.selectbox("选择门店", sorted(df['Store'].unique()), index=0)
show_attn = st.sidebar.checkbox("显示注意力权重", value=False)

# Get store data
store_df = df[df['Store'] == store].sort_values('Date').tail(60)

# Scale data
scaler = StandardScaler()
values = store_df[FEATURE_COLS].values
values_scaled = scaler.fit_transform(values)

# Prepare last window for prediction
last_window = values_scaled[-LOOKBACK:]

# Predict
device = next(model.parameters()).device
x = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    pred, attn_weights = model(x)
pred = pred.cpu().numpy()[0]  # (horizon, 3)

# Inverse transform: we need the sales mean/std
sales_mean = scaler.mean_[0]
sales_std = scaler.scale_[0]

pred_original = pred * sales_std + sales_mean  # Approximate

# ── Display results ──
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 门店 {store} — 销售趋势与预测")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Historical (last 24 weeks)
    hist_sales = store_df['Weekly_Sales'].values[-24:] / 1e6
    ax.plot(range(-23, 1), hist_sales, 'b-', linewidth=2, label='历史销售额', marker='o', markersize=4)

    # Forecast
    forecast_p50 = pred_original[:, 1] / 1e6
    forecast_p10 = pred_original[:, 0] / 1e6
    forecast_p90 = pred_original[:, 2] / 1e6
    ax.plot(range(1, 5), forecast_p50, 'r-o', linewidth=2, markersize=8, label='P50 预测')
    ax.fill_between(range(1, 5), forecast_p10, forecast_p90,
                     alpha=0.3, color='red', label='P10-P90 区间')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='当前时间')

    ax.set_xlabel('周（0 = 当前，正值 = 预测）')
    ax.set_ylabel('周销售额（百万美元）')
    ax.set_title(f'门店 {store} — 未来4周需求预测', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("📋 预测详情")
    last_date = store_df['Date'].max()
    for i in range(HORIZON):
        future_date = last_date + pd.Timedelta(weeks=i + 1)
        st.metric(
            label=f"第 {i+1} 周 ({future_date.strftime('%m/%d')})",
            value=f"${pred_original[i, 1]:,.0f}",
            delta=f"P10: ${pred_original[i, 0]:,.0f} / P90: ${pred_original[i, 2]:,.0f}"
        )

    st.markdown("---")
    st.caption("💡 P10 = 保守估计（10%概率低于此值）")
    st.caption("💡 P50 = 中位预测（最可能的值）")
    st.caption("💡 P90 = 乐观估计（10%概率高于此值）")

# ── Model info ──
st.markdown("---")
col_a, col_b, col_c = st.columns(3)
col_a.metric("模型参数", "832,652")
col_b.metric("训练门店数", "45")
col_c.metric("预测窗口", "12周 → 4周")

# ── Attention weights ──
if show_attn:
    st.markdown("---")
    st.subheader("🔍 注意力权重分析")

    attn_matrix = attn_weights[0].cpu().numpy()
    avg_attn = attn_matrix.mean(axis=0)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    weeks_labels = [f'{LOOKBACK-i}周前' for i in range(LOOKBACK)]
    colors = ['#2196F3' if v > avg_attn.mean() else '#BBDEFB' for v in avg_attn]
    ax2.barh(range(LOOKBACK), avg_attn, color=colors)
    ax2.set_yticks(range(LOOKBACK))
    ax2.set_yticklabels(weeks_labels)
    ax2.set_xlabel('注意力权重')
    ax2.set_title(f'门店 {store} — 不同历史时间步的重要性', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    st.pyplot(fig2)

    top_weeks = np.argsort(avg_attn)[::-1][:3]
    st.caption(f"📌 最受关注的3个时间步：{', '.join([f'{LOOKBACK-w}周前' for w in top_weeks])}")

# ── Footer ──
st.markdown("---")
st.caption("🔗 github.com/lingtao-meng/lstm-attention-demand-forecasting")
st.caption("🛠 PyTorch · Streamlit · Walmart Retail Data")
