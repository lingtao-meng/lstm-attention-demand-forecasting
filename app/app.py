"""
LSTM + Attention 零售需求预测 — Streamlit Web App
"""
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
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

    # Historical (last 24 weeks)
    hist_sales = (store_df['Weekly_Sales'].values[-24:] / 1e6)
    hist_weeks = list(range(-23, 1))
    hist_dates = store_df['Date'].dt.strftime('%m/%d').values[-24:]

    # Forecast
    forecast_p50 = pred_original[:, 1] / 1e6
    forecast_p10 = pred_original[:, 0] / 1e6
    forecast_p90 = pred_original[:, 2] / 1e6
    forecast_weeks = list(range(1, 5))

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_weeks, y=hist_sales,
        mode='lines+markers', name='Historical Sales',
        line=dict(color='#2196F3', width=2),
        marker=dict(size=5),
        hovertemplate='Week %{x}<br>Sales: $%{y:.2f}M<extra></extra>'
    ))

    # P10-P90 band
    fig.add_trace(go.Scatter(
        x=forecast_weeks + forecast_weeks[::-1],
        y=list(forecast_p90) + list(forecast_p10)[::-1],
        fill='toself', fillcolor='rgba(255,0,0,0.15)',
        line=dict(color='rgba(255,0,0,0)'),
        name='P10-P90 Interval',
        hoverinfo='skip'
    ))

    # P50 forecast line
    fig.add_trace(go.Scatter(
        x=forecast_weeks, y=forecast_p50,
        mode='lines+markers', name='P50 Forecast',
        line=dict(color='#F44336', width=2.5),
        marker=dict(size=9, symbol='diamond'),
        hovertemplate='Week +%{x}<br>P50: $%{y:.2f}M<br>P10: $%{customdata[0]:.2f}M<br>P90: $%{customdata[1]:.2f}M<extra></extra>',
        customdata=list(zip(forecast_p10, forecast_p90))
    ))

    # Now line
    fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5,
                  annotation_text='Now', annotation_position='top left')

    fig.update_layout(
        xaxis_title='Week (0 = current, positive = forecast)',
        yaxis_title='Weekly Sales (Million $)',
        title=f'Store {store} — 4-Week Demand Forecast',
        hovermode='x unified',
        template='plotly_white',
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

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

    weeks_labels = [f'{LOOKBACK-i}w ago' for i in range(LOOKBACK)]
    colors = ['#2196F3' if v > avg_attn.mean() else '#90CAF9' for v in avg_attn]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=weeks_labels, x=avg_attn,
        orientation='h', marker_color=colors,
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'
    ))
    fig2.update_layout(
        xaxis_title='Attention Weight',
        title=f'Store {store} — Attention Weights by Time Step',
        template='plotly_white',
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

    top_weeks = np.argsort(avg_attn)[::-1][:3]
    st.caption(f"📌 Top 3 most attended weeks: {', '.join([f'{LOOKBACK-w}w ago' for w in top_weeks])}")

# ── Footer ──
st.markdown("---")
st.caption("🔗 github.com/lingtao-meng/lstm-attention-demand-forecasting")
st.caption("🛠 PyTorch · Streamlit · Walmart Retail Data")
