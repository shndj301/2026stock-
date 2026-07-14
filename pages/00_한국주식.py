import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="🇰🇷 한국 AI·반도체 주식 분석", layout="wide", initial_sidebar_state="expanded")

# ---------------------- 스타일 ----------------------
st.markdown("""
<style>
    .main-title {font-size: 2.3rem; font-weight: 800; margin-bottom: 0;}
    .sub-title {color: #888; margin-top: 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🇰🇷 한국 AI·반도체 대표주 분석</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Yahoo Finance 데이터 기반 실시간 인터랙티브 대시보드</p>', unsafe_allow_html=True)

# ---------------------- 종목 매핑 (KRX 티커: 한글명) ----------------------
KR_STOCKS = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "042700.KQ": "한미반도체",
    "000990.KS": "DB하이텍",
    "011070.KS": "LG이노텍",
    "009150.KS": "삼성전기",
    "035420.KS": "네이버",
    "035720.KS": "카카오",
    "066570.KS": "LG전자",
    "058470.KQ": "리노공업",
    "240810.KQ": "원익IPS",
    "403870.KQ": "HPSP",
}

# ---------------------- 사이드바 ----------------------
st.sidebar.header("⚙️ 설정")

st.sidebar.caption("💡 AI·반도체 관련 대표 종목 목록입니다")
selected_names = st.sidebar.multiselect(
    "종목 선택 (복수 선택 가능)",
    options=list(KR_STOCKS.values()),
    default=["삼성전자", "SK하이닉스", "한미반도체", "네이버"]
)
name_to_ticker = {v: k for k, v in KR_STOCKS.items()}
tickers = [name_to_ticker[n] for n in selected_names]

period_options = {
    "1개월": "1mo", "3개월": "3mo", "6개월": "6mo",
    "1년": "1y", "2년": "2y", "5년": "5y", "전체": "max"
}
period_label = st.sidebar.selectbox("기간", list(period_options.keys()), index=3)
period = period_options[period_label]

interval_options = {"1일": "1d", "1주": "1wk", "1개월": "1mo"}
interval_label = st.sidebar.selectbox("간격", list(interval_options.keys()))
interval = interval_options[interval_label]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 보조 지표")
show_ma = st.sidebar.checkbox("이동평균선 (MA)", value=True)
ma_periods = st.sidebar.multiselect("MA 기간", [5, 20, 60, 120, 200], default=[20, 60])
show_bb = st.sidebar.checkbox("볼린저 밴드", value=False)
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_macd = st.sidebar.checkbox("MACD", value=False)
show_volume = st.sidebar.checkbox("거래량", value=True)

st.sidebar.markdown("---")
if not selected_names:
    st.warning("사이드바에서 최소 하나의 종목을 선택해주세요.")
    st.stop()

main_name = st.sidebar.selectbox("상세 분석 종목", selected_names)
main_ticker = name_to_ticker[main_name]

# ---------------------- 데이터 로드 ----------------------
@st.cache_data(ttl=3600)
def load_data(ticker, period, interval):
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    return df

@st.cache_data(ttl=3600)
def load_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

with st.spinner("데이터를 불러오는 중..."):
    df = load_data(main_ticker, period, interval)
    info = load_info(main_ticker)

if df.empty:
    st.error(f"'{main_name}' 데이터를 가져올 수 없습니다. 잠시 후 다시 시도해주세요.")
    st.stop()

# ---------------------- 상단 요약 지표 ----------------------
last_close = df["Close"].iloc[-1]
prev_close = df["Close"].iloc[-2] if len(df) > 1 else last_close
change = last_close - prev_close
pct_change = (change / prev_close) * 100 if prev_close else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("현재가", f"₩{last_close:,.0f}", f"{change:+,.0f} ({pct_change:+.2f}%)")
col2.metric("기간 최고", f"₩{df['High'].max():,.0f}")
col3.metric("기간 최저", f"₩{df['Low'].min():,.0f}")
col4.metric("평균 거래량", f"{df['Volume'].mean():,.0f}")
market_cap = info.get("marketCap")
col5.metric("시가총액", f"₩{market_cap/1e12:,.1f}조" if market_cap else "N/A")

st.caption(f"**{main_name}** ({main_ticker}) · {info.get('sector', '')} / {info.get('industry', '')}")

# ---------------------- 보조지표 계산 ----------------------
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

for p in ma_periods:
    df[f"MA{p}"] = df["Close"].rolling(p).mean()

df["BB_MID"] = df["Close"].rolling(20).mean()
df["BB_STD"] = df["Close"].rolling(20).std()
df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]

df["RSI"] = calc_rsi(df["Close"])
df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = calc_macd(df["Close"])

# ---------------------- 서브플롯 구성 ----------------------
row_count = 1 + show_volume + show_rsi + show_macd
row_heights = [0.55] + [0.15] * (row_count - 1) if row_count > 1 else [1.0]

specs_titles = ["가격"]
if show_volume: specs_titles.append("거래량")
if show_rsi: specs_titles.append("RSI")
if show_macd: specs_titles.append("MACD")

fig = make_subplots(
    rows=row_count, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=row_heights,
    subplot_titles=specs_titles
)

current_row = 1

fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name=main_name,
    increasing_line_color="#ef5350", decreasing_line_color="#2962ff"
), row=current_row, col=1)

if show_ma:
    colors = ["#ffb703", "#8ecae6", "#c77dff", "#ff70a6", "#90be6d"]
    for i, p in enumerate(ma_periods):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"MA{p}"], name=f"MA{p}",
            line=dict(width=1.3, color=colors[i % len(colors)])
        ), row=current_row, col=1)

if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], name="BB 상단",
        line=dict(width=1, color="rgba(173,216,230,0.6)")), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], name="BB 하단",
        line=dict(width=1, color="rgba(173,216,230,0.6)"), fill="tonexty",
        fillcolor="rgba(173,216,230,0.1)"), row=current_row, col=1)

if show_volume:
    current_row += 1
    vol_colors = np.where(df["Close"] >= df["Open"], "#ef5350", "#2962ff")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량",
        marker_color=vol_colors), row=current_row, col=1)

if show_rsi:
    current_row += 1
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#ab47bc", width=1.5)), row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)

if show_macd:
    current_row += 1
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
        line=dict(color="#42a5f5", width=1.3)), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal",
        line=dict(color="#ffa726", width=1.3)), row=current_row, col=1)
    hist_colors = np.where(df["MACD_HIST"] >= 0, "#ef5350", "#2962ff")
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Histogram",
        marker_color=hist_colors), row=current_row, col=1)

fig.update_layout(
    height=250 * row_count + 300,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=40, b=10)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------- 종목 비교 ----------------------
st.markdown("---")
st.subheader("📊 선택 종목 수익률 비교")

if len(tickers) > 1:
    compare_data = {}
    for t, n in zip(tickers, selected_names):
        d = load_data(t, period, interval)
        if not d.empty:
            compare_data[n] = (d["Close"] / d["Close"].iloc[0] - 1) * 100

    compare_df = pd.DataFrame(compare_data)
    fig2 = go.Figure()
    for col in compare_df.columns:
        fig2.add_trace(go.Scatter(x=compare_df.index, y=compare_df[col], name=col, mode="lines"))

    fig2.update_layout(
        template="plotly_dark", height=450,
        yaxis_title="누적 수익률 (%)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("비교하려면 사이드바에서 종목을 2개 이상 선택하세요.")

# ---------------------- 데이터 테이블 ----------------------
st.markdown("---")
with st.expander("📋 원본 데이터 보기"):
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)
    csv = df.to_csv().encode("utf-8-sig")
    st.download_button("CSV 다운로드", csv, f"{main_name}_data.csv", "text/csv")

st.caption("데이터 출처: Yahoo Finance (yfinance) · 투자 판단의 참고용이며 투자 조언이 아닙니다.")
