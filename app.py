import streamlit as st
import openai
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
import numpy as np
import feedparser
import time

CONFIG_FILE = Path("config.json")

OPENAI_MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
    "gpt-3.5-turbo", "o1", "o1-mini", "o3-mini",
]

MODEL_DESC = {
    "gpt-4o": "최신 멀티모달 모델, 빠르고 저렴 (권장)",
    "gpt-4o-mini": "경량화 버전, 간단한 작업에 적합",
    "gpt-4-turbo": "GPT-4 터보, 128K 컨텍스트",
    "gpt-4": "GPT-4 기본 모델",
    "gpt-3.5-turbo": "빠르고 경제적, 간단한 작업",
    "o1": "추론 특화 모델, 복잡한 문제 해결",
    "o1-mini": "경량 추론 모델",
    "o3-mini": "최신 추론 모델 (미니)",
}

NEWS_FEEDS = {
    "📍 부산 낚시": "https://news.google.com/rss/search?q=부산+낚시&hl=ko&gl=KR&ceid=KR:ko",
    "📍 경남 낚시": "https://news.google.com/rss/search?q=경남+낚시&hl=ko&gl=KR&ceid=KR:ko",
    "📍 경북 낚시": "https://news.google.com/rss/search?q=경북+낚시&hl=ko&gl=KR&ceid=KR:ko",
    "🎣 낚시 전체": "https://news.google.com/rss/search?q=낚시&hl=ko&gl=KR&ceid=KR:ko",
    "🌊 바다낚시": "https://news.google.com/rss/search?q=바다낚시&hl=ko&gl=KR&ceid=KR:ko",
    "🏞️ 민물낚시": "https://news.google.com/rss/search?q=민물낚시&hl=ko&gl=KR&ceid=KR:ko",
    "🐟 낚시터/포인트": "https://news.google.com/rss/search?q=낚시터+포인트&hl=ko&gl=KR&ceid=KR:ko",
    "🦑 갈치/오징어낚시": "https://news.google.com/rss/search?q=갈치낚시+오징어낚시&hl=ko&gl=KR&ceid=KR:ko",
}

REGION_PRIORITY = ["부산", "경남", "경북", "울산", "포항", "거제", "통영", "남해", "여수"]


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"api_key": "", "model": "gpt-4o", "temperature": 0.7, "max_tokens": 1024}


def save_config(config: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def test_api_key(api_key: str, model: str) -> tuple[bool, str]:
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "안녕하세요. 연결 테스트입니다."}],
            max_tokens=50,
        )
        return True, response.choices[0].message.content
    except openai.AuthenticationError:
        return False, "API 키가 유효하지 않습니다."
    except openai.NotFoundError:
        return False, f"모델 '{model}'을 찾을 수 없습니다."
    except Exception as e:
        return False, f"오류: {str(e)}"


def make_sample_data() -> pd.DataFrame:
    regions = ["서울", "경기", "부산", "대구", "인천", "광주", "대전", "울산"]
    rng = np.random.default_rng(42)
    rows = []
    for r in regions:
        base = rng.uniform(800, 2500)
        for m in range(24):
            d = date(2023, 1, 1) + timedelta(days=m * 30)
            season = 1.0 + 0.3 * np.sin((d.month - 7) * np.pi / 6)
            usage = base * season + rng.normal(0, 50)
            peak = usage * rng.uniform(0.08, 0.12)
            temp = 15 + 12 * np.sin((d.month - 1) * np.pi / 6) + rng.normal(0, 2)
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "region": r,
                "usage_mwh": round(max(usage, 100), 1),
                "peak_load": round(max(peak, 10), 1),
                "temperature": round(temp, 1),
            })
    return pd.DataFrame(rows)


def render_dashboard():
    st.title("⚡ KDN 전력사용량 분석 대시보드")
    st.caption("CSV 업로드 또는 샘플 데이터로 전력 현황을 분석합니다.")

    uploaded = st.file_uploader(
        "CSV 파일 업로드 (date, region, usage_mwh, peak_load, temperature)",
        type=["csv"],
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"데이터 로드 완료: {len(df):,}행")
    else:
        df = make_sample_data()
        st.info("샘플 데이터를 사용합니다. CSV를 업로드하면 실제 데이터를 분석합니다.")

    df["date"] = pd.to_datetime(df["date"])

    # ── 필터 패널 (탭 내부 왼쪽 컬럼) ──
    filter_col, main_col = st.columns([1, 3])

    with filter_col:
        st.markdown("### 🔍 필터")
        min_d, max_d = df["date"].min().date(), df["date"].max().date()
        date_range = st.date_input("기간 선택", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_d, end_d = date_range
        else:
            start_d, end_d = min_d, max_d

        all_regions = sorted(df["region"].unique().tolist())
        selected_regions = st.multiselect("지역 선택", all_regions, default=all_regions)

    fdf = df[
        (df["date"].dt.date >= start_d) &
        (df["date"].dt.date <= end_d) &
        (df["region"].isin(selected_regions if selected_regions else all_regions))
    ]

    with main_col:
        if fdf.empty:
            st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            return

        # KPI 카드
        total_usage = fdf["usage_mwh"].sum()
        max_peak = fdf["peak_load"].max()
        monthly = fdf.groupby(fdf["date"].dt.to_period("M"))["usage_mwh"].sum().sort_index()
        mom_delta = None
        if len(monthly) >= 2:
            mom_delta = (monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("총 사용량 (MWh)", f"{total_usage:,.0f}")
        with k2:
            delta_str = f"{mom_delta:+.1f}%" if mom_delta is not None else None
            st.metric("전월 대비", delta_str or "-", delta=delta_str)
        with k3:
            st.metric("최대 부하 (MW)", f"{max_peak:,.1f}")

        st.divider()

        # 월별 추이 + 지역별 막대
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📈 월별 사용량 추이")
            mr = fdf.groupby([fdf["date"].dt.to_period("M"), "region"])["usage_mwh"].sum().reset_index()
            mr["date"] = mr["date"].dt.to_timestamp()
            fig_line = px.line(mr, x="date", y="usage_mwh", color="region",
                               labels={"date": "월", "usage_mwh": "사용량(MWh)", "region": "지역"},
                               template="plotly_white")
            fig_line.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_line, use_container_width=True)

        with c2:
            st.subheader("📊 지역별 총 사용량")
            rs = fdf.groupby("region")["usage_mwh"].sum().reset_index().sort_values("usage_mwh")
            fig_bar = px.bar(rs, x="usage_mwh", y="region", orientation="h",
                             color="usage_mwh", color_continuous_scale="Blues",
                             labels={"usage_mwh": "사용량(MWh)", "region": "지역"},
                             template="plotly_white")
            fig_bar.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

        # 기온 vs 사용량
        st.subheader("🌡️ 기온 vs 사용량 상관관계")
        fig_s = px.scatter(fdf, x="temperature", y="usage_mwh", color="region",
                           trendline="ols", opacity=0.7,
                           labels={"temperature": "기온(°C)", "usage_mwh": "사용량(MWh)", "region": "지역"},
                           template="plotly_white")
        fig_s.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_s, use_container_width=True)

        st.divider()

        # 데이터 테이블
        st.subheader("📋 원본 데이터")
        show = fdf.copy()
        show["date"] = show["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(show.sort_values(["date", "region"]).reset_index(drop=True),
                     use_container_width=True, height=280)
        csv_bytes = show.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ CSV 다운로드", data=csv_bytes,
                           file_name="kdn_filtered.csv", mime="text/csv")


@st.cache_data(ttl=300)
def fetch_news(feed_url: str) -> list[dict]:
    try:
        feed = feedparser.parse(feed_url)
        items = []
        for entry in feed.entries[:15]:
            items.append({
                "title": entry.get("title", "제목 없음"),
                "link": entry.get("link", "#"),
                "summary": entry.get("summary", entry.get("description", ""))[:200],
                "published": entry.get("published", ""),
            })
        return items
    except Exception:
        return []


def region_score(title: str, summary: str) -> int:
    text = title + summary
    score = 0
    priority = [("부산", 30), ("경남", 25), ("경북", 20),
                ("울산", 15), ("포항", 15), ("거제", 15),
                ("통영", 15), ("남해", 10), ("여수", 10)]
    for keyword, weight in priority:
        if keyword in text:
            score += weight
    return score


def render_news():
    st.title("🎣 낚시 뉴스")
    st.caption("부산·경남·경북 지역 뉴스를 우선 표시합니다.")

    col_select, col_region, col_refresh = st.columns([3, 2, 1])
    with col_select:
        feed_name = st.selectbox("뉴스 채널", list(NEWS_FEEDS.keys()), index=0)
    with col_region:
        region_first = st.toggle("📍 부산·경남·경북 우선", value=True)
    with col_refresh:
        st.write("")
        if st.button("🔄 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    with st.spinner("뉴스를 불러오는 중..."):
        articles = fetch_news(NEWS_FEEDS[feed_name])

    if not articles:
        st.warning("뉴스를 불러오지 못했습니다. 네트워크 상태를 확인하거나 다른 채널을 선택하세요.")
        return

    if region_first:
        articles = sorted(articles, key=lambda a: region_score(a["title"], a["summary"]), reverse=True)

    for art in articles:
        score = region_score(art["title"], art["summary"])
        badge = ""
        if "부산" in art["title"] + art["summary"]:
            badge = " 🔴 **부산**"
        elif "경남" in art["title"] + art["summary"]:
            badge = " 🟠 **경남**"
        elif "경북" in art["title"] + art["summary"]:
            badge = " 🟡 **경북**"
        elif score > 0:
            badge = " 📍"

        with st.container():
            st.markdown(f"**[{art['title']}]({art['link']})**{badge}")
            if art["published"]:
                st.caption(f"🕐 {art['published']}")
            if art["summary"]:
                clean = art["summary"].replace("<b>", "").replace("</b>", "")
                st.markdown(f"<small>{clean}...</small>", unsafe_allow_html=True)
            st.divider()


# ── 앱 시작 ─────────────────────────────────────────────
st.set_page_config(page_title="KDN 전력 분석", page_icon="⚡", layout="wide")

st.markdown("""
<style>
/* ── 전체 배경 ── */
.stApp { background-color: #0B1E3D; }

/* ── 탭 스타일 ── */
.stTabs [data-baseweb="tab-list"] {
    background: #112B52;
    border-radius: 10px;
    padding: 4px 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #8AACDF;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 14px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #4169E1 !important;
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }

/* ── 버튼 ── */
.stButton > button {
    background: linear-gradient(135deg, #4169E1, #1E3A8A);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #5A7FFF, #2B52CC);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(65,105,225,0.5);
}

/* ── 메트릭 카드 ── */
[data-testid="metric-container"] {
    background: #112B52;
    border: 1px solid #1E4080;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label { color: #8AACDF !important; font-size: 13px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 28px !important;
    font-weight: 700;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 13px; }

/* ── 입력 필드 ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #112B52 !important;
    border: 1px solid #1E4080 !important;
    border-radius: 8px !important;
    color: #D6E4FF !important;
}

/* ── 파일 업로더 ── */
[data-testid="stFileUploader"] {
    background: #112B52;
    border: 2px dashed #4169E1;
    border-radius: 10px;
}

/* ── 구분선 ── */
hr { border-color: #1E4080 !important; opacity: 0.5; }

/* ── 사이드바 ── */
[data-testid="stSidebar"] { background: #0D2242 !important; }
[data-testid="stSidebar"] * { color: #D6E4FF !important; }

/* ── 채팅 메시지 ── */
[data-testid="stChatMessage"] {
    background: #112B52;
    border-radius: 12px;
    border: 1px solid #1E4080;
}

/* ── 데이터프레임 ── */
[data-testid="stDataFrame"] { border: 1px solid #1E4080; border-radius: 8px; }

/* ── expander ── */
.streamlit-expanderHeader {
    background: #112B52 !important;
    border-radius: 8px !important;
    color: #D6E4FF !important;
}

/* ── 제목 ── */
h1 { color: #7EB3FF !important; }
h2, h3 { color: #5A9BFF !important; }
</style>
""", unsafe_allow_html=True)

tab_dash, tab_news, tab_settings, tab_chat = st.tabs([
    "⚡ 전력 대시보드", "🎣 낚시 뉴스", "⚙️ 설정", "💬 대화"
])

with tab_dash:
    render_dashboard()

with tab_news:
    render_news()

# ── 설정 탭 ──────────────────────────────────────────────
with tab_settings:
    st.title("⚙️ OpenAI API 설정")
    st.caption("API 키와 모델을 설정하고 저장합니다.")

    config = load_config()
    st.divider()

    st.subheader("🔑 API 키")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=config.get("api_key", ""),
        type="password",
        placeholder="sk-...",
        help="OpenAI 플랫폼(platform.openai.com)에서 발급받은 API 키를 입력하세요.",
    )

    st.subheader("🤖 모델 선택")
    current_model = config.get("model", "gpt-4o")
    model_index = OPENAI_MODELS.index(current_model) if current_model in OPENAI_MODELS else 0
    selected_model = st.selectbox("모델", OPENAI_MODELS, index=model_index)
    st.caption(f"ℹ️ {MODEL_DESC.get(selected_model, '')}")

    with st.expander("고급 설정"):
        temperature = st.slider("Temperature", 0.0, 2.0,
                                float(config.get("temperature", 0.7)), 0.1)
        max_tokens = st.number_input("Max Tokens", 64, 128000,
                                     int(config.get("max_tokens", 1024)), 64)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 저장", use_container_width=True, type="primary"):
            if not api_key_input:
                st.error("API 키를 입력하세요.")
            else:
                save_config({"api_key": api_key_input, "model": selected_model,
                             "temperature": temperature, "max_tokens": max_tokens})
                st.success("설정이 저장되었습니다.")
    with col2:
        if st.button("🔌 연결 테스트", use_container_width=True):
            if not api_key_input:
                st.error("API 키를 입력하세요.")
            else:
                with st.spinner("테스트 중..."):
                    success, message = test_api_key(api_key_input, selected_model)
                if success:
                    st.success(f"연결 성공!\n\n응답: {message}")
                else:
                    st.error(message)

    st.divider()
    st.subheader("📋 현재 저장된 설정")
    saved = load_config()
    c1, c2, c3 = st.columns(3)
    with c1:
        masked = f"{saved['api_key'][:8]}..." if len(saved.get("api_key", "")) > 8 else "미설정"
        st.metric("API 키", masked)
    with c2:
        st.metric("모델", saved.get("model", "미설정"))
    with c3:
        st.metric("Temperature", saved.get("temperature", "-"))


# ── 대화 탭 ──────────────────────────────────────────────
with tab_chat:
    st.title("💬 AI 대화")

    cfg = load_config()
    api_key = cfg.get("api_key", "")
    chat_model = cfg.get("model", "gpt-4o")
    chat_temp = float(cfg.get("temperature", 0.7))
    chat_max_tokens = int(cfg.get("max_tokens", 1024))

    if not api_key:
        st.warning("⚠️ 먼저 **설정** 탭에서 API 키를 저장하세요.")
        st.stop()

    info_col1, info_col2, info_col3 = st.columns([2, 2, 1])
    with info_col1:
        st.caption(f"모델: **{chat_model}**")
    with info_col2:
        st.caption(f"Temperature: **{chat_temp}**")
    with info_col3:
        if st.button("🗑️ 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                client = openai.OpenAI(api_key=api_key)
                is_o_model = chat_model.startswith("o1") or chat_model.startswith("o3")
                kwargs = {
                    "model": chat_model,
                    "messages": st.session_state.messages,
                    "max_tokens": chat_max_tokens,
                    "stream": not is_o_model,
                }
                if not is_o_model:
                    kwargs["temperature"] = chat_temp

                if is_o_model:
                    response = client.chat.completions.create(**kwargs)
                    full_response = response.choices[0].message.content
                    placeholder.markdown(full_response)
                else:
                    with client.chat.completions.create(**kwargs) as stream:
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content or ""
                            full_response += delta
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

            except openai.AuthenticationError:
                full_response = "API 키가 유효하지 않습니다. 설정 탭에서 확인하세요."
                placeholder.error(full_response)
            except Exception as e:
                full_response = f"오류가 발생했습니다: {str(e)}"
                placeholder.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
