import streamlit as st
import openai
import json
import os
from pathlib import Path

CONFIG_FILE = Path("config.json")

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o3-mini",
]

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


st.set_page_config(page_title="OpenAI 설정", page_icon="⚙️", layout="centered")

st.title("⚙️ OpenAI API 설정")
st.caption("API 키와 모델을 설정하고 저장합니다.")

config = load_config()

st.divider()

# API Key
st.subheader("🔑 API 키")
api_key_input = st.text_input(
    "OpenAI API Key",
    value=config.get("api_key", ""),
    type="password",
    placeholder="sk-...",
    help="OpenAI 플랫폼(platform.openai.com)에서 발급받은 API 키를 입력하세요.",
)

# Model
st.subheader("🤖 모델 선택")
current_model = config.get("model", "gpt-4o")
model_index = OPENAI_MODELS.index(current_model) if current_model in OPENAI_MODELS else 0
selected_model = st.selectbox("모델", OPENAI_MODELS, index=model_index)

model_descriptions = {
    "gpt-4o": "최신 멀티모달 모델, 빠르고 저렴 (권장)",
    "gpt-4o-mini": "경량화 버전, 간단한 작업에 적합",
    "gpt-4-turbo": "GPT-4 터보, 128K 컨텍스트",
    "gpt-4": "GPT-4 기본 모델",
    "gpt-3.5-turbo": "빠르고 경제적, 간단한 작업",
    "o1": "추론 특화 모델, 복잡한 문제 해결",
    "o1-mini": "경량 추론 모델",
    "o3-mini": "최신 추론 모델 (미니)",
}
st.caption(f"ℹ️ {model_descriptions.get(selected_model, '')}")

# Advanced settings
with st.expander("고급 설정"):
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(config.get("temperature", 0.7)),
        step=0.1,
        help="높을수록 창의적, 낮을수록 일관된 응답",
    )
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=64,
        max_value=128000,
        value=int(config.get("max_tokens", 1024)),
        step=64,
        help="응답의 최대 토큰 수",
    )

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("💾 저장", use_container_width=True, type="primary"):
        if not api_key_input:
            st.error("API 키를 입력하세요.")
        else:
            new_config = {
                "api_key": api_key_input,
                "model": selected_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            save_config(new_config)
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

# Current status
st.divider()
st.subheader("📋 현재 저장된 설정")
saved = load_config()
status_cols = st.columns(3)
with status_cols[0]:
    masked_key = f"{saved['api_key'][:8]}..." if len(saved.get("api_key", "")) > 8 else "미설정"
    st.metric("API 키", masked_key)
with status_cols[1]:
    st.metric("모델", saved.get("model", "미설정"))
with status_cols[2]:
    st.metric("Temperature", saved.get("temperature", "-"))
