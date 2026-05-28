# KDN6 - OpenAI 설정 도구

OpenAI API 키와 모델을 Streamlit 으로 설정하는 앱입니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 기능

- OpenAI API 키 설정 및 저장
- 모델 선택 (gpt-4o, gpt-4, gpt-3.5-turbo, o1 등)
- Temperature / Max Tokens 고급 설정
- 실시간 API 연결 테스트

## 설정 저장

설정은 `config.json` 에 저장됩니다. (`.gitignore` 에 포함되어 Git에 업로드되지 않습니다.)
