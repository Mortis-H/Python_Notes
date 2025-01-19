import streamlit as st
import requests

# 設定 LM Studio API 的基本 URL
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

# 初始化聊天歷史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 主要介面
st.title("本地 LLM 聊天室")

# 顯示聊天記錄
for message in st.session_state.chat_history:
    role = "🧑 使用者" if message["role"] == "user" else "🤖 助手"
    st.write(f"{role}：{message['content']}")

# 使用者輸入
user_input = st.text_area("請輸入訊息", height=100, key="user_input")

# 送出按鈕
if st.button("送出") and user_input.strip():
    # 將用戶輸入加入聊天記錄
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 準備對話歷史
    messages = [
        {"role": "system", "content": "你是一個有幫助的AI助手。請用中文回答。"},
    ] + [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.chat_history
    ]

    # 發送請求到 LM Studio
    try:
        payload = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }
        
        response = requests.post(
            LM_STUDIO_API_URL, 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        model_reply = response.json()["choices"][0]["message"]["content"].strip()
        st.session_state.chat_history.append({"role": "assistant", "content": model_reply})
        
    except Exception as e:
        st.error(f"與模型通訊時發生錯誤: {str(e)}")

# 清空聊天記錄按鈕
if st.button("清空聊天記錄"):
    st.session_state.chat_history = []
    st.experimental_rerun()
