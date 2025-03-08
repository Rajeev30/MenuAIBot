import streamlit as st
from streamlit_lottie import st_lottie
from Bot_code import build_conversational_chain


st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("MenuData Chatbot")

if "chain" not in st.session_state:
    st.session_state.chain = build_conversational_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

def load_lottie_url(url: str):
    import requests
    res = requests.get(url)
    if res.status_code != 200:
        return None
    return res.json()

lottie_loader = load_lottie_url("https://lottie.host/81c89726-313e-46b8-960e-3a1c506ed78c/oW4uJp8RSp.json")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type your query here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)  

    loader_placeholder = st.empty()
    with loader_placeholder:
        st_lottie(lottie_loader, height=40, width=40)

    chain = st.session_state.chain
    result = chain.invoke({"question": user_input})

    bot_response = result.get("answer", "").strip()
    if not bot_response:
        bot_response = "No response generated!"

    loader_placeholder.empty()

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.write(bot_response)
