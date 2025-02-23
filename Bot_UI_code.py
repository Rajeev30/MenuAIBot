import streamlit as st
from Bot_code import build_conversational_chain

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("MenuData Chatbot")

if "chain" not in st.session_state:
    st.session_state.chain = build_conversational_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type your query here...")
# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})


if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    chain = st.session_state.chain
    result = chain.invoke({"question" :user_input})

    bot_response = result.get("answer", "").strip()

    if not bot_response:
        bot_response = "⚠️ No response generated!"
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.rerun()
