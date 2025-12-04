import streamlit as st


def render_history_download():
    if st.session_state.get("message"):
        chat_texts="\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("Download Chat history",chat_texts,file_name="chat_history.txt",mime="text/plain")