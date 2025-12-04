import streamlit as st
from components.upload import render_uploader
from components.history_download import render_history_download
from components.chatUi import render_chat
from components.auth import check_password

st.set_page_config(page_title="ISC chatbot",layout="wide")
st.title=("Chatbot")

if not check_password():
    st.stop() 

render_uploader()
render_chat()
render_history_download()