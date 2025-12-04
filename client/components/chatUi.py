import streamlit as st
from utils.api import ask_question, reset_chat

def render_chat():
    st.subheader("💬 Chat with your documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add reset button in the sidebar or above chat
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Reset Chat", type="secondary", use_container_width=True):
            # Call the reset API
            response = reset_chat()
            if response.status_code == 200:
                # Clear the chat history
                st.session_state.messages = []
                st.success("Chat reset successfully!")
                st.rerun()
            else:
                st.error(f"❌ Error resetting chat: {response.text}")

    # Render existing chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Input and response
    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = ask_question(user_input)
        if response.status_code == 200:
            data = response.json()
            answer = data["response"]
            sources = data.get("sources", [])
            st.chat_message("assistant").markdown(answer)
            if sources:
                st.markdown("📄 **Sources:**")
                for src in sources:
                    st.markdown(f"- `{src}`")
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error(f"Error: {response.text}")