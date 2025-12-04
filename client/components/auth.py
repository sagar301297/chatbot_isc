import streamlit as st
from config import PASSWORD

def check_password():
    """Returns True if the user has entered the correct password."""
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # If already authenticated, return True
    if st.session_state.authenticated:
        return True

    # Show login form
    st.markdown("# Login Required")
    st.markdown("Please enter the password to access the chatbot.")
    
    # Password input
    password = st.text_input("Password", type="password", key="password_input")
    
    if st.button("Login", type="primary"):
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.success("Authentication successful!")
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    
    return False