import streamlit as st

def reset_session() -> None:
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
