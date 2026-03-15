import streamlit as st

def apply_global_styles() -> None:
    # Professional, minimal polish without fighting Streamlit theming.
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2rem; padding-bottom: 3rem; }

          /* Make metrics feel more "dashboard" */
          [data-testid="stMetricValue"] { font-size: 1.5rem; }
          [data-testid="stMetricLabel"] { font-size: 0.9rem; opacity: 0.85; }

          /* Card helper */
          .kb-card {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 16px;
            padding: 16px 18px;
            background: rgba(255,255,255,0.6);
          }
          .kb-muted { color: rgba(49, 51, 63, 0.70); }
          
          
  
        </style>
        """,
        unsafe_allow_html=True,
    )

