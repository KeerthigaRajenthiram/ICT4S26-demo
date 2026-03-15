from pathlib import Path

STRUCTURE = [
    "streamlit_app/app.py",
    "streamlit_app/ui/__init__.py",
    "streamlit_app/ui/state.py",
    "streamlit_app/ui/styles.py",
    "streamlit_app/ui/components.py",
    "streamlit_app/data/__init__.py",
    "streamlit_app/data/access.py",
    "streamlit_app/data/recommend.py",
]

def scaffold():
    base = Path.cwd()  # you are already inside GUI/

    for rel_path in STRUCTURE:
        path = base / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            path.touch()
            print(f"Created: {path}")
        else:
            print(f"Exists:  {path}")

if __name__ == "__main__":
    scaffold()
