"""Запуск Streamlit дашборда."""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard/app.py"
    ])

