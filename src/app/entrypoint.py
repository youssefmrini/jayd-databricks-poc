import os

port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
os.execvp("streamlit", [
    "streamlit", "run", "app.py",
    "--server.port", str(port),
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false",
])
