import subprocess
import time

# Function to run Flask API
def run_flask():
    subprocess.run(["python", "data.py"])

# Function to run Streamlit Dashboard
def run_streamlit():
    subprocess.run(["streamlit", "run", "dashboard.py"])

if __name__ == "__main__":
    # Start Flask API in a separate process
    flask_process = subprocess.Popen(["python", "data.py"])

    # Wait for a moment to ensure Flask API is up and running
    time.sleep(2)

    # Start Streamlit Dashboard
    streamlit_process = subprocess.Popen(["streamlit", "run", "dashboard.py"])

    # Wait for the processes to finish
    flask_process.wait()
    streamlit_process.wait()
