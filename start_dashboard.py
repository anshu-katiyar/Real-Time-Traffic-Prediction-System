import subprocess
import sys
import os

def start_dashboard():
    """Start the Streamlit dashboard"""
    try:
        print("Starting Traffic Prediction Dashboard...")
        print("Dashboard will open at: http://localhost:8501")
        
        # Run streamlit dashboard
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], 
                      cwd=os.getcwd())
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        print("Try running manually: streamlit run dashboard.py")

if __name__ == "__main__":
    start_dashboard()