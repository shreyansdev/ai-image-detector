#!/usr/bin/env python3
import os
import sys
import subprocess
import platform

BANNER = """
=================================================================
             AI IMAGE DETECTOR - RUN ON THE GO
=================================================================
"""

def main():
    print(BANNER)
    
    # 1. Check Python version
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        print(f"[!] Warning: Python 3.10+ is recommended. Found Python {major}.{minor}.")
        # We'll try to proceed anyway, but warn the user.
        
    # Get the project root directory (directory where run.py is located)
    project_root = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(project_root, "backend")
    
    # 2. Virtual environment path setup
    venv_dir = os.path.join(backend_dir, ".venv")
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_executable = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python_executable = os.path.join(venv_dir, "bin", "python")
        pip_executable = os.path.join(venv_dir, "bin", "pip")

    # 3. Create virtual environment if it doesn't exist
    if not os.path.exists(python_executable):
        print("[*] Creating Python virtual environment in 'backend/.venv'...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            print("[+] Virtual environment created successfully.")
        except Exception as e:
            print(f"[-] Error creating virtual environment: {e}")
            print("[!] Please install virtualenv or check permissions.")
            sys.exit(1)
            
    # 4. Install requirements
    print("[*] Checking / Installing dependencies...")
    reqs_path = os.path.join(backend_dir, "requirements.txt")
    if os.path.exists(reqs_path):
        try:
            # Upgrade pip first
            subprocess.run([python_executable, "-m", "pip", "install", "--upgrade", "pip"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Install requirements
            subprocess.run([pip_executable, "install", "-r", reqs_path], check=True)
            print("[+] Dependencies successfully verified/installed.")
        except Exception as e:
            print(f"[-] Error installing dependencies: {e}")
            print("[!] Try running requirements installation manually.")
            sys.exit(1)
    else:
        print(f"[-] Error: requirements.txt not found at {reqs_path}")
        sys.exit(1)

    # 5. Check if model weights exist
    weights_path = os.path.join(backend_dir, "ml", "model_weights", "best_efficientnet_b0.pth")
    if not os.path.exists(weights_path):
        print(f"[!] Warning: Trained model weights not found at {weights_path}")
        print("    The application will fall back to using an uninitialized EfficientNet-B0 backbone.")
        print("    To train the model, refer to the README.md instructions.")

    # 6. Check if frontend distribution build exists; if missing, build it automatically
    frontend_dir = os.path.join(project_root, "frontend")
    frontend_dist_index = os.path.join(frontend_dir, "dist", "index.html")
    if not os.path.exists(frontend_dist_index):
        print("[*] Frontend distribution build not found in 'frontend/dist'.")
        npm_cmd = "npm.cmd" if is_windows else "npm"
        try:
            print("[*] Running 'npm install' in frontend directory...")
            subprocess.run([npm_cmd, "install"], cwd=frontend_dir, check=True)
            print("[*] Running 'npm run build' in frontend directory...")
            subprocess.run([npm_cmd, "run", "build"], cwd=frontend_dir, check=True)
            print("[+] Frontend build completed successfully.")
        except FileNotFoundError:
            print("[!] Warning: npm is not installed or not in PATH. Frontend SPA will not be served unless built.")
        except Exception as e:
            print(f"[!] Warning: Failed to build frontend: {e}")

    # 7. Run FastAPI application
    print("\n[*] Starting AI Image Detector server...")
    print("[*] Access the application in your browser at: http://localhost:8000")
    print("[*] Press Ctrl+C to stop the server.\n")
    
    # We run uvicorn from the backend directory to ensure .env is correctly loaded
    try:
        # We use python -m uvicorn app.main:app instead of direct uvicorn to ensure
        # python paths are set up correctly relative to virtual environment.
        subprocess.run(
            [
                python_executable, "-m", "uvicorn", "app.main:app",
                "--host", "0.0.0.0", "--port", "8000",
                "--proxy-headers", "--forwarded-allow-ips", "*"
            ],
            cwd=backend_dir
        )
    except KeyboardInterrupt:
        print("\n[+] Server stopped by user.")
    except Exception as e:
        print(f"[-] Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
