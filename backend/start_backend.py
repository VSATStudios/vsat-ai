#!/usr/bin/env python3
"""
VSAT AI Backend Starter Script
This script helps set up and start the VSAT AI backend with proper error handling.
"""

import subprocess
import sys
import os
import time

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        print("💡 Try running: pip install -r requirements.txt manually")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def check_ollama():
    """Check if Ollama is available"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
        else:
            print("⚠️ Ollama is installed but not responding properly")
            return False
    except FileNotFoundError:
        print("⚠️ Ollama is not installed or not in PATH")
        print("💡 Install Ollama from: https://ollama.ai/")
        return False

def start_ollama():
    """Start Ollama service"""
    print("🚀 Starting Ollama service...")
    try:
        # Try to start ollama serve in background
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # Give it time to start
        
        # Check if it's running
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama service is running")
            return True
        else:
            print("⚠️ Ollama service may not be running properly")
            return False
    except Exception as e:
        print(f"⚠️ Could not start Ollama automatically: {e}")
        print("💡 Please start Ollama manually: ollama serve")
        return False

def pull_ollama_model():
    """Pull a default model for Ollama"""
    print("📥 Checking for Ollama models...")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            models = result.stdout.strip()
            if "llama2" in models or "mistral" in models or "codellama" in models:
                print("✅ Ollama models are available")
                return True
            else:
                print("📦 No models found, pulling llama2 (this may take a while)...")
                subprocess.run(["ollama", "pull", "llama2"], timeout=600)
                print("✅ Model downloaded successfully")
                return True
    except subprocess.TimeoutExpired:
        print("⏱️ Model download timed out, but continuing...")
        return True
    except Exception as e:
        print(f"⚠️ Could not pull model: {e}")
        print("💡 You can manually pull a model later: ollama pull llama2")
        return True

def start_backend():
    """Start the Flask backend"""
    print("🚀 Starting VSAT AI Backend...")
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 VSAT AI Backend stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def main():
    """Main setup and start function"""
    print("=" * 60)
    print("🚀 VSAT AI Backend Setup & Starter")
    print("👨‍💻 Developed by Vedant Roy")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Cannot continue without required packages")
        return
    
    # Check and setup Ollama
    if check_ollama():
        start_ollama()
        pull_ollama_model()
    else:
        print("⚠️ Continuing without Ollama (chat functionality will be limited)")
    
    print("\n" + "=" * 60)
    print("✅ Setup complete! Starting backend...")
    print("💻 Backend will run on: http://localhost:5000")
    print("🔧 Make sure to start frontend with: npm run dev")
    print("=" * 60)
    
    # Start the backend
    start_backend()

if __name__ == "__main__":
    main()