# VSAT AI - Backend (Flask)
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import base64
from io import BytesIO
from PIL import Image
import torch
import os
import time
import logging
import json
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

# Global variables
pipe = None
model_loaded = False

def install_dependencies():
    """Install required dependencies if not available"""
    try:
        import diffusers
        import transformers
        import accelerate
        logger.info("✅ All dependencies are available")
        return True
    except ImportError as e:
        logger.info(f"📦 Installing missing dependencies: {e}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "diffusers", "transformers", "accelerate", "safetensors"
            ])
            logger.info("✅ Dependencies installed successfully")
            return True
        except Exception as install_error:
            logger.error(f"❌ Failed to install dependencies: {install_error}")
            return False

def load_stable_diffusion():
    """Load Stable Diffusion model with error handling"""
    global pipe, model_loaded
    
    if not install_dependencies():
        logger.error("❌ Cannot load Stable Diffusion due to missing dependencies")
        return
    
    try:
        logger.info("Loading Stable Diffusion model...")
        
        from diffusers import StableDiffusionPipeline
        
        # Use a smaller, faster model for better performance
        model_id = "runwayml/stable-diffusion-v1-5"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        # Optimize for better performance
        pipe.enable_attention_slicing()
        if hasattr(pipe, 'enable_model_cpu_offload'):
            pipe.enable_model_cpu_offload()
        
        # Use GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            pipe = pipe.to(device)
            logger.info("🔧 Using CPU for image generation (slower but works)")
        else:
            pipe = pipe.to(device)
            logger.info("🚀 Using GPU for image generation")
        
        model_loaded = True
        logger.info(f"✅ Stable Diffusion loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"❌ Error loading Stable Diffusion: {e}")
        logger.info("💡 Image generation will be unavailable")
        model_loaded = False

def check_ollama():
    """Check if Ollama is running and has models"""
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, "Ollama not responding"
    except FileNotFoundError:
        return False, "Ollama not installed"
    except Exception as e:
        return False, str(e)

def ensure_ollama_model():
    """Ensure Ollama has a model available"""
    try:
        # Check if any model is available
        is_running, output = check_ollama()
        if not is_running:
            logger.warning("⚠️ Ollama is not running")
            return False
        
        # If no models, try to pull a lightweight one
        if "NAME" in output and len(output.strip().split('\n')) <= 1:
            logger.info("📦 No Ollama models found, pulling llama2...")
            subprocess.run(["ollama", "pull", "llama2"], timeout=300)
        
        return True
    except Exception as e:
        logger.error(f"❌ Error with Ollama setup: {e}")
        return False

def generate_chat_response(prompt):
    """Generate chatbot response with custom logic and Ollama fallback"""
    prompt_lower = prompt.lower().strip()
    
    # Custom responses for VSAT AI
    if any(phrase in prompt_lower for phrase in ["who made you", "who is your developer", "who created you"]):
        return "I was developed by Vedant Roy 👨‍💻, the creator of VSAT AI (Vedant's Smart AI Technology)!"
    
    if any(phrase in prompt_lower for phrase in ["what is your name", "who are you", "what are you"]):
        return "I am VSAT AI 🤖, your intelligent assistant for chat and image generation! I can help you with conversations and create amazing images from text descriptions."
    
    if "vsat" in prompt_lower:
        return "VSAT AI stands for Vedant's Smart AI Technology! I'm here to help you with conversations and create amazing images. I combine the power of language models with image generation capabilities. 🚀"
    
    if any(phrase in prompt_lower for phrase in ["hello", "hi", "hey"]):
        return "Hello! 👋 I'm VSAT AI, your intelligent assistant. I can chat with you about anything or generate beautiful images from your descriptions. What would you like to do today?"
    
    # Try Ollama for other queries
    try:
        logger.info(f"🤖 Calling Ollama with prompt: {prompt[:50]}...")
        
        # Try different models in order of preference
        models_to_try = ["mistral", "llama2", "codellama"]
        
        for model in models_to_try:
            try:
                command = ["ollama", "run", model, prompt]
                result = subprocess.run(
                    command, 
                    capture_output=True, 
                    text=True, 
                    timeout=30,
                    cwd=os.getcwd()
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    response = result.stdout.strip()
                    logger.info(f"✅ Ollama response received from {model}")
                    return response
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"⏱️ Timeout with model {model}")
                continue
            except Exception as e:
                logger.warning(f"⚠️ Error with model {model}: {e}")
                continue
        
        # If all models fail, return helpful message
        return "I'm having trouble connecting to my language model, but I'm still here to help! You can ask me about VSAT AI, request image generation, or try asking a different question. 😊"
            
    except Exception as e:
        logger.error(f"❌ Ollama error: {e}")
        return "I encountered a small hiccup, but I'm still functional! Try asking me about VSAT AI or request an image generation. You can also try restarting Ollama if you have it installed. 🔧"

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'VSAT AI Backend is running! 🚀',
        'version': '1.0.0',
        'developer': 'Vedant Roy',
        'endpoints': ['/chat', '/generate-image', '/status'],
        'message': 'Welcome to VSAT AI - Vedant\'s Smart AI Technology!'
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Please provide a prompt.'}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty.'}), 400
        
        logger.info(f"💬 Chat request: {prompt[:100]}...")
        response = generate_chat_response(prompt)
        
        return jsonify({
            'response': response,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        return jsonify({
            'response': 'I apologize, but I encountered an error. Please try again! I\'m still here to help with conversations and image generation. 😊',
            'timestamp': time.time()
        }), 200

@app.route('/generate-image', methods=['POST'])
def generate_image():
    """Handle image generation requests"""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Please provide an image prompt.'}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Image prompt cannot be empty.'}), 400
        import requests
        response = requests.post('http://127.0.0.1:7860/sdapi/v1/txt2img', json={
            'prompt': prompt,
            'steps': 20})
        
        logger.info(f"🎨 Image generation request: {prompt}")
        
        if not model_loaded or pipe is None:
            # Try to load the model again
            load_stable_diffusion()
            if not model_loaded:
                return jsonify({
                    'error': 'Image generation model is not available. Please ensure PyTorch and diffusers are installed correctly. You can install them with: pip install torch diffusers transformers'
                }), 500
        
        # Generate image with error handling
        try:
            with torch.no_grad():
                # Use lower settings for faster generation
                image = pipe(
                    prompt,
                    num_inference_steps=15,  # Reduced for speed
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator().manual_seed(42)  # For reproducible results
                ).images[0]
        except Exception as gen_error:
            logger.error(f"❌ Image generation failed: {gen_error}")
            return jsonify({
                'error': f'Image generation failed: {str(gen_error)}. This might be due to insufficient memory or model issues.'
            }), 500
        
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Create images directory if it doesn't exist
        os.makedirs("generated_images", exist_ok=True)
        
        # Save locally for debugging
        timestamp = int(time.time())
        filename = f"generated_{timestamp}.png"
        image.save(f"generated_images/{filename}")
        logger.info(f"💾 Image saved as: generated_images/{filename}")
        
        # Encode for web
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'image': f"data:image/png;base64,{encoded_image}",
            'prompt': prompt,
            'timestamp': timestamp
        })
        
    except Exception as e:
        logger.error(f"❌ Image generation error: {e}")
        return jsonify({
            'error': f'Image generation failed: {str(e)}. Please ensure all dependencies are installed correctly.'
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Check system status"""
    # Check Ollama status
    ollama_running, ollama_info = check_ollama()
    ollama_status = "Running" if ollama_running else "Not Running"
    
    if ollama_running and "mistral" in ollama_info:
        ollama_status = "Running (Mistral loaded)"
    elif ollama_running and "llama2" in ollama_info:
        ollama_status = "Running (Llama2 loaded)"
    elif ollama_running:
        ollama_status = "Running (Models available)"
    
    return jsonify({
        'vsat_ai': 'Running',
        'ollama': ollama_status,
        'stable_diffusion': 'Loaded' if model_loaded else 'Not Loaded',
        'device': 'CUDA' if torch.cuda.is_available() else 'CPU',
        'torch_version': torch.__version__,
        'developer': 'Vedant Roy',
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found. Available endpoints: /, /chat, /generate-image, /status'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error. Please check the logs.'}), 500

if __name__ == '__main__':
    print("🚀 Starting VSAT AI - Vedant's Smart AI Technology...")
    print("👨‍💻 Developed by Vedant Roy")
    print("📦 Loading AI models...")
    
    # Ensure Ollama is set up
    ensure_ollama_model()
    
    # Load Stable Diffusion model
    load_stable_diffusion()
    
    print("🌐 Starting Flask server...")
    print("💻 Backend running at: http://localhost:5000")
    print("🔧 Frontend should connect automatically")
    print("📱 Make sure to start your frontend with: npm run dev")
    
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
    from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# 💬 Route to interact with Ollama Mistral
@app.route("/api/chat", methods=["POST"])
def chat_with_ollama():
    prompt = request.json.get("prompt")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    data = response.json()
    return jsonify({"response": data.get("response")})

