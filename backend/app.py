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
import requests

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
        # First check if Ollama service is running
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            timeout=10  # Increased timeout
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, "Ollama not responding"
    except subprocess.TimeoutExpired:
        return False, "Ollama timeout - service might be slow to respond"
    except FileNotFoundError:
        return False, "Ollama not installed"
    except Exception as e:
        return False, str(e)

def check_ollama_api():
    """Check if Ollama API is accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, [model['name'] for model in models]
        return False, []
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Ollama API not accessible: {e}")
        return False, []

def ensure_ollama_model():
    """Ensure Ollama has a model available"""
    try:
        # Check API first
        api_running, available_models = check_ollama_api()
        if not api_running:
            logger.warning("⚠️ Ollama API is not accessible at localhost:11434")
            return False
        
        if not available_models:
            logger.info("📦 No Ollama models found, pulling mistral...")
            subprocess.run(["ollama", "pull", "mistral"], timeout=600)  # 10 minutes timeout
            return True
        
        logger.info(f"✅ Available Ollama models: {available_models}")
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
    
    # Try Ollama API first (more reliable)
    try:
        logger.info(f"🤖 Calling Ollama API with prompt: {prompt[:50]}...")
        
        # Check if models are available
        api_running, available_models = check_ollama_api()
        if not api_running or not available_models:
            return "I'm having trouble connecting to my language model. Please make sure Ollama is running with: 'ollama serve' and that you have models installed. 🔧"
        
        # Try to use the best available model
        preferred_models = ["mistral", "llama2", "codellama", "tinyllama"]
        model_to_use = None
        
        for preferred in preferred_models:
            for available in available_models:
                if preferred in available.lower():
                    model_to_use = available
                    break
            if model_to_use:
                break
        
        if not model_to_use:
            model_to_use = available_models[0]  # Use first available model
        
        logger.info(f"🎯 Using model: {model_to_use}")
        
        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 200  # Limit response length for faster generation
                }
            },
            timeout=60  # Increased timeout to 60 seconds
        )
        
        if response.status_code == 200:
            data = response.json()
            ollama_response = data.get("response", "").strip()
            if ollama_response:
                logger.info(f"✅ Ollama API response received from {model_to_use}")
                return ollama_response
        else:
            logger.error(f"❌ Ollama API returned status code: {response.status_code}")
    
    except requests.exceptions.Timeout:
        logger.error("⏱️ Ollama API timeout - the model might be processing a complex request")
        return "I'm taking a bit longer to process your request. The AI model might be handling a complex query. Please try a simpler question or wait a moment and try again. 🤖"
    
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Cannot connect to Ollama API")
        return "I can't connect to my language model. Please make sure Ollama is running with 'ollama serve' command. 🔧"
    
    except Exception as e:
        logger.error(f"❌ Ollama API error: {e}")
    
    # Fallback to subprocess method if API fails
    try:
        logger.info("🔄 Trying subprocess method as fallback...")
        
        models_to_try = ["mistral", "llama2", "codellama"]
        
        for model in models_to_try:
            try:
                command = ["ollama", "run", model, prompt]
                result = subprocess.run(
                    command, 
                    capture_output=True, 
                    text=True, 
                    timeout=90,  # Increased timeout
                    cwd=os.getcwd()
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    response = result.stdout.strip()
                    logger.info(f"✅ Ollama subprocess response received from {model}")
                    return response
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"⏱️ Timeout with model {model} (subprocess)")
                continue
            except Exception as e:
                logger.warning(f"⚠️ Error with model {model}: {e}")
                continue
        
        # If all methods fail, return helpful message
        return "I'm having trouble connecting to my language model. Please check that Ollama is running ('ollama serve') and you have models installed ('ollama pull mistral'). You can still ask me about VSAT AI or request image generation! 😊"
            
    except Exception as e:
        logger.error(f"❌ All Ollama methods failed: {e}")
        return "I encountered an issue with my language model, but I'm still here to help! Try asking me about VSAT AI, request an image generation, or check if Ollama is running properly. 🔧"

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
        
        logger.info(f"🎨 Image generation request: {prompt}")
        
        # Try Automatic1111 WebUI first (if available)
        try:
            logger.info("🔍 Checking for Automatic1111 WebUI...")
            webui_response = requests.post(
                'http://127.0.0.1:7860/sdapi/v1/txt2img', 
                json={
                    'prompt': prompt,
                    'steps': 20,
                    'width': 512,
                    'height': 512,
                    'cfg_scale': 7.5
                },
                timeout=30
            )
            
            if webui_response.status_code == 200:
                logger.info("✅ Using Automatic1111 WebUI for image generation")
                webui_data = webui_response.json()
                if 'images' in webui_data and webui_data['images']:
                    # Save the generated image
                    os.makedirs("generated_images", exist_ok=True)
                    timestamp = int(time.time())
                    filename = f"generated_{timestamp}.png"
                    
                    # Decode and save image
                    image_data = base64.b64decode(webui_data['images'][0])
                    with open(f"generated_images/{filename}", "wb") as f:
                        f.write(image_data)
                    logger.info(f"💾 Image saved as: generated_images/{filename}")
                    
                    return jsonify({
                        'image': f"data:image/png;base64,{webui_data['images'][0]}",
                        'prompt': prompt,
                        'timestamp': timestamp,
                        'method': 'Automatic1111 WebUI'
                    })
        
        except requests.exceptions.RequestException as webui_error:
            logger.info(f"⚠️ Automatic1111 WebUI not available: {webui_error}")
            logger.info("🔄 Falling back to local Stable Diffusion...")
        
        # Fallback to local Stable Diffusion pipeline
        if not model_loaded or pipe is None:
            logger.info("📦 Loading Stable Diffusion model...")
            load_stable_diffusion()
            if not model_loaded:
                return jsonify({
                    'error': 'Image generation is not available. Please either:\n1. Start Automatic1111 WebUI at http://127.0.0.1:7860\n2. Install PyTorch and diffusers: pip install torch diffusers transformers accelerate',
                    'suggestion': 'For faster image generation, we recommend using Automatic1111 WebUI'
                }), 500
        
        # Generate image with local pipeline
        try:
            logger.info("🎨 Generating image with local Stable Diffusion...")
            with torch.no_grad():
                # Clear GPU cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Generate image with optimized settings
                image = pipe(
                    prompt,
                    num_inference_steps=20,  # Good balance of quality and speed
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator().manual_seed(int(time.time()))  # Random seed for variety
                ).images[0]
                
        except RuntimeError as gpu_error:
            if "out of memory" in str(gpu_error).lower():
                logger.warning("⚠️ GPU out of memory, trying CPU...")
                try:
                    # Move to CPU and try again
                    pipe = pipe.to("cpu")
                    with torch.no_grad():
                        image = pipe(
                            prompt,
                            num_inference_steps=15,  # Reduced steps for CPU
                            guidance_scale=7.5,
                            height=512,
                            width=512
                        ).images[0]
                except Exception as cpu_error:
                    logger.error(f"❌ CPU generation also failed: {cpu_error}")
                    return jsonify({
                        'error': 'Image generation failed due to memory constraints. Try using Automatic1111 WebUI or reducing system load.'
                    }), 500
            else:
                logger.error(f"❌ Image generation failed: {gpu_error}")
                return jsonify({
                    'error': f'Image generation failed: {str(gpu_error)}'
                }), 500
        
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
            'timestamp': timestamp,
            'method': 'Local Stable Diffusion'
        })
        
    except Exception as e:
        logger.error(f"❌ Image generation error: {e}")
        return jsonify({
            'error': f'Image generation failed: {str(e)}. Please ensure dependencies are installed or try using Automatic1111 WebUI.'
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Check system status"""
    # Check Ollama status
    ollama_running, ollama_info = check_ollama()
    api_running, available_models = check_ollama_api()
    
    # Check Automatic1111 WebUI
    webui_available = False
    try:
        webui_response = requests.get("http://127.0.0.1:7860", timeout=5)
        webui_available = webui_response.status_code == 200
    except:
        webui_available = False
    
    if api_running and available_models:
        ollama_status = f"Running (Models: {', '.join(available_models[:3])})"  # Show first 3 models
    elif ollama_running:
        ollama_status = "Running (CLI available)"
    else:
        ollama_status = "Not Running"
    
    return jsonify({
        'vsat_ai': 'Running',
        'ollama_cli': 'Running' if ollama_running else 'Not Running',
        'ollama_api': 'Running' if api_running else 'Not Running',
        'ollama_models': available_models if api_running else [],
        'stable_diffusion_local': 'Loaded' if model_loaded else 'Not Loaded',
        'automatic1111_webui': 'Available' if webui_available else 'Not Available',
        'image_generation': 'Available' if (model_loaded or webui_available) else 'Unavailable',
        'device': 'CUDA' if torch.cuda.is_available() else 'CPU',
        'torch_version': torch.__version__,
        'developer': 'Vedant Roy',
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'recommendations': {
            'chat': 'Install and run Ollama with Mistral model' if not api_running else 'Working perfectly!',
            'images': 'Use Automatic1111 WebUI for best performance' if not webui_available else 'WebUI available - optimal setup!'
        }
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
    print("\n🔍 To check system status, visit: http://localhost:5000/status")
    
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
