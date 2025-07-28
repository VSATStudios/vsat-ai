# VSAT AI - Vedant's Smart AI Technology

A powerful AI assistant that combines chat capabilities with image generation, built with React and Flask.

## Features

- 🤖 **Intelligent Chat**: Powered by Ollama for natural conversations
- 🎨 **Image Generation**: Create stunning images from text descriptions using Stable Diffusion
- 🎯 **Dual Mode Interface**: Switch between chat and image generation modes
- 🚀 **Real-time Status**: Monitor system components and performance
- 💫 **Beautiful UI**: Modern, responsive design with smooth animations
- 🔧 **Easy Setup**: Automated installation and configuration

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- Ollama (optional, for enhanced chat features)

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Run the setup script (recommended):
```bash
python start_backend.py
```

Or manually:
```bash
pip install -r requirements.txt
python app.py
```

### Frontend Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser to `http://localhost:5173`

## System Requirements

### For Chat Features
- **Ollama** (recommended): Install from [ollama.ai](https://ollama.ai/)
- Models will be automatically downloaded on first use

### For Image Generation
- **PyTorch**: Automatically installed with requirements
- **GPU** (optional): CUDA-compatible GPU for faster generation
- **RAM**: At least 8GB recommended for image generation

## Usage

### Chat Mode
- Ask questions about anything
- Get information about VSAT AI
- Natural conversation powered by Ollama

### Image Generation Mode
- Describe the image you want to create
- Wait for the AI to generate your image
- Images are automatically saved locally

## API Endpoints

- `GET /` - Health check and system info
- `POST /chat` - Send chat messages
- `POST /generate-image` - Generate images from text
- `GET /status` - Check system component status

## Configuration

The system automatically configures itself, but you can customize:

- **Backend Port**: Set `PORT` environment variable (default: 5000)
- **Frontend Port**: Configured in Vite (default: 5173)
- **Models**: Ollama models are auto-downloaded

## Troubleshooting

### Common Issues

1. **Ollama not found**: Install Ollama from the official website
2. **Image generation fails**: Ensure PyTorch is properly installed
3. **CORS errors**: Make sure both frontend and backend are running
4. **Memory issues**: Close other applications or use CPU mode

### Getting Help

1. Check the console logs for detailed error messages
2. Verify all dependencies are installed
3. Ensure ports 5000 and 5173 are available
4. Try restarting both frontend and backend

## Development

### Project Structure
```
├── src/                 # React frontend
│   ├── App.tsx         # Main application component
│   └── ...
├── backend/            # Flask backend
│   ├── app.py         # Main Flask application
│   ├── requirements.txt
│   └── start_backend.py
└── README.md
```

### Technologies Used

**Frontend:**
- React 18 with TypeScript
- Tailwind CSS for styling
- Framer Motion for animations
- Axios for API calls
- React Hot Toast for notifications

**Backend:**
- Flask for web framework
- PyTorch for AI models
- Stable Diffusion for image generation
- Ollama integration for chat

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is created by Vedant Roy as part of VSAT AI (Vedant's Smart AI Technology).

## Credits

- **Developer**: Vedant Roy
- **AI Models**: Stable Diffusion, Ollama
- **UI Framework**: React + Tailwind CSS
- **Icons**: Lucide React

---

**VSAT AI** - Bringing the future of AI interaction to your fingertips! 🚀