# 🚀 VSAT AI - Complete Setup Instructions

## 📁 Project Structure
```
your-project-folder/
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── start_backend.py
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── README.md
```

## 🛠️ Step-by-Step Setup

### 1. Create Project Folder
```bash
mkdir vsat-ai-project
cd vsat-ai-project
```

### 2. Copy All Files
Copy all the files from this project into your VS Code workspace:
- Frontend files (src/, package.json, config files)
- Backend files (backend/ folder with all Python files)

### 3. Install Frontend Dependencies
```bash
npm install
```

### 4. Setup Backend
```bash
cd backend
python start_backend.py
```

### 5. Start Frontend (in new terminal)
```bash
npm run dev
```

## 🔧 Manual Backend Setup (if needed)

If the automatic setup doesn't work:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

## 📋 Prerequisites

### Required Software:
- **Node.js** (v16+): Download from [nodejs.org](https://nodejs.org/)
- **Python** (3.8+): Download from [python.org](https://python.org/)
- **Ollama** (optional): Download from [ollama.ai](https://ollama.ai/)

### For Image Generation:
- **PyTorch**: Automatically installed
- **CUDA** (optional): For GPU acceleration

## 🚀 Quick Start Commands

### Terminal 1 (Backend):
```bash
cd backend
python start_backend.py
```

### Terminal 2 (Frontend):
```bash
npm run dev
```

## 🌐 Access Your App
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000

## 🔍 Troubleshooting

### Common Issues:

1. **Port already in use**:
   ```bash
   # Kill processes on ports
   npx kill-port 5173
   npx kill-port 5000
   ```

2. **Python dependencies fail**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Ollama not found**:
   - Install from https://ollama.ai/
   - Run: `ollama serve`
   - Pull a model: `ollama pull llama2`

4. **Image generation fails**:
   - Ensure you have enough RAM (8GB+)
   - Try CPU mode if GPU fails

## 📱 Features

- ✅ **Chat Mode**: AI conversations
- ✅ **Image Generation**: Text-to-image
- ✅ **Real-time Status**: System monitoring
- ✅ **Beautiful UI**: Modern design
- ✅ **Responsive**: Works on all devices

## 🎯 Usage

1. **Chat Mode**: Ask questions, have conversations
2. **Image Mode**: Describe images to generate
3. **Toggle**: Switch between modes anytime
4. **Status**: Monitor AI system health

## 🔧 Development

### File Structure:
- `src/App.tsx`: Main React component
- `backend/app.py`: Flask API server
- `backend/requirements.txt`: Python dependencies
- `package.json`: Node.js dependencies

### Key Technologies:
- **Frontend**: React, TypeScript, Tailwind CSS, Framer Motion
- **Backend**: Flask, PyTorch, Stable Diffusion, Ollama
- **AI**: Ollama (chat), Stable Diffusion (images)

## 🎉 You're Ready!

Your VSAT AI is now ready to use! Enjoy chatting and generating images with your personal AI assistant.

**Developed by Vedant Roy** 👨‍💻