import React, { useState, useRef, useEffect } from 'react';
import { Send, Image, Bot, User, Sparkles, Cpu, Zap, MessageCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import toast, { Toaster } from 'react-hot-toast';
import axios from 'axios';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: number;
  isImage?: boolean;
}

interface SystemStatus {
  vsat_ai: string;
  ollama: string;
  stable_diffusion: string;
  device: string;
  torch_version: string;
  developer: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      content: "Hello! 👋 I'm VSAT AI, your intelligent assistant created by Vedant Roy. I can chat with you about anything or generate beautiful images from your descriptions. What would you like to do today?",
      timestamp: Date.now()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isImageMode, setIsImageMode] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const API_BASE_URL = 'http://localhost:5000';

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    checkSystemStatus();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkSystemStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/status`);
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Failed to check system status:', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      if (isImageMode) {
        await handleImageGeneration(userMessage.content);
      } else {
        await handleChatMessage(userMessage.content);
      }
    } catch (error) {
      console.error('Error:', error);
      toast.error('Something went wrong. Please try again.');
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: "I apologize, but I encountered an error. Please make sure the backend server is running on http://localhost:5000 and try again.",
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChatMessage = async (prompt: string) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, { prompt });
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: response.data.response,
        timestamp: response.data.timestamp * 1000
      };
      
      setMessages(prev => [...prev, botMessage]);
      toast.success('Response received!');
    } catch (error) {
      throw error;
    }
  };

  const handleImageGeneration = async (prompt: string) => {
    try {
      toast.loading('Generating image...', { duration: 2000 });
      const response = await axios.post(`${API_BASE_URL}/generate-image`, { prompt });
      
      const imageMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: response.data.image,
        timestamp: response.data.timestamp * 1000,
        isImage: true
      };
      
      setMessages(prev => [...prev, imageMessage]);
      toast.success('Image generated successfully!');
    } catch (error) {
      throw error;
    }
  };

  const toggleMode = () => {
    setIsImageMode(!isImageMode);
    setInputValue('');
    inputRef.current?.focus();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Toaster position="top-right" />
      
      {/* Header */}
      <motion.header 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="bg-black/20 backdrop-blur-xl border-b border-white/10 sticky top-0 z-50"
      >
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center"
              >
                <Sparkles className="w-6 h-6 text-white" />
              </motion.div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  VSAT AI
                </h1>
                <p className="text-sm text-gray-400">Vedant's Smart AI Technology</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {systemStatus && (
                <div className="hidden md:flex items-center space-x-4 text-sm">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${systemStatus.ollama.includes('Running') ? 'bg-green-400' : 'bg-red-400'}`}></div>
                    <span className="text-gray-300">Ollama</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${systemStatus.stable_diffusion === 'Loaded' ? 'bg-green-400' : 'bg-yellow-400'}`}></div>
                    <span className="text-gray-300">Stable Diffusion</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Cpu className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-300">{systemStatus.device}</span>
                  </div>
                </div>
              )}
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={toggleMode}
                className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  isImageMode 
                    ? 'bg-gradient-to-r from-pink-500 to-purple-500 text-white shadow-lg shadow-pink-500/25' 
                    : 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg shadow-blue-500/25'
                }`}
              >
                {isImageMode ? (
                  <div className="flex items-center space-x-2">
                    <Image className="w-4 h-4" />
                    <span>Image Mode</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <MessageCircle className="w-4 h-4" />
                    <span>Chat Mode</span>
                  </div>
                )}
              </motion.button>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Chat Container */}
      <div className="max-w-4xl mx-auto px-4 py-6 h-[calc(100vh-140px)] flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-6 scrollbar-thin scrollbar-thumb-purple-500/20 scrollbar-track-transparent">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex items-start space-x-3 max-w-3xl ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                    className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.type === 'user' 
                        ? 'bg-gradient-to-r from-blue-500 to-cyan-500' 
                        : 'bg-gradient-to-r from-purple-500 to-pink-500'
                    }`}
                  >
                    {message.type === 'user' ? (
                      <User className="w-5 h-5 text-white" />
                    ) : (
                      <Bot className="w-5 h-5 text-white" />
                    )}
                  </motion.div>
                  
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    className={`px-4 py-3 rounded-2xl backdrop-blur-xl border ${
                      message.type === 'user'
                        ? 'bg-blue-500/20 border-blue-500/30 text-white'
                        : 'bg-white/10 border-white/20 text-gray-100'
                    }`}
                  >
                    {message.isImage ? (
                      <motion.img
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        src={message.content}
                        alt="Generated image"
                        className="max-w-md rounded-lg shadow-lg"
                      />
                    ) : (
                      <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                    )}
                    <p className="text-xs opacity-60 mt-2">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </p>
                  </motion.div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div className="flex items-start space-x-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div className="px-4 py-3 rounded-2xl bg-white/10 border border-white/20 backdrop-blur-xl">
                  <div className="flex space-x-2">
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, repeat: Infinity, delay: 0 }}
                      className="w-2 h-2 bg-purple-400 rounded-full"
                    />
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                      className="w-2 h-2 bg-pink-400 rounded-full"
                    />
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                      className="w-2 h-2 bg-purple-400 rounded-full"
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        <motion.form
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          onSubmit={handleSubmit}
          className="relative"
        >
          <div className="flex items-center space-x-3 bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-3">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={isImageMode ? "Describe the image you want to generate..." : "Type your message..."}
              className="flex-1 bg-transparent text-white placeholder-gray-400 outline-none px-3 py-2"
              disabled={isLoading}
            />
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className={`p-3 rounded-xl font-medium transition-all duration-200 ${
                inputValue.trim() && !isLoading
                  ? isImageMode
                    ? 'bg-gradient-to-r from-pink-500 to-purple-500 text-white shadow-lg shadow-pink-500/25'
                    : 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg shadow-blue-500/25'
                  : 'bg-gray-600 text-gray-400 cursor-not-allowed'
              }`}
            >
              {isLoading ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                >
                  <Zap className="w-5 h-5" />
                </motion.div>
              ) : isImageMode ? (
                <Image className="w-5 h-5" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </motion.button>
          </div>
          
          <p className="text-center text-xs text-gray-400 mt-2">
            {isImageMode ? 'Image generation mode active' : 'Chat mode active'} • 
            <span className="text-purple-400"> Developed by Vedant Roy</span>
          </p>
        </motion.form>
      </div>
    </div>
  );
}

export default App;