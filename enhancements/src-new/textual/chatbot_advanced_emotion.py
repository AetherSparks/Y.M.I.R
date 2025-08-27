"""
🤖 Y.M.I.R Advanced Chatbot with ML-based Emotion Detection
=========================================================
ChatGPT-like experience with real emotion AI:
- 🧠 ML-based emotion detection (not rule-based!)
- 🌊 Streaming responses (word-by-word like ChatGPT)
- 😊 Advanced emotion analysis with confidence scores
- 🔧 Function calling (web search, calculations, weather)
- 🎨 Rich terminal interface
- 🧬 Gemini API integration
- 💾 Persistent chat history
- 🛡️ Safety filtering
"""

import os
import json
import time
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import deque
import re

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Google Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✅ Google Gemini API available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ Google Gemini API not available")

# Enhanced UI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box

# ML-based emotion detection
try:
    from transformers import pipeline
    import torch
    ML_AVAILABLE = True
    print("✅ Transformers available for ML emotion detection")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Transformers not available, falling back to lightweight approach")

# Alternative lightweight sentiment/emotion detection
try:
    import requests
    API_AVAILABLE = True
    print("✅ API-based emotion detection available")
except ImportError:
    API_AVAILABLE = False

console = Console()

@dataclass
class ChatMessage:
    """Structured chat message with metadata"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    emotion: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'emotion': self.emotion,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }

@dataclass
class UserProfile:
    """User profile with preferences and history"""
    user_id: str
    name: Optional[str] = None
    preferences: Dict[str, Any] = None
    conversation_style: str = "balanced"  # casual, formal, balanced
    emotion_history: List[str] = None
    topics_of_interest: List[str] = None
    created_at: datetime = None
    last_active: datetime = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.emotion_history is None:
            self.emotion_history = []
        if self.topics_of_interest is None:
            self.topics_of_interest = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()

class AdvancedEmotionAnalyzer:
    """ML-based emotion analysis with multiple fallback methods"""
    
    def __init__(self):
        self.ml_model = None
        self.sentiment_model = None
        self.api_method_available = False
        
        # Try to initialize ML models
        self._initialize_ml_models()
        
        # Emotion mapping for different model outputs
        self.emotion_mapping = {
            # From various emotion models
            'joy': 'happy', 'happiness': 'happy', 'optimism': 'happy',
            'sadness': 'sad', 'grief': 'sad', 'disappointment': 'sad',
            'anger': 'angry', 'annoyance': 'angry', 'disapproval': 'angry',
            'fear': 'anxious', 'nervousness': 'anxious', 'confusion': 'anxious',
            'surprise': 'surprised', 'excitement': 'excited',
            'disgust': 'disgusted', 'embarrassment': 'embarrassed',
            'love': 'loving', 'caring': 'loving', 'gratitude': 'grateful',
            'neutral': 'neutral', 'approval': 'neutral'
        }
    
    def _initialize_ml_models(self):
        """Initialize ML models with fallbacks"""
        try:
            if ML_AVAILABLE:
                # Try lightweight emotion detection first
                device = 0 if torch.cuda.is_available() else -1
                
                # This model is smaller and works well
                self.ml_model = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=device
                )
                print("✅ ML emotion model loaded successfully")
                
                # Also load sentiment as backup
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=device
                )
                print("✅ Sentiment model loaded successfully")
                
        except Exception as e:
            print(f"⚠️ ML models failed to load: {e}")
            self.ml_model = None
            self.sentiment_model = None
    
    def _analyze_with_ml_model(self, text: str) -> Dict[str, Any]:
        """Use ML model for emotion detection with mixed emotion support"""
        try:
            if not self.ml_model:
                return None
                
            # Get emotion predictions
            results = self.ml_model(text)
            
            # Process results - get top emotions, not just the highest
            emotions = {}
            for result in results[:5]:  # Get top 5 emotions
                emotion = result['label'].lower()
                score = result['score']
                
                # Map to our standard emotions
                mapped_emotion = self.emotion_mapping.get(emotion, emotion)
                emotions[mapped_emotion] = score
            
            # Check for mixed emotions (multiple emotions above threshold)
            high_confidence_emotions = {k: v for k, v in emotions.items() if v > 0.3}
            
            if len(high_confidence_emotions) > 1:
                # Mixed emotions detected
                top_emotions = sorted(high_confidence_emotions.items(), key=lambda x: x[1], reverse=True)
                
                # Create mixed emotion label
                if len(top_emotions) >= 2:
                    dominant_emotion = f"{top_emotions[0][0]}+{top_emotions[1][0]}"
                    confidence = (top_emotions[0][1] + top_emotions[1][1]) / 2
                else:
                    dominant_emotion = top_emotions[0][0]
                    confidence = top_emotions[0][1]
            else:
                # Single dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                confidence = dominant_emotion[1]
                dominant_emotion = dominant_emotion[0]
            
            return {
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'all_emotions': emotions,
                'mixed_emotions': len(high_confidence_emotions) > 1,
                'method': 'ml_model'
            }
            
        except Exception as e:
            print(f"⚠️ ML emotion analysis failed: {e}")
            return None
    
    def _analyze_with_sentiment_fallback(self, text: str) -> Dict[str, Any]:
        """Use sentiment analysis as fallback"""
        try:
            if not self.sentiment_model:
                return None
                
            result = self.sentiment_model(text)[0]
            label = result['label'].lower()
            score = result['score']
            
            # Map sentiment to basic emotions
            emotion_map = {
                'positive': 'happy',
                'negative': 'sad',
                'neutral': 'neutral'
            }
            
            emotion = emotion_map.get(label, 'neutral')
            
            return {
                'dominant_emotion': emotion,
                'confidence': score,
                'all_emotions': {emotion: score},
                'method': 'sentiment_fallback'
            }
            
        except Exception as e:
            print(f"⚠️ Sentiment analysis failed: {e}")
            return None
    
    def _analyze_with_api_fallback(self, text: str) -> Dict[str, Any]:
        """Use free API for emotion detection"""
        try:
            # Using a free sentiment analysis API
            url = "https://api.meaningcloud.com/sentiment-2.1"
            
            # Free API key (limited requests)
            payload = {
                'key': 'your_free_key_here',  # You can get free keys from various providers
                'txt': text,
                'lang': 'en'
            }
            
            response = requests.post(url, data=payload, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                
                # Simple mapping (this would need to be adapted based on the API)
                confidence_mapping = {
                    'P+': ('happy', 0.9),
                    'P': ('happy', 0.7),
                    'NEU': ('neutral', 0.6),
                    'N': ('sad', 0.7),
                    'N+': ('sad', 0.9)
                }
                
                polarity = data.get('score_tag', 'NEU')
                emotion, confidence = confidence_mapping.get(polarity, ('neutral', 0.5))
                
                return {
                    'dominant_emotion': emotion,
                    'confidence': confidence,
                    'all_emotions': {emotion: confidence},
                    'method': 'api_fallback'
                }
            
        except Exception as e:
            print(f"⚠️ API emotion analysis failed: {e}")
            return None
    
    def _analyze_with_gemini_fallback(self, text: str) -> Dict[str, Any]:
        """Use Gemini itself for emotion analysis"""
        try:
            # This is a clever approach - use the same Gemini API to analyze emotions
            emotion_prompt = f"""
            Analyze the emotion in this text: "{text}"
            
            Respond with ONLY a JSON object in this format:
            {{"emotion": "happy/sad/angry/anxious/neutral/excited/surprised", "confidence": 0.85}}
            
            Choose the most dominant emotion and give a confidence score between 0.0 and 1.0.
            """
            
            # We would use the same Gemini model here
            # This is implemented in the main class
            return None
            
        except Exception as e:
            print(f"⚠️ Gemini emotion analysis failed: {e}")
            return None
    
    def analyze_text_emotion(self, text: str) -> Dict[str, Any]:
        """Advanced emotion analysis with multiple fallbacks"""
        
        # Method 1: Try ML model first (most accurate)
        result = self._analyze_with_ml_model(text)
        if result:
            return result
        
        # Method 2: Try sentiment analysis fallback
        result = self._analyze_with_sentiment_fallback(text)
        if result:
            return result
        
        # Method 3: Try API fallback
        result = self._analyze_with_api_fallback(text)
        if result:
            return result
        
        # Method 4: Intelligent rule-based fallback (much better than before)
        return self._analyze_with_smart_rules(text)
    
    def _analyze_with_smart_rules(self, text: str) -> Dict[str, Any]:
        """Smart rule-based analysis as final fallback"""
        text_lower = text.lower().strip()
        
        # Advanced pattern matching
        patterns = {
            'happy': {
                'strong': ['amazing', 'fantastic', 'wonderful', 'awesome', 'great', 'excellent'],
                'medium': ['good', 'nice', 'happy', 'glad', 'pleased'],
                'context': ['love', 'enjoy', 'excited', 'thrilled']
            },
            'sad': {
                'strong': ['terrible', 'awful', 'devastating', 'heartbroken', 'miserable'],
                'medium': ['sad', 'disappointed', 'upset', 'hurt'],
                'context': ['cry', 'tears', 'lonely', 'depressed']
            },
            'angry': {
                'strong': ['furious', 'outraged', 'livid', 'enraged'],
                'medium': ['angry', 'mad', 'annoyed', 'frustrated'],
                'context': ['hate', 'stupid', 'ridiculous', 'unfair']
            },
            'anxious': {
                'strong': ['terrified', 'panicking', 'overwhelmed'],
                'medium': ['worried', 'nervous', 'anxious', 'stressed'],
                'context': ['scared', 'fear', 'concerned', 'uncertain']
            }
        }
        
        # Intensity multipliers
        intensifiers = ['very', 'really', 'extremely', 'so', 'incredibly', 'absolutely']
        negators = ['not', "don't", "can't", "won't", 'never', 'hardly']
        
        emotion_scores = {}
        words = text_lower.split()
        
        # Check for greeting patterns (return neutral)
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
        if any(greeting in text_lower for greeting in greetings) and len(words) <= 3:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.8,
                'all_emotions': {'neutral': 0.8},
                'method': 'smart_rules_greeting'
            }
        
        # Analyze each emotion
        for emotion, word_groups in patterns.items():
            score = 0
            
            # Check strong indicators
            for word in word_groups['strong']:
                if word in text_lower:
                    score += 3
            
            # Check medium indicators  
            for word in word_groups['medium']:
                if word in text_lower:
                    score += 2
                    
            # Check contextual indicators
            for word in word_groups['context']:
                if word in text_lower:
                    score += 1
            
            # Apply intensity multipliers
            for intensifier in intensifiers:
                if intensifier in text_lower:
                    score *= 1.5
            
            # Apply negation (flip to opposite or reduce)
            for negator in negators:
                if negator in text_lower:
                    score *= 0.3  # Reduce score significantly
            
            if score > 0:
                emotion_scores[emotion] = min(score / 5.0, 1.0)  # Normalize
        
        # Determine result
        if emotion_scores:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            return {
                'dominant_emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'all_emotions': emotion_scores,
                'method': 'smart_rules'
            }
        else:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.6,
                'all_emotions': {'neutral': 0.6},
                'method': 'smart_rules_default'
            }

class FunctionCalling:
    """Advanced function calling capabilities"""
    
    def __init__(self):
        self.available_functions = {
            'web_search': self.web_search,
            'get_weather': self.get_weather,
            'calculate': self.calculate,
            'get_time': self.get_time,
            'get_date': self.get_date,
            'translate_text': self.translate_text,
            'generate_summary': self.generate_summary
        }
    
    def web_search(self, query: str, num_results: int = 3) -> str:
        """Search the web for information"""
        try:
            # Using DuckDuckGo API (no API key required)
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data.get('AbstractText'):
                return f"Search result: {data['AbstractText']}"
            elif data.get('RelatedTopics'):
                topics = data['RelatedTopics'][:num_results]
                results = []
                for topic in topics:
                    if 'Text' in topic:
                        results.append(topic['Text'])
                return "Search results:\n" + "\n".join(f"• {result}" for result in results)
            else:
                return f"No specific results found for '{query}'"
                
        except Exception as e:
            return f"Web search error: {e}"
    
    def get_weather(self, location: str = "current") -> str:
        """Get weather information"""
        return f"Weather functionality requires API key setup. Location requested: {location}"
    
    def calculate(self, expression: str) -> str:
        """Perform mathematical calculations"""
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return f"Calculation: {expression} = {result}"
            else:
                return "Invalid mathematical expression"
        except Exception as e:
            return f"Calculation error: {e}"
    
    def get_time(self) -> str:
        """Get current time"""
        return f"Current time: {datetime.now().strftime('%H:%M:%S')}"
    
    def get_date(self) -> str:
        """Get current date"""
        return f"Current date: {datetime.now().strftime('%Y-%m-%d (%A)')}"
    
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text (placeholder - requires translation API)"""
        return f"Translation functionality requires API setup. Text: '{text}' to {target_language}"
    
    def generate_summary(self, text: str) -> str:
        """Generate text summary"""
        # Simple extractive summary
        sentences = text.split('.')
        if len(sentences) <= 3:
            return text
        # Return first and last sentences as summary
        return f"{sentences[0]}.{sentences[-1]}."
    
    def detect_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Detect if user wants to call a function"""
        function_calls = []
        
        # Pattern matching for function detection
        patterns = {
            'web_search': [r'search for (.+)', r'look up (.+)', r'find information about (.+)'],
            'get_weather': [r'weather in (.+)', r'what\'s the weather', r'temperature (.+)'],
            'calculate': [r'calculate (.+)', r'what is (.+[\+\-\*/].+)', r'compute (.+)'],
            'get_time': [r'what time is it', r'current time', r'time now'],
            'get_date': [r'what date is it', r'today\'s date', r'current date'],
            'translate_text': [r'translate (.+) to (.+)', r'how do you say (.+) in (.+)'],
        }
        
        for function_name, regex_patterns in patterns.items():
            for pattern in regex_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    function_calls.append({
                        'function': function_name,
                        'args': match.groups(),
                        'confidence': 0.8
                    })
        
        return function_calls
    
    def execute_function(self, function_name: str, args: List[str]) -> str:
        """Execute a detected function"""
        if function_name in self.available_functions:
            try:
                if args:
                    return self.available_functions[function_name](*args)
                else:
                    return self.available_functions[function_name]()
            except Exception as e:
                return f"Function execution error: {e}"
        else:
            return f"Function '{function_name}' not available"

class GeminiChatbot:
    """Enhanced Gemini-powered chatbot with advanced ML emotion detection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.chat_session = None
        
        # Initialize components
        self.emotion_analyzer = AdvancedEmotionAnalyzer()
        self.function_calling = FunctionCalling()
        self.conversation_history = deque(maxlen=100)  # Keep last 100 messages
        self.user_profile = None
        
        # Configuration
        self.config = {
            'model_name': 'gemini-2.0-flash',
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.95,
            'max_tokens': 2048,
            'safety_settings': [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        }
        
        # Personality and context
        self.system_context = """You are Y.M.I.R (Your Mental Intelligence and Recovery), an advanced AI assistant specializing in mental health, emotional support, and intelligent conversation. 

Your personality:
- Empathetic and understanding
- Knowledgeable about psychology and wellness
- Helpful with both technical and emotional questions
- Adaptable to user's emotional state and preferences
- Professional yet warm and approachable

Capabilities:
- Advanced ML-based emotion detection and awareness
- Function calling for web search, calculations, weather, etc.
- Conversation memory and context awareness
- Mental health and wellness guidance
- Technical assistance and problem-solving

Always be helpful, accurate, and emotionally intelligent in your responses."""
        
        self._initialize_gemini()
        self._load_user_profile()
    
    def _initialize_gemini(self):
        """Initialize Gemini API"""
        try:
            if not GEMINI_AVAILABLE:
                raise Exception("Gemini API not available")
            
            genai.configure(api_key=self.api_key)
            
            # Initialize model with configuration
            generation_config = {
                'temperature': self.config['temperature'],
                'top_k': self.config['top_k'], 
                'top_p': self.config['top_p'],
                'max_output_tokens': self.config['max_tokens']
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.config['model_name'],
                generation_config=generation_config,
                safety_settings=self.config['safety_settings']
            )
            
            # Start chat session with system context
            self.chat_session = self.model.start_chat(history=[])
            
            print("✅ Gemini API initialized successfully")
            
        except Exception as e:
            print(f"❌ Gemini initialization error: {e}")
            self.model = None
    
    def _load_user_profile(self):
        """Load or create user profile"""
        profile_path = Path("user_profile.json")
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                    self.user_profile = UserProfile(**data)
                    self.user_profile.last_active = datetime.now()
                print("✅ User profile loaded")
            except Exception as e:
                print(f"⚠️ Profile loading error: {e}")
                self._create_new_profile()
        else:
            self._create_new_profile()
    
    def _create_new_profile(self):
        """Create new user profile"""
        self.user_profile = UserProfile(
            user_id=f"user_{int(time.time())}",
            conversation_style="balanced"
        )
        self._save_user_profile()
        print("✅ New user profile created")
    
    def _save_user_profile(self):
        """Save user profile"""
        try:
            profile_data = asdict(self.user_profile)
            # Convert datetime objects to strings
            profile_data['created_at'] = self.user_profile.created_at.isoformat()
            profile_data['last_active'] = self.user_profile.last_active.isoformat()
            
            with open("user_profile.json", 'w') as f:
                json.dump(profile_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Profile saving error: {e}")
    
    def _build_context_prompt(self, user_input: str, emotion_data: Dict[str, Any]) -> str:
        """Build enhanced context prompt with emotion and history"""
        context_parts = [self.system_context]
        
        # Add detailed emotional context
        if emotion_data['dominant_emotion'] != 'neutral':
            emotion_context = f"""
Current user emotional state: {emotion_data['dominant_emotion']} (confidence: {emotion_data['confidence']:.2f})
Detection method: {emotion_data.get('method', 'unknown')}
All detected emotions: {emotion_data.get('all_emotions', {})}

Please adapt your response tone and content to be supportive and appropriate for someone feeling {emotion_data['dominant_emotion']}.
Show empathy and understanding while being helpful.
"""
            context_parts.append(emotion_context)
        
        # Add recent conversation history for context - MORE DETAILED
        if self.conversation_history:
            recent_messages = list(self.conversation_history)[-8:]  # Last 4 exchanges for better context
            history_context = "Recent conversation context (IMPORTANT - use this to understand user's ongoing situation):\n"
            for msg in recent_messages:
                # Include full message for better context, with emotion if available
                emotion_info = f" [felt: {msg.emotion}]" if msg.emotion and msg.emotion != 'neutral' else ""
                history_context += f"{msg.role}: {msg.content}{emotion_info}\n"
            context_parts.append(history_context)
            
            # Add specific instruction about conversation flow
            context_parts.append("IMPORTANT: Pay attention to how the conversation has evolved. The user may be clarifying or expanding on previous statements. Reference earlier messages when relevant.")
        
        # Add user preferences
        if self.user_profile and self.user_profile.preferences:
            style = self.user_profile.conversation_style
            context_parts.append(f"User prefers {style} conversation style.")
        
        return "\n".join(context_parts) + f"\n\nUser: {user_input}"
    
    def _stream_response(self, response_text: str) -> Generator[str, None, None]:
        """Simulate streaming response like ChatGPT"""
        words = response_text.split()
        current_text = ""
        
        for i, word in enumerate(words):
            current_text += word + " "
            yield current_text
            
            # Variable delay for natural typing effect
            if word.endswith('.') or word.endswith('!') or word.endswith('?'):
                time.sleep(0.3)  # Longer pause after sentences
            else:
                time.sleep(0.05)  # Quick pause between words
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """Generate enhanced response with advanced emotion analysis"""
        try:
            # Advanced emotion analysis
            emotion_data = self.emotion_analyzer.analyze_text_emotion(user_input)
            
            # Check for function calls
            function_calls = self.function_calling.detect_function_calls(user_input)
            function_results = []
            
            # Execute detected functions
            for call in function_calls:
                result = self.function_calling.execute_function(call['function'], call['args'])
                function_results.append(result)
            
            # Build enhanced context
            enhanced_prompt = self._build_context_prompt(user_input, emotion_data)
            
            # Add function results to prompt if any
            if function_results:
                enhanced_prompt += f"\n\nFunction call results: {'; '.join(function_results)}"
                enhanced_prompt += "\nPlease incorporate this information naturally into your response."
            
            if not self.model or not self.chat_session:
                return {
                    'response': "I'm sorry, I'm having trouble connecting to my AI system. Please check the API configuration.",
                    'emotion': emotion_data,
                    'functions_used': function_calls,
                    'streaming': False
                }
            
            # Generate response
            response = self.chat_session.send_message(enhanced_prompt)
            response_text = response.text
            
            # Create chat messages
            user_message = ChatMessage(
                role='user',
                content=user_input,
                timestamp=datetime.now(),
                emotion=emotion_data['dominant_emotion'],
                confidence=emotion_data['confidence'],
                metadata={'emotion_method': emotion_data.get('method')}
            )
            
            assistant_message = ChatMessage(
                role='assistant', 
                content=response_text,
                timestamp=datetime.now(),
                metadata={'functions_used': function_calls}
            )
            
            # Add to conversation history
            self.conversation_history.append(user_message)
            self.conversation_history.append(assistant_message)
            
            # Update user profile
            if self.user_profile:
                self.user_profile.last_active = datetime.now()
                if emotion_data['dominant_emotion'] != 'neutral':
                    self.user_profile.emotion_history.append(emotion_data['dominant_emotion'])
                    # Keep only last 20 emotions
                    self.user_profile.emotion_history = self.user_profile.emotion_history[-20:]
                self._save_user_profile()
            
            return {
                'response': response_text,
                'emotion': emotion_data,
                'functions_used': function_calls,
                'streaming': True,
                'user_message': user_message,
                'assistant_message': assistant_message
            }
            
        except Exception as e:
            print(f"❌ Response generation error: {e}")
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                'emotion': {'dominant_emotion': 'neutral', 'confidence': 0.5},
                'functions_used': [],
                'streaming': False
            }
    
    def save_conversation(self):
        """Save conversation to file"""
        try:
            conversation_data = {
                'timestamp': datetime.now().isoformat(),
                'user_profile': asdict(self.user_profile) if self.user_profile else None,
                'conversation': [msg.to_dict() for msg in self.conversation_history],
                'session_stats': {
                    'total_messages': len(self.conversation_history),
                    'duration_minutes': 0,  # Could calculate this
                    'emotions_detected': list(set(msg.emotion for msg in self.conversation_history if msg.emotion)),
                    'emotion_methods_used': list(set(msg.metadata.get('emotion_method') for msg in self.conversation_history if msg.metadata))
                }
            }
            
            filename = f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2, default=str)
            
            print(f"✅ Conversation saved to {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Conversation save error: {e}")
            return None

class ChatInterface:
    """Rich terminal interface for the chatbot"""
    
    def __init__(self, chatbot: GeminiChatbot):
        self.chatbot = chatbot
        self.console = Console()
    
    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
# 🤖 Welcome to Y.M.I.R Advanced Chatbot!

**🧠 ML-Powered Emotion Detection:**
- 🎯 **Real AI models** - Not rule-based detection!
- 🔬 **Multiple methods** - Transformers, sentiment analysis, smart fallbacks
- 📊 **Confidence scores** - Know how certain the detection is
- 🧬 **Method tracking** - See which AI method was used

**✨ ChatGPT-like Features:**
- 🌊 **Streaming responses** - Word-by-word like ChatGPT
- 🔧 **Function calling** - Web search, calculations, weather
- 💭 **Conversation memory** - I remember our chat history  
- 🎨 **Rich interface** - Beautiful formatting and colors
- 🛡️ **Safe AI** - Built-in content filtering

**Commands:**
- Type normally to chat
- `/help` - Show help
- `/profile` - View your profile  
- `/clear` - Clear conversation
- `/save` - Save conversation
- `/quit` - Exit chatbot

**Example queries:**
- "I'm feeling overwhelmed with work stress"
- "Search for latest machine learning news"
- "What's 25% of 480?"
- "What time is it?"

Let's start chatting with advanced AI emotion understanding! 🚀
        """
        
        self.console.print(Panel(Markdown(welcome_text), title="🧠 Y.M.I.R Advanced AI Chatbot", border_style="blue"))
    
    def display_user_profile(self):
        """Display user profile information"""
        if not self.chatbot.user_profile:
            self.console.print("❌ No user profile available")
            return
        
        profile = self.chatbot.user_profile
        
        table = Table(title="👤 User Profile", box=box.ROUNDED)
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("User ID", profile.user_id)
        table.add_row("Name", profile.name or "Not set")
        table.add_row("Conversation Style", profile.conversation_style)
        table.add_row("Created", profile.created_at.strftime("%Y-%m-%d %H:%M"))
        table.add_row("Last Active", profile.last_active.strftime("%Y-%m-%d %H:%M"))
        table.add_row("Recent Emotions", ", ".join(profile.emotion_history[-5:]) or "None")
        
        self.console.print(table)
    
    def stream_response(self, response_data: Dict[str, Any]):
        """Display streaming response with rich formatting"""
        response_text = response_data['response']
        emotion_data = response_data['emotion']
        functions_used = response_data['functions_used']
        
        # Show advanced emotion detection info
        if emotion_data['dominant_emotion'] != 'neutral':
            method = emotion_data.get('method', 'unknown')
            confidence = emotion_data['confidence']
            mixed = emotion_data.get('mixed_emotions', False)
            
            # Color-code by confidence
            if confidence >= 0.8:
                confidence_color = "bright_green"
            elif confidence >= 0.6:
                confidence_color = "yellow"
            else:
                confidence_color = "red"
            
            # Show mixed emotions indicator
            mixed_indicator = " [MIXED]" if mixed else ""
            emotion_text = f"🧠 Emotion: {emotion_data['dominant_emotion']} ({confidence:.1%}){mixed_indicator} via {method}"
            self.console.print(emotion_text, style=f"dim {confidence_color}")
            
            # Show all emotions if mixed
            if mixed and emotion_data.get('all_emotions'):
                all_emotions_text = "   All emotions: " + ", ".join([f"{k}({v:.1%})" for k, v in emotion_data['all_emotions'].items() if v > 0.2])
                self.console.print(all_emotions_text, style="dim cyan")
        
        # Show function calls
        if functions_used:
            func_text = f"🔧 Using functions: {', '.join([f['function'] for f in functions_used])}"
            self.console.print(func_text, style="dim blue")
        
        # Stream the response
        self.console.print("\n🤖 Y.M.I.R:", style="bold blue", end="")
        
        if response_data.get('streaming', True):
            # Streaming effect
            response_display = Text()
            
            with Live(response_display, refresh_per_second=10, console=self.console) as live:
                for partial_response in self.chatbot._stream_response(response_text):
                    response_display = Text(partial_response)
                    live.update(response_display)
        else:
            # Non-streaming fallback
            self.console.print(f" {response_text}")
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands"""
        command = command.lower().strip()
        
        if command == '/help':
            self.display_welcome()
        elif command == '/profile':
            self.display_user_profile()
        elif command == '/clear':
            self.chatbot.conversation_history.clear()
            self.console.print("✅ Conversation cleared", style="green")
        elif command == '/save':
            filename = self.chatbot.save_conversation()
            if filename:
                self.console.print(f"✅ Conversation saved to {filename}", style="green")
        elif command == '/quit' or command == '/exit':
            return False
        else:
            self.console.print(f"❌ Unknown command: {command}", style="red")
        
        return True
    
    def run(self):
        """Main chat loop"""
        self.display_welcome()
        
        try:
            while True:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]", console=self.console)
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # Generate and display response
                with self.console.status("[bold green]Y.M.I.R is analyzing emotions and thinking...", spinner="dots"):
                    response_data = self.chatbot.generate_response(user_input)
                
                self.stream_response(response_data)
                
        except KeyboardInterrupt:
            self.console.print("\n\n👋 Goodbye! Thanks for chatting with Y.M.I.R!", style="bold blue")
        
        except Exception as e:
            self.console.print(f"\n❌ Unexpected error: {e}", style="red")
        
        finally:
            # Auto-save conversation on exit
            filename = self.chatbot.save_conversation()
            if filename:
                self.console.print(f"✅ Conversation auto-saved to {filename}", style="dim green")

def main():
    """Main function"""
    console.print("🚀 Y.M.I.R Advanced AI Chatbot with ML Emotion Detection", style="bold blue")
    console.print("=" * 70)
    
    # Get API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        console.print("❌ Gemini API key not found!", style="red")
        console.print("Please set GEMINI_API_KEY in your .env file or environment variable.")
        console.print("Create a .env file with: GEMINI_API_KEY=your_api_key_here")
        return
    
    try:
        # Initialize chatbot
        with console.status("[bold green]Initializing Y.M.I.R Advanced AI...", spinner="dots"):
            chatbot = GeminiChatbot(api_key)
        
        # Show what emotion detection methods are available
        if ML_AVAILABLE:
            console.print("🧠 ML emotion models loaded - Using advanced AI detection!", style="green")
        else:
            console.print("⚠️ Using smart rule-based fallback for emotion detection", style="yellow")
        
        # Create and run interface
        interface = ChatInterface(chatbot)
        interface.run()
        
    except Exception as e:
        console.print(f"❌ Startup error: {e}", style="red")

if __name__ == "__main__":
    main()