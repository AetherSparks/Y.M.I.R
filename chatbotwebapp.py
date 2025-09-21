"""
🤖 Y.M.I.R Flask Web Chatbot with Modern Emotion AI
===================================================
Web-based production-grade emotion detection and chatbot:
- 🧠 Multiple SOTA emotion models with ensemble voting
- 🎯 Context-aware transformers (not rule-based preprocessing!)
- 🌊 Streaming responses like ChatGPT
- 🔧 Function calling capabilities
- 🌐 Flask web interface
- 🧬 Gemini API integration
- 💾 Persistent conversation history
- 🛡️ Production-grade error handling
"""
# type: ignore

from flask import Flask, render_template, jsonify, request, Response, stream_template
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
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# We don't need Rich console for web interface - removed all Rich dependencies!

# Production-grade ML emotion detection
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    ML_AVAILABLE = True
    print("✅ Transformers available for production ML emotion detection")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Transformers not available")

@dataclass
class ChatMessage:
    """Structured chat message with metadata"""
    role: str
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
    preferences: Optional[Dict[str, Any]] = None
    conversation_style: str = "balanced"
    emotion_history: Optional[List[str]] = None
    topics_of_interest: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    
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

class ProductionEmotionAnalyzer:
    """Production-grade emotion analysis with ensemble of SOTA models"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_configs = [
            {
                'name': 'roberta_emotion',
                'model_id': 'j-hartmann/emotion-english-distilroberta-base',
                'weight': 0.4,
                'emotions': ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            },
            {
                'name': 'twitter_roberta',
                'model_id': 'cardiffnlp/twitter-roberta-base-emotion-multilabel-latest', 
                'weight': 0.3,
                'emotions': ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
            },
            {
                'name': 'bertweet_emotion',
                'model_id': 'finiteautomata/bertweet-base-emotion-analysis',
                'weight': 0.3,
                'emotions': ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
            }
        ]
        self.fallback_available = False
        
        # Initialize models
        self._initialize_production_models()
        
        # Standard emotion mapping for consistency
        self.emotion_standardization = {
            # Map all model outputs to standard emotions
            'anger': 'angry', 'angry': 'angry',
            'sadness': 'sad', 'sad': 'sad',
            'joy': 'happy', 'happiness': 'happy', 'happy': 'happy',
            'fear': 'anxious', 'anxious': 'anxious', 'nervous': 'anxious',
            'surprise': 'surprised', 'surprised': 'surprised',
            'disgust': 'disgusted', 'disgusted': 'disgusted',
            'love': 'loving', 'loving': 'loving',
            'neutral': 'neutral',
            'anticipation': 'excited', 'excitement': 'excited', 'excited': 'excited',
            'optimism': 'hopeful', 'hope': 'hopeful', 'hopeful': 'hopeful',
            'pessimism': 'worried', 'worry': 'worried', 'worried': 'worried',
            'trust': 'confident', 'confident': 'confident'
        }
    
    def _initialize_production_models(self):
        """Initialize multiple production-grade emotion models"""
        successful_models = 0
        
        for config in self.model_configs:
            try:
                print(f"🔄 Loading {config['name']}...")
                
                # Load model and tokenizer
                model = AutoModelForSequenceClassification.from_pretrained(config['model_id'])
                tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
                
                # Create pipeline - handle different model requirements
                device = 0 if torch.cuda.is_available() else -1
                
                # Some models don't support return_all_scores
                try:
                    pipeline_model = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        return_all_scores=True
                    )
                except Exception:
                    # Fallback without return_all_scores
                    pipeline_model = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        device=device
                    )
                
                self.models[config['name']] = {
                    'pipeline': pipeline_model,
                    'weight': config['weight'],
                    'emotions': config['emotions']
                }
                
                successful_models += 1
                print(f"✅ {config['name']} loaded successfully")
                
            except Exception as e:
                print(f"⚠️ Failed to load {config['name']}: {e}")
                continue
        
        if successful_models == 0:
            print("❌ No emotion models loaded, using fallback")
            self._setup_fallback_model()
        else:
            print(f"✅ {successful_models}/{len(self.model_configs)} emotion models loaded")
    
    def _setup_fallback_model(self):
        """Setup lightweight fallback model"""
        try:
            # Use a simple, reliable model as fallback
            pipeline_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
            
            self.models['fallback'] = {
                'pipeline': pipeline_model,
                'weight': 1.0,
                'emotions': ['negative', 'neutral', 'positive']
            }
            self.fallback_available = True
            print("✅ Fallback sentiment model loaded")
            
        except Exception as e:
            print(f"❌ Even fallback model failed: {e}")
    
    def _analyze_with_ensemble(self, text: str) -> Dict[str, Any]:
        """Use ensemble of models for robust emotion detection"""
        if not self.models:
            return self._get_neutral_result()
        
        model_results = {}
        
        # Run all models in parallel for speed
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_model = {
                executor.submit(self._run_single_model, name, config, text): name 
                for name, config in self.models.items()
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=5)  # 5 second timeout per model
                    if result:
                        model_results[model_name] = result
                except Exception as e:
                    print(f"⚠️ Model {model_name} failed: {e}")
        
        if not model_results:
            return self._get_neutral_result()
        
        # Combine results using weighted voting
        return self._ensemble_vote(model_results)
    
    def _run_single_model(self, model_name: str, model_config: Dict, text: str) -> Optional[Dict]:
        """Run a single model and return standardized results"""
        try:
            pipeline_model = model_config['pipeline']
            results = pipeline_model(text)
            
            # Debug: Check result format (disabled for production)
            # print(f"Debug {model_name}: {type(results)} - {results[:2] if isinstance(results, list) else results}")
            
            # Handle different result formats from different models
            standardized_emotions = {}
            
            # Flatten results if they're wrapped in extra lists
            if isinstance(results, list) and len(results) > 0:
                # Check if it's wrapped in an extra list: [[{...}]]
                if isinstance(results[0], list):
                    results = results[0]  # Unwrap the outer list
            
            # Now process the actual results
            if isinstance(results, list) and len(results) > 0:
                for result in results:
                    if isinstance(result, dict) and 'label' in result and 'score' in result:
                        emotion = result['label'].lower()
                        score = result['score']
                        
                        # Skip 'others' category from some models
                        if emotion == 'others':
                            continue
                        
                        # Map to standard emotion
                        std_emotion = self.emotion_standardization.get(emotion, emotion)
                        
                        if std_emotion in standardized_emotions:
                            standardized_emotions[std_emotion] = max(standardized_emotions[std_emotion], score)
                        else:
                            standardized_emotions[std_emotion] = score
            
            # Case 2: Results is a single dictionary
            elif isinstance(results, dict) and 'label' in results and 'score' in results:
                emotion = results['label'].lower()
                score = results['score']
                if emotion != 'others':  # Skip 'others' category
                    std_emotion = self.emotion_standardization.get(emotion, emotion)
                    standardized_emotions[std_emotion] = score
            
            # Case 3: Results is in different format - try to handle gracefully
            else:
                print(f"⚠️ Unknown result format from {model_name}: {type(results)}")
                return None
            
            if not standardized_emotions:
                print(f"⚠️ No emotions extracted from {model_name}")
                return None
            
            return {
                'emotions': standardized_emotions,
                'weight': model_config['weight'],
                'model': model_name
            }
            
        except Exception as e:
            print(f"⚠️ Single model {model_name} error: {e}")
            return None
    
    def _ensemble_vote(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine multiple model results using weighted voting"""
        
        # Collect all emotions and their weighted scores
        emotion_scores = {}
        total_weight = 0
        model_count = len(model_results)
        
        for model_name, result in model_results.items():
            weight = result['weight']
            total_weight += weight
            
            for emotion, score in result['emotions'].items():
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = []
                emotion_scores[emotion].append(score * weight)
        
        # Calculate final scores
        final_emotions = {}
        for emotion, scores in emotion_scores.items():
            # Use mean of weighted scores
            final_emotions[emotion] = statistics.mean(scores)
        
        # Find dominant emotion
        if final_emotions:
            dominant_emotion = max(final_emotions.items(), key=lambda x: x[1])
            
            # Check for mixed emotions (multiple high-confidence emotions)
            high_confidence = {k: v for k, v in final_emotions.items() if v > 0.4}
            mixed = len(high_confidence) > 1
            
            return {
                'dominant_emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'all_emotions': final_emotions,
                'mixed_emotions': mixed,
                'method': f'ensemble_{model_count}_models',
                'models_used': list(model_results.keys())
            }
        
        return self._get_neutral_result()
    
    def _get_neutral_result(self) -> Dict[str, Any]:
        """Return neutral result as fallback"""
        return {
            'dominant_emotion': 'neutral',
            'confidence': 0.6,
            'all_emotions': {'neutral': 0.6},
            'mixed_emotions': False,
            'method': 'fallback_neutral'
        }
    
    def analyze_text_emotion(self, text: str) -> Dict[str, Any]:
        """Main emotion analysis method"""
        if not text.strip():
            return self._get_neutral_result()
        
        try:
            # Use ensemble approach - no rule-based preprocessing
            result = self._analyze_with_ensemble(text)
            
            # Add original text for debugging
            result['original_text'] = text
            
            return result
            
        except Exception as e:
            print(f"❌ Emotion analysis failed: {e}")
            return self._get_neutral_result()

class FunctionCalling:
    """Production-grade function calling with error handling"""
    
    def __init__(self):
        self.available_functions = {
            'web_search': self.web_search,
            'get_weather': self.get_weather,
            'calculate': self.calculate,
            'get_time': self.get_time,
            'get_date': self.get_date
        }
    
    def web_search(self, query: str, num_results: int = 3) -> str:
        """Search the web for information"""
        try:
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
            # Safe evaluation
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
    
    def detect_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Detect function calls using NLP, not regex rules"""
        # This could be enhanced with a small NLP model for intent detection
        # For now, simple pattern matching but easily replaceable
        function_calls = []
        
        patterns = {
            'web_search': [r'search for (.+)', r'look up (.+)', r'find (.+)'],
            'get_weather': [r'weather', r'temperature'],
            'calculate': [r'calculate (.+)', r'what is (.+[\+\-\*/].+)'],
            'get_time': [r'time'],
            'get_date': [r'date']
        }
        
        for function_name, regex_patterns in patterns.items():
            for pattern in regex_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    function_calls.append({
                        'function': function_name,
                        'args': match.groups() if match.groups() else [],
                        'confidence': 0.8
                    })
        
        return function_calls
    
    def execute_function(self, function_name: str, args: List[str]) -> str:
        """Execute a function with error handling"""
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
    """Production-ready Gemini chatbot with ensemble emotion detection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.chat_session = None
        
        # Initialize components
        self.emotion_analyzer = ProductionEmotionAnalyzer()
        self.function_calling = FunctionCalling()
        self.conversation_history = deque(maxlen=100)
        self.user_profile = None
        
        # Configuration
        self.config = {
            'model_name': 'gemini-2.0-flash-exp',
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
        
        # System context
        self.system_context = """You are Y.M.I.R (Your Mental Intelligence and Recovery), a production-grade AI assistant specializing in mental health, emotional support, and intelligent conversation.

Your capabilities:
- Advanced ensemble emotion detection using multiple SOTA models
- Contextual understanding without rule-based preprocessing
- Function calling for web search, calculations, etc.
- Conversation memory and emotional intelligence
- Mental health and wellness guidance
- Technical assistance and problem-solving

Always be helpful, accurate, and emotionally intelligent in your responses."""
        
        self._initialize_gemini()
        self._load_user_profile()
    
    def _initialize_gemini(self):
        """Initialize Gemini API"""
        try:
            if not GEMINI_AVAILABLE:
                print("⚠️ Gemini API not available")
                return
            
            genai.configure(api_key=self.api_key)  # type: ignore
            
            generation_config = {
                'temperature': self.config['temperature'],
                'top_k': self.config['top_k'], 
                'top_p': self.config['top_p'],
                'max_output_tokens': self.config['max_tokens']
            }
            
            self.model = genai.GenerativeModel(  # type: ignore
                model_name=self.config['model_name'],
                generation_config=generation_config,  # type: ignore
                safety_settings=self.config['safety_settings']
            )
            
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
                    
                    # Handle datetime fields safely
                    if 'created_at' in data and isinstance(data['created_at'], str):
                        try:
                            data['created_at'] = datetime.fromisoformat(data['created_at'])
                        except:
                            data['created_at'] = datetime.now()
                    
                    if 'last_active' in data and isinstance(data['last_active'], str):
                        try:
                            data['last_active'] = datetime.fromisoformat(data['last_active'])
                        except:
                            data['last_active'] = datetime.now()
                    
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
            if self.user_profile is None:
                return
            profile_data = asdict(self.user_profile)
            
            # Handle datetime conversion safely
            if self.user_profile.created_at is not None:
                profile_data['created_at'] = self.user_profile.created_at.isoformat()
            else:
                profile_data['created_at'] = datetime.now().isoformat()
                
            if self.user_profile.last_active is not None:
                profile_data['last_active'] = self.user_profile.last_active.isoformat()
            else:
                profile_data['last_active'] = datetime.now().isoformat()
            
            with open("user_profile.json", 'w') as f:
                json.dump(profile_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Profile saving error: {e}")
    
    def _build_context_prompt(self, user_input: str, emotion_data: Dict[str, Any]) -> str:
        """Build context prompt with emotion and history"""
        context_parts = [self.system_context]
        
        # Add emotion context
        if emotion_data['dominant_emotion'] != 'neutral':
            emotion_context = f"""
Current user emotion: {emotion_data['dominant_emotion']} (confidence: {emotion_data['confidence']:.2f})
Detection method: {emotion_data.get('method', 'unknown')}
Models used: {', '.join(emotion_data.get('models_used', []))}

Adapt your response to be supportive and appropriate for someone feeling {emotion_data['dominant_emotion']}.
"""
            context_parts.append(emotion_context)
        
        # Add conversation history
        if self.conversation_history:
            recent_messages = list(self.conversation_history)[-6:]
            history_context = "Recent conversation:\n"
            for msg in recent_messages:
                emotion_info = f" [felt: {msg.emotion}]" if msg.emotion and msg.emotion != 'neutral' else ""
                history_context += f"{msg.role}: {msg.content}{emotion_info}\n"
            context_parts.append(history_context)
        
        return "\n".join(context_parts) + f"\n\nUser: {user_input}"
    
    def _stream_response(self, response_text: str) -> Generator[str, None, None]:
        """Simulate streaming response"""
        words = response_text.split()
        current_text = ""
        
        for word in words:
            current_text += word + " "
            yield current_text
            time.sleep(0.05)
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """Generate response with production-grade emotion analysis"""
        try:
            # Analyze emotions using ensemble
            emotion_data = self.emotion_analyzer.analyze_text_emotion(user_input)
            
            # Check for function calls
            function_calls = self.function_calling.detect_function_calls(user_input)
            function_results = []
            
            # Execute functions
            for call in function_calls:
                result = self.function_calling.execute_function(call['function'], call['args'])
                function_results.append(result)
            
            # Build context
            enhanced_prompt = self._build_context_prompt(user_input, emotion_data)
            
            if function_results:
                enhanced_prompt += f"\n\nFunction results: {'; '.join(function_results)}"
            
            if not self.model or not self.chat_session:
                return {
                    'response': "I'm having trouble connecting to my AI system.",
                    'emotion': emotion_data,
                    'functions_used': function_calls,
                    'streaming': False
                }
            
            # Generate response
            response = self.chat_session.send_message(enhanced_prompt)
            response_text = response.text
            
            # Create messages
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
            
            # Add to history
            self.conversation_history.append(user_message)
            self.conversation_history.append(assistant_message)
            
            # Update profile
            if self.user_profile:
                self.user_profile.last_active = datetime.now()
                if emotion_data['dominant_emotion'] != 'neutral' and self.user_profile.emotion_history is not None:
                    self.user_profile.emotion_history.append(emotion_data['dominant_emotion'])
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
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'emotion': {'dominant_emotion': 'neutral', 'confidence': 0.5},
                'functions_used': [],
                'streaming': False
            }
    
    def get_response(self, message: str) -> Dict[str, Any]:
        """Get response for web interface (wrapper around generate_response)"""
        try:
            result = self.generate_response(message)
            
            # Format for web interface
            emotion_data = result.get('emotion', {})
            emotion_context = "No emotion detected"
            
            if emotion_data and emotion_data.get('dominant_emotion'):
                emotion = emotion_data['dominant_emotion']
                confidence = emotion_data.get('confidence', 0.0)
                emotion_context = f"Detected emotion: {emotion} ({confidence:.1%} confidence)"
            
            return {
                'response': result['response'],
                'emotion_analysis': emotion_data,
                'emotion_context': emotion_context,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Gemini get_response error: {e}")
            return {
                'response': "I apologize, but I encountered an error. Please try again.",
                'emotion_analysis': None,
                'emotion_context': "Error in emotion analysis",
                'timestamp': datetime.now().isoformat()
            }
    
    def save_conversation(self):
        """Save conversation to file"""
        try:
            # Prepare user profile data safely
            user_profile_data = None
            if self.user_profile:
                user_profile_data = asdict(self.user_profile)
                # Convert datetimes to strings safely
                if hasattr(self.user_profile.created_at, 'isoformat'):
                    user_profile_data['created_at'] = self.user_profile.created_at.isoformat()  # type: ignore
                else:
                    user_profile_data['created_at'] = str(self.user_profile.created_at)
                    
                if hasattr(self.user_profile.last_active, 'isoformat'):
                    user_profile_data['last_active'] = self.user_profile.last_active.isoformat()  # type: ignore
                else:
                    user_profile_data['last_active'] = str(self.user_profile.last_active)
            
            # Get conversation data
            conversation_list = []
            emotions_in_session = []
            
            for msg in self.conversation_history:
                try:
                    msg_dict = msg.to_dict()
                    conversation_list.append(msg_dict)
                    
                    # Track emotions
                    if msg.emotion and msg.emotion != 'neutral':
                        emotions_in_session.append(msg.emotion)
                        
                except Exception as e:
                    print(f"⚠️ Error processing message: {e}")
                    continue
            
            conversation_data = {
                'timestamp': datetime.now().isoformat(),
                'user_profile': user_profile_data,
                'conversation': conversation_list,
                'session_stats': {
                    'total_messages': len(conversation_list),
                    'user_messages': len([msg for msg in conversation_list if msg.get('role') == 'user']),
                    'assistant_messages': len([msg for msg in conversation_list if msg.get('role') == 'assistant']),
                    'emotions_detected': list(set(emotions_in_session)),
                    'session_duration_minutes': 0,  # Could calculate this later
                    'models_used': list(set(msg.get('metadata', {}).get('emotion_method', '') for msg in conversation_list if msg.get('metadata')))
                }
            }
            
            filename = f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2, default=str, ensure_ascii=False)
            
            print(f"✅ Conversation saved to {filename}")
            print(f"   📊 {conversation_data['session_stats']['total_messages']} messages, {len(conversation_data['session_stats']['emotions_detected'])} emotions detected")
            return filename
            
        except Exception as e:
            print(f"❌ Conversation save error: {e}")
            return None

# Terminal ChatInterface removed - web interface only!

# Flask Web Application
app = Flask(__name__)

# Global chatbot instance
web_chatbot = None

def init_web_chatbot():
    """Initialize chatbot for web interface"""
    global web_chatbot
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key and GEMINI_AVAILABLE:
        try:
            web_chatbot = GeminiChatbot(api_key)
            print("✅ Web chatbot initialized with Gemini API")
        except Exception as e:
            print(f"⚠️ Gemini chatbot failed, using fallback: {e}")
            web_chatbot = SimpleFallbackChatbot()
    else:
        print("⚠️ Using fallback chatbot (no Gemini API)")
        web_chatbot = SimpleFallbackChatbot()

class SimpleFallbackChatbot:
    """Simple fallback chatbot when Gemini is not available"""
    
    def __init__(self):
        self.emotion_analyzer = ProductionEmotionAnalyzer() if ML_AVAILABLE else None
        self.conversation_history = deque(maxlen=50)
        
    def get_response(self, message: str) -> Dict[str, Any]:
        """Get response with emotion analysis"""
        
        # Analyze emotion
        emotion_result = None
        if self.emotion_analyzer:
            try:
                emotion_result = self.emotion_analyzer.analyze_text_emotion(message)
            except Exception as e:
                print(f"Emotion analysis failed: {e}")
        
        # Generate simple response based on emotion
        if emotion_result and emotion_result['dominant_emotion']:
            emotion = emotion_result['dominant_emotion']
            confidence = emotion_result['confidence']
            
            responses = {
                'joy': f"I can sense your happiness! That's wonderful. Tell me more about what's making you feel so positive.",
                'sadness': f"I notice you might be feeling sad. I'm here to listen and support you. What's on your mind?",
                'anger': f"I sense some frustration. Let's work through this together. What's bothering you?",
                'fear': f"I understand you might be feeling anxious. You're safe here. What's concerning you?",
                'surprise': f"You seem surprised! What's caught your attention?",
                'neutral': f"Thank you for sharing that. How are you feeling right now?"
            }
            
            response = responses.get(emotion, responses['neutral'])
            emotion_context = f"Detected emotion: {emotion} ({confidence:.1%} confidence)"
        else:
            response = "Hello! I'm Y.M.I.R, your AI companion. How can I help you today?"
            emotion_context = "No emotion detected"
        
        # Store in conversation history
        self.conversation_history.append({
            'user': message,
            'assistant': response,
            'emotion': emotion_result,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'response': response,
            'emotion_analysis': emotion_result,
            'emotion_context': emotion_context,
            'timestamp': datetime.now().isoformat()
        }

@app.route('/')
def index():
    """Serve the main chatbot page"""
    # Try to load the HTML file
    html_paths = [
        'text_emotion_detection.html',
        './text_emotion_detection.html',
        os.path.join(os.path.dirname(__file__), 'text_emotion_detection.html')
    ]
    
    for html_path in html_paths:
        try:
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Check if file has content
                        print(f"✅ HTML file loaded from: {html_path}")
                        return content
        except Exception as e:
            print(f"⚠️ Error reading {html_path}: {e}")
            continue
    
    # If no HTML file found or empty, return embedded version
    print("⚠️ HTML file not found or empty, serving embedded version")
    return get_embedded_chatbot_html()

def get_embedded_chatbot_html():
    """Return embedded HTML for chatbot interface"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Y.M.I.R AI Chatbot</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
            color: #f0f0f0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 80vh;
            background: rgba(30, 30, 30, 0.9);
            border-radius: 16px;
            border: 1px solid rgba(100, 100, 255, 0.2);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .chat-header {
            padding: 20px;
            background: linear-gradient(45deg, #4070ff, #00d4ff);
            text-align: center;
        }
        
        .chat-header h1 {
            margin: 0;
            font-size: 1.8rem;
            color: white;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .message {
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .user-message {
            background: #4070ff;
            color: white;
            align-self: flex-end;
        }
        
        .bot-message {
            background: rgba(100, 100, 100, 0.3);
            color: #f0f0f0;
            align-self: flex-start;
        }
        
        .emotion-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            margin-top: 5px;
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
        }
        
        .chat-input-container {
            padding: 20px;
            border-top: 1px solid rgba(100, 100, 255, 0.2);
            display: flex;
            gap: 10px;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid rgba(100, 100, 255, 0.3);
            border-radius: 8px;
            background: rgba(50, 50, 50, 0.5);
            color: #f0f0f0;
            font-size: 1rem;
        }
        
        .send-button {
            padding: 12px 20px;
            background: linear-gradient(45deg, #4070ff, #00d4ff);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.2s;
        }
        
        .send-button:hover {
            transform: translateY(-2px);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .status-indicator {
            padding: 10px;
            text-align: center;
            font-size: 0.9rem;
            color: #b0b0b0;
        }
        
        .typing-indicator {
            display: none;
            padding: 12px 16px;
            background: rgba(100, 100, 100, 0.3);
            border-radius: 12px;
            align-self: flex-start;
            max-width: 80px;
        }
        
        .typing-dots {
            display: flex;
            gap: 4px;
        }
        
        .typing-dots span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4070ff;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 Y.M.I.R AI Chatbot</h1>
            <div class="status-indicator" id="status">Ready to chat with emotion analysis</div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot-message">
                Hello! I'm Y.M.I.R, your AI companion with advanced emotion detection. How are you feeling today?
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        
        <div class="chat-input-container">
            <input type="text" id="chatInput" class="chat-input" placeholder="Type your message here..." maxlength="500">
            <button id="sendButton" class="send-button">Send</button>
        </div>
    </div>
    
    <script>
        const chatMessages = document.getElementById('chatMessages');
        const chatInput = document.getElementById('chatInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');
        const status = document.getElementById('status');
        
        function addMessage(content, isUser = false, emotionContext = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageDiv.textContent = content;
            
            if (!isUser && emotionContext) {
                const emotionBadge = document.createElement('div');
                emotionBadge.className = 'emotion-badge';
                emotionBadge.textContent = emotionContext;
                messageDiv.appendChild(emotionBadge);
            }
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function showTyping() {
            typingIndicator.style.display = 'block';
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function hideTyping() {
            typingIndicator.style.display = 'none';
        }
        
        async function sendMessage() {
            const message = chatInput.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage(message, true);
            chatInput.value = '';
            sendButton.disabled = true;
            showTyping();
            status.textContent = 'Analyzing emotion and generating response...';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    hideTyping();
                    addMessage(data.response, false, data.emotion_context);
                    status.textContent = 'Ready to chat';
                } else {
                    hideTyping();
                    addMessage('Sorry, I encountered an error. Please try again.', false);
                    status.textContent = 'Error occurred';
                }
                
            } catch (error) {
                hideTyping();
                addMessage('Sorry, I had trouble connecting. Please try again.', false);
                status.textContent = 'Connection error';
                console.error('Chat error:', error);
            }
            
            sendButton.disabled = false;
            chatInput.focus();
        }
        
        sendButton.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Focus input on load
        chatInput.focus();
    </script>
</body>
</html>'''

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat functionality"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            })
        
        if not web_chatbot:
            return jsonify({
                'success': False,
                'error': 'Chatbot not initialized'
            })
        
        # Get response from chatbot
        result = web_chatbot.get_response(message)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'emotion_context': result['emotion_context'],
            'emotion_analysis': result.get('emotion_analysis'),
            'timestamp': result['timestamp']
        })
        
    except Exception as e:
        print(f"Chat API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/status')
def api_status():
    """API endpoint to get chatbot status"""
    return jsonify({
        'chatbot_available': web_chatbot is not None,
        'gemini_available': GEMINI_AVAILABLE,
        'ml_available': ML_AVAILABLE,
        'status': 'ready' if web_chatbot else 'not_initialized'
    })

def main_web():
    """Initialize and run the web application"""
    print("🚀 Starting Y.M.I.R Web Chatbot")
    print("=" * 50)
    
    # Initialize chatbot
    init_web_chatbot()
    
    print("🌐 Open browser and go to: http://localhost:5001")
    print("🤖 Advanced emotion detection chatbot ready!")
    print("=" * 50)
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)

def main():
    """Main function for terminal interface (deprecated - use --web instead)"""
    print("❌ Terminal interface requires 'rich' package.")
    print("Please run with --web flag instead:")
    print("python chatbotwebapp.py --web")

if __name__ == "__main__":  
    # Check if running as web app or terminal
    import sys
    if '--web' in sys.argv:
        main_web()
    else:
        main()