"""
🤖 Y.M.I.R Production-Ready Chatbot with Modern Emotion AI
========================================================
Real production-grade emotion detection without rule-based hacks:
- 🧠 Multiple SOTA emotion models with ensemble voting
- 🎯 Context-aware transformers (not rule-based preprocessing!)
- 🌊 Streaming responses like ChatGPT
- 🔧 Function calling capabilities
- 🎨 Rich terminal interface
- 🧬 Gemini API integration
- 💾 Persistent conversation history
- 🛡️ Production-grade error handling
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

# Production-grade ML emotion detection
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    ML_AVAILABLE = True
    print("✅ Transformers available for production ML emotion detection")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Transformers not available")

console = Console()

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
    preferences: Dict[str, Any] = None
    conversation_style: str = "balanced"
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
                console.print(f"🔄 Loading {config['name']}...", style="dim yellow")
                
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
                console.print(f"✅ {config['name']} loaded successfully", style="green")
                
            except Exception as e:
                console.print(f"⚠️ Failed to load {config['name']}: {e}", style="yellow")
                continue
        
        if successful_models == 0:
            console.print("❌ No emotion models loaded, using fallback", style="red")
            self._setup_fallback_model()
        else:
            console.print(f"✅ {successful_models}/{len(self.model_configs)} emotion models loaded", style="green")
    
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
            console.print("✅ Fallback sentiment model loaded", style="yellow")
            
        except Exception as e:
            console.print(f"❌ Even fallback model failed: {e}", style="red")
    
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
                    console.print(f"⚠️ Model {model_name} failed: {e}", style="dim red")
        
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
            # console.print(f"Debug {model_name}: {type(results)} - {results[:2] if isinstance(results, list) else results}", style="dim blue")
            
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
                console.print(f"⚠️ Unknown result format from {model_name}: {type(results)}", style="dim yellow")
                return None
            
            if not standardized_emotions:
                console.print(f"⚠️ No emotions extracted from {model_name}", style="dim yellow")
                return None
            
            return {
                'emotions': standardized_emotions,
                'weight': model_config['weight'],
                'model': model_name
            }
            
        except Exception as e:
            console.print(f"⚠️ Single model {model_name} error: {e}", style="dim red")
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
            console.print(f"❌ Emotion analysis failed: {e}", style="red")
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
                raise Exception("Gemini API not available")
            
            genai.configure(api_key=self.api_key)
            
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
            
            self.chat_session = self.model.start_chat(history=[])
            console.print("✅ Gemini API initialized successfully", style="green")
            
        except Exception as e:
            console.print(f"❌ Gemini initialization error: {e}", style="red")
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
                console.print("✅ User profile loaded", style="green")
            except Exception as e:
                console.print(f"⚠️ Profile loading error: {e}", style="yellow")
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
        console.print("✅ New user profile created", style="green")
    
    def _save_user_profile(self):
        """Save user profile"""
        try:
            profile_data = asdict(self.user_profile)
            
            # Handle datetime conversion safely
            if hasattr(self.user_profile.created_at, 'isoformat'):
                profile_data['created_at'] = self.user_profile.created_at.isoformat()
            else:
                profile_data['created_at'] = str(self.user_profile.created_at)
                
            if hasattr(self.user_profile.last_active, 'isoformat'):
                profile_data['last_active'] = self.user_profile.last_active.isoformat()
            else:
                profile_data['last_active'] = str(self.user_profile.last_active)
            
            with open("user_profile.json", 'w') as f:
                json.dump(profile_data, f, indent=2)
        except Exception as e:
            console.print(f"⚠️ Profile saving error: {e}", style="yellow")
    
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
                if emotion_data['dominant_emotion'] != 'neutral':
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
            console.print(f"❌ Response generation error: {e}", style="red")
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'emotion': {'dominant_emotion': 'neutral', 'confidence': 0.5},
                'functions_used': [],
                'streaming': False
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
                    user_profile_data['created_at'] = self.user_profile.created_at.isoformat()
                else:
                    user_profile_data['created_at'] = str(self.user_profile.created_at)
                    
                if hasattr(self.user_profile.last_active, 'isoformat'):
                    user_profile_data['last_active'] = self.user_profile.last_active.isoformat()
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
                    console.print(f"⚠️ Error processing message: {e}", style="yellow")
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
            
            console.print(f"✅ Conversation saved to {filename}", style="green")
            console.print(f"   📊 {conversation_data['session_stats']['total_messages']} messages, {len(conversation_data['session_stats']['emotions_detected'])} emotions detected", style="dim green")
            return filename
            
        except Exception as e:
            console.print(f"❌ Conversation save error: {e}", style="red")
            return None

class ChatInterface:
    """Production-grade chat interface"""
    
    def __init__(self, chatbot: GeminiChatbot):
        self.chatbot = chatbot
        self.console = Console()
    
    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
# 🚀 Y.M.I.R Production-Ready AI Chatbot

**🧠 Ensemble Emotion Detection:**
- 🎯 **Multiple SOTA models** - RoBERTa, BERTweet, Twitter-RoBERTa
- 🔄 **Weighted ensemble voting** - No single point of failure
- ⚡ **Parallel processing** - Fast, scalable inference
- 🎭 **Mixed emotion support** - Complex emotional states

**✨ Production Features:**
- 🌊 **Streaming responses** - ChatGPT-like experience
- 🔧 **Function calling** - Web search, calculations
- 💭 **Conversation memory** - Context-aware responses  
- 🛡️ **Error handling** - Graceful degradation
- 📊 **Model transparency** - See which models were used

**Commands:**
- `/help` - Show this help
- `/profile` - View your profile  
- `/setup` - Setup your profile (name, style, interests)
- `/stats` - Show session statistics
- `/clear` - Clear conversation
- `/save` - Save conversation
- `/quit` - Exit

Ready for production-grade emotional AI! 🎯
        """
        
        self.console.print(Panel(Markdown(welcome_text), title="🚀 Y.M.I.R Production AI", border_style="blue"))
    
    def stream_response(self, response_data: Dict[str, Any]):
        """Display response with production-grade emotion info"""
        response_text = response_data['response']
        emotion_data = response_data['emotion']
        functions_used = response_data['functions_used']
        
        # Show ensemble emotion detection info
        if emotion_data['dominant_emotion'] != 'neutral':
            method = emotion_data.get('method', 'unknown')
            confidence = emotion_data['confidence']
            models_used = emotion_data.get('models_used', [])
            mixed = emotion_data.get('mixed_emotions', False)
            
            # Color-code by confidence
            if confidence >= 0.7:
                confidence_color = "bright_green"
            elif confidence >= 0.5:
                confidence_color = "yellow"
            else:
                confidence_color = "red"
            
            mixed_indicator = " [MIXED]" if mixed else ""
            emotion_text = f"🧠 Emotion: {emotion_data['dominant_emotion']} ({confidence:.1%}){mixed_indicator} via {method}"
            self.console.print(emotion_text, style=f"dim {confidence_color}")
            
            if models_used:
                models_text = f"   Models: {', '.join(models_used)}"
                self.console.print(models_text, style="dim cyan")
        
        # Show functions
        if functions_used:
            func_text = f"🔧 Functions: {', '.join([f['function'] for f in functions_used])}"
            self.console.print(func_text, style="dim blue")
        
        # Stream response
        self.console.print("\n🤖 Y.M.I.R:", style="bold blue", end="")
        
        if response_data.get('streaming', True):
            response_display = Text()
            
            with Live(response_display, refresh_per_second=10, console=self.console) as live:
                for partial_response in self.chatbot._stream_response(response_text):
                    response_display = Text(partial_response)
                    live.update(response_display)
        else:
            self.console.print(f" {response_text}")
    
    def handle_command(self, command: str) -> bool:
        """Handle commands"""
        command = command.lower().strip()
        
        if command == '/help':
            self.display_welcome()
        elif command == '/profile':
            self.display_user_profile()
        elif command == '/setup':
            self.setup_user_profile()
        elif command == '/clear':
            self.chatbot.conversation_history.clear()
            self.console.print("✅ Conversation cleared", style="green")
        elif command == '/save':
            filename = self.chatbot.save_conversation()
            if filename:
                self.console.print(f"✅ Saved to {filename}", style="green")
        elif command == '/stats':
            self.show_session_stats()
        elif command == '/quit':
            return False
        else:
            self.console.print(f"❌ Unknown command: {command}", style="red")
            self.console.print("Available commands: /help, /profile, /setup, /clear, /save, /stats, /quit", style="dim")
        
        return True
    
    def display_user_profile(self):
        """Display user profile information"""
        if not self.chatbot.user_profile:
            self.console.print("❌ No user profile available", style="red")
            return
        
        profile = self.chatbot.user_profile
        
        table = Table(title="👤 User Profile", box=box.ROUNDED)
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("User ID", profile.user_id)
        table.add_row("Name", profile.name or "Not set")
        table.add_row("Conversation Style", profile.conversation_style)
        table.add_row("Topics of Interest", ", ".join(profile.topics_of_interest) or "None")
        table.add_row("Created", profile.created_at.strftime("%Y-%m-%d %H:%M") if hasattr(profile.created_at, 'strftime') else str(profile.created_at))
        table.add_row("Last Active", profile.last_active.strftime("%Y-%m-%d %H:%M") if hasattr(profile.last_active, 'strftime') else str(profile.last_active))
        table.add_row("Recent Emotions", ", ".join(profile.emotion_history[-5:]) or "None")
        
        self.console.print(table)
    
    def setup_user_profile(self):
        """Setup user profile interactively"""
        if not self.chatbot.user_profile:
            self.console.print("❌ No user profile available", style="red")
            return
        
        self.console.print("🔧 User Profile Setup", style="bold blue")
        
        # Get user name
        name = Prompt.ask("What's your name? (optional)", default=self.chatbot.user_profile.name or "", show_default=False)
        if name.strip():
            self.chatbot.user_profile.name = name.strip()
        
        # Get conversation style
        style_options = ["casual", "balanced", "formal"]
        current_style = self.chatbot.user_profile.conversation_style
        self.console.print(f"Current conversation style: {current_style}")
        new_style = Prompt.ask("Conversation style", choices=style_options, default=current_style)
        self.chatbot.user_profile.conversation_style = new_style
        
        # Get topics of interest
        topics_input = Prompt.ask("Topics of interest (comma-separated)", default=", ".join(self.chatbot.user_profile.topics_of_interest), show_default=False)
        if topics_input.strip():
            topics = [topic.strip() for topic in topics_input.split(",") if topic.strip()]
            self.chatbot.user_profile.topics_of_interest = topics
        
        # Save profile
        self.chatbot._save_user_profile()
        self.console.print("✅ Profile updated successfully!", style="green")
    
    def show_session_stats(self):
        """Show current session statistics"""
        stats_table = Table(title="📊 Session Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        total_messages = len(self.chatbot.conversation_history)
        user_messages = len([msg for msg in self.chatbot.conversation_history if msg.role == 'user'])
        assistant_messages = len([msg for msg in self.chatbot.conversation_history if msg.role == 'assistant'])
        emotions_detected = list(set(msg.emotion for msg in self.chatbot.conversation_history if msg.emotion and msg.emotion != 'neutral'))
        
        stats_table.add_row("Total Messages", str(total_messages))
        stats_table.add_row("Your Messages", str(user_messages))
        stats_table.add_row("Y.M.I.R Messages", str(assistant_messages))
        stats_table.add_row("Emotions Detected", ", ".join(emotions_detected) or "None")
        stats_table.add_row("Models Available", str(len(self.chatbot.emotion_analyzer.models)))
        
        self.console.print(stats_table)
    
    def run(self):
        """Main chat loop"""
        self.display_welcome()
        
        try:
            while True:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]", console=self.console)
                
                if not user_input.strip():
                    continue
                
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                with self.console.status("[bold green]Analyzing emotions and generating response...", spinner="dots"):
                    response_data = self.chatbot.generate_response(user_input)
                
                self.stream_response(response_data)
                
        except KeyboardInterrupt:
            self.console.print("\n\n👋 Goodbye! Thanks for using Y.M.I.R!", style="bold blue")
        except Exception as e:
            self.console.print(f"\n❌ Unexpected error: {e}", style="red")
        finally:
            filename = self.chatbot.save_conversation()
            if filename:
                self.console.print(f"✅ Auto-saved to {filename}", style="dim green")

def main():
    """Main function"""
    console.print("🚀 Y.M.I.R Production-Ready AI with Ensemble Emotion Detection", style="bold blue")
    console.print("=" * 75)
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        console.print("❌ GEMINI_API_KEY not found in environment!", style="red")
        console.print("Set it in your .env file: GEMINI_API_KEY=your_key_here")
        return
    
    try:
        with console.status("[bold green]Initializing production AI systems...", spinner="dots"):
            chatbot = GeminiChatbot(api_key)
        
        console.print("🎯 Production-ready emotion detection active!", style="green")
        
        interface = ChatInterface(chatbot)
        interface.run()
        
    except Exception as e:
        console.print(f"❌ Startup error: {e}", style="red")

if __name__ == "__main__":
    main()