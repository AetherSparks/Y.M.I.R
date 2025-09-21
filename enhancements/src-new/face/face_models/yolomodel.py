"""
🎯 Enhanced YOLO Model Component for Y.M.I.R
============================================
Advanced object detection with emotion/environment context analysis.
Goes beyond object detection to provide environmental influence on emotions.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import time
import sys
import os

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO available for enhanced emotion context analysis")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available - install: pip install ultralytics")

# Gemini API for intelligent environment classification
try:
    # Load environment variables first
    try:
        from dotenv import load_dotenv
        import sys
        import os
        
        # Get the root project directory and load .env
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
        env_file = os.path.join(project_root, '.env')
        
        if os.path.exists(env_file):
            load_dotenv(env_file)
            print(f"📁 Loaded environment variables from {env_file}")
        else:
            load_dotenv()  # Try default .env loading
            print("📁 Loaded environment variables")
            
    except ImportError:
        print("⚠️ python-dotenv not available, trying without .env loading")
    
    # Add project root to path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Try importing Gemini manager
    from gemini_api_manager import GeminiAPIManager
    GEMINI_AVAILABLE = True
    print("✅ Gemini available for intelligent environment classification")
    
except ImportError as e:
    print(f"⚠️ Gemini import failed: {e}")
    # Try alternative import path
    try:
        current_file = os.path.abspath(__file__)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
        sys.path.insert(0, root_dir)
        from gemini_api_manager import GeminiAPIManager
        GEMINI_AVAILABLE = True
        print("✅ Gemini available for intelligent environment classification (alternative path)")
    except ImportError as e2:
        print(f"⚠️ Gemini not available after trying multiple paths: {e2}")
        GEMINI_AVAILABLE = False
        print("🔄 Using fallback environment classification")
        
except Exception as e:
    print(f"⚠️ Gemini initialization error: {e}")
    GEMINI_AVAILABLE = False

@dataclass
class YOLOConfig:
    """Configuration for enhanced YOLO emotion context analysis"""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    device: str = 'cpu'
    emotion_context_enabled: bool = True
    environment_analysis_enabled: bool = True
    object_tracking_enabled: bool = True

class EmotionContextAnalyzer:
    """Analyzes environmental context to influence emotion detection"""
    
    def __init__(self):
        # Emotion context mapping based on objects and environment
        self.emotion_influence_map = {
            # Positive emotion enhancers
            'positive_objects': {
                'dog', 'cat', 'bird', 'cake', 'pizza', 'wine glass', 'cup',
                'potted plant', 'flower', 'book', 'tv', 'laptop', 'cell phone',
                'sports ball', 'frisbee', 'surfboard', 'bicycle', 'motorcycle'
            },
            
            # Stress/anxiety indicators
            'stress_objects': {
                'clock', 'suitcase', 'backpack', 'tie', 'scissors', 'knife',
                'fire hydrant', 'stop sign', 'parking meter'
            },
            
            # Relaxation/comfort indicators
            'comfort_objects': {
                'couch', 'bed', 'chair', 'dining table', 'vase', 'teddy bear',
                'pillow', 'blanket', 'bowl', 'spoon', 'fork'
            },
            
            # Social context indicators
            'social_objects': {
                'person', 'dining table', 'wine glass', 'cup', 'cake',
                'pizza', 'donut', 'sandwich'
            },
            
            # Work/focus context
            'work_objects': {
                'laptop', 'computer', 'keyboard', 'mouse', 'book', 'pen',
                'desk', 'chair', 'monitor'
            },
            
            # Entertainment/leisure
            'entertainment_objects': {
                'tv', 'remote', 'game controller', 'sports ball', 'frisbee',
                'kite', 'skateboard', 'snowboard', 'surfboard'
            }
        }
        
        # Emotion modifiers based on environment
        self.environment_modifiers = {
            'indoor_comfortable': 1.2,  # Boost positive emotions indoors
            'outdoor_active': 1.3,     # Boost happiness/excitement outdoors
            'social_setting': 1.1,     # Boost positive emotions in social settings
            'work_environment': 0.9,   # Slightly reduce happiness, boost neutral
            'cluttered_space': 0.8,    # Reduce positive emotions in cluttered spaces
            'organized_space': 1.1     # Boost positive emotions in organized spaces
        }
    
    def analyze_emotion_context(self, objects: List[Dict[str, Any]]) -> Dict[str, float]:
        """🚫 DISABLED: Old rule-based system replaced by TRUE ML"""
        # Return neutral modifiers - TRUE ML system handles all context analysis
        context_modifiers = {
            'happiness': 1.0,
            'sadness': 1.0,
            'anger': 1.0,
            'fear': 1.0,
            'surprise': 1.0,
            'disgust': 1.0,
            'neutral': 1.0
        }
        
        # Skip all rule-based logic - TRUE ML system is active
        return context_modifiers
        
        if not objects:
            return context_modifiers
        
        try:
            object_classes = [obj['class'] for obj in objects]
            
            # Analyze environment type
            environment_type = self._determine_environment_type(object_classes)
            
            # Count object categories
            positive_count = sum(1 for obj in object_classes if obj in self.emotion_influence_map['positive_objects'])
            stress_count = sum(1 for obj in object_classes if obj in self.emotion_influence_map['stress_objects'])
            comfort_count = sum(1 for obj in object_classes if obj in self.emotion_influence_map['comfort_objects'])
            social_count = sum(1 for obj in object_classes if obj in self.emotion_influence_map['social_objects'])
            work_count = sum(1 for obj in object_classes if obj in self.emotion_influence_map['work_objects'])
            entertainment_count = sum(1 for obj in object_classes if obj in self.emotion_influence_map['entertainment_objects'])
            
            # Calculate environment influence
            total_objects = len(objects)
            if total_objects > 0:
                positive_ratio = positive_count / total_objects
                stress_ratio = stress_count / total_objects
                comfort_ratio = comfort_count / total_objects
                social_ratio = social_count / total_objects
                work_ratio = work_count / total_objects
                entertainment_ratio = entertainment_count / total_objects
                
                # Apply modifiers based on environment analysis
                
                # Positive environment boosts happiness, reduces sadness
                if positive_ratio > 0.3 or entertainment_ratio > 0.2:
                    context_modifiers['happiness'] *= (1.0 + positive_ratio * 0.5)
                    context_modifiers['sadness'] *= (1.0 - positive_ratio * 0.3)
                    context_modifiers['neutral'] *= (1.0 - positive_ratio * 0.2)
                
                # Comfort environment reduces stress emotions
                if comfort_ratio > 0.3:
                    context_modifiers['anger'] *= (1.0 - comfort_ratio * 0.4)
                    context_modifiers['fear'] *= (1.0 - comfort_ratio * 0.3)
                    context_modifiers['neutral'] *= (1.0 + comfort_ratio * 0.2)
                
                # Stress environment increases negative emotions
                if stress_ratio > 0.2:
                    context_modifiers['anger'] *= (1.0 + stress_ratio * 0.5)
                    context_modifiers['fear'] *= (1.0 + stress_ratio * 0.3)
                    context_modifiers['happiness'] *= (1.0 - stress_ratio * 0.4)
                
                # Social environment boosts happiness
                if social_ratio > 0.3:
                    context_modifiers['happiness'] *= (1.0 + social_ratio * 0.3)
                    context_modifiers['sadness'] *= (1.0 - social_ratio * 0.2)
                
                # Work environment increases neutral, may increase stress
                if work_ratio > 0.4:
                    context_modifiers['neutral'] *= (1.0 + work_ratio * 0.3)
                    context_modifiers['happiness'] *= (1.0 - work_ratio * 0.1)
                    if stress_ratio > 0.1:  # Work + stress objects
                        context_modifiers['anger'] *= (1.0 + work_ratio * 0.2)
                
                # Apply environment type modifiers
                env_modifier = self.environment_modifiers.get(environment_type, 1.0)
                for emotion in context_modifiers:
                    if emotion == 'happiness' and env_modifier > 1.0:
                        context_modifiers[emotion] *= env_modifier
                    elif emotion == 'neutral' and env_modifier < 1.0:
                        context_modifiers[emotion] *= (2.0 - env_modifier)  # Inverse for neutral
            
            # Ensure modifiers stay within reasonable bounds
            for emotion in context_modifiers:
                context_modifiers[emotion] = max(0.1, min(2.0, context_modifiers[emotion]))
                
        except Exception as e:
            print(f"⚠️ Emotion context analysis error: {e}")
        
        return context_modifiers
    
    def _determine_environment_type(self, object_classes: List[str]) -> str:
        """Determine the type of environment based on detected objects"""
        # Indoor furniture indicators
        indoor_objects = {'couch', 'chair', 'dining table', 'bed', 'tv', 'laptop', 'book', 'vase'}
        # Outdoor indicators  
        outdoor_objects = {'bicycle', 'motorcycle', 'car', 'bus', 'truck', 'tree', 'sports ball'}
        # Work indicators
        work_objects = {'laptop', 'keyboard', 'mouse', 'book', 'tie', 'desk', 'chair'}
        # Social indicators
        social_objects = {'person', 'dining table', 'wine glass', 'cake', 'pizza'}
        
        indoor_count = sum(1 for obj in object_classes if obj in indoor_objects)
        outdoor_count = sum(1 for obj in object_classes if obj in outdoor_objects)
        work_count = sum(1 for obj in object_classes if obj in work_objects)
        social_count = sum(1 for obj in object_classes if obj in social_objects)
        
        # Determine primary environment
        if work_count > 2:
            return 'work_environment'
        elif social_count > 2:
            return 'social_setting'
        elif outdoor_count > indoor_count:
            return 'outdoor_active'
        elif indoor_count > 0:
            return 'indoor_comfortable'
        else:
            return 'neutral_environment'
    
    def get_environment_summary(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get comprehensive environment analysis summary"""
        if not objects:
            return {'type': 'unknown', 'confidence': 0.0, 'emotion_influence': 'neutral'}
        
        object_classes = [obj['class'] for obj in objects]
        environment_type = self._determine_environment_type(object_classes)
        context_modifiers = self.analyze_emotion_context(objects)
        
        # Determine dominant emotion influence
        happiness_modifier = context_modifiers.get('happiness', 1.0)
        sadness_modifier = context_modifiers.get('sadness', 1.0)
        
        if happiness_modifier > 1.2:
            emotion_influence = 'positive'
        elif sadness_modifier > 1.2 or context_modifiers.get('anger', 1.0) > 1.2:
            emotion_influence = 'negative'
        else:
            emotion_influence = 'neutral'
        
        return {
            'type': environment_type,
            'confidence': min(1.0, len(objects) / 10),  # More objects = higher confidence
            'emotion_influence': emotion_influence,
            'object_count': len(objects),
            'primary_objects': object_classes[:5],
            'context_modifiers': context_modifiers
        }

class EnhancedYOLOProcessor:
    """Enhanced YOLO processor with emotion context analysis"""
    
    def __init__(self, config: Optional[YOLOConfig] = None):
        self.config = config or YOLOConfig()
        self.yolo_model = None
        self.yolo_model_name = "None"
        self.yolo_model_info = "Not available"
        
        # Initialize emotion context analyzer
        self.emotion_analyzer = EmotionContextAnalyzer()
        
        # Initialize Gemini for intelligent environment analysis
        self.gemini_manager = None
        self.last_gemini_call = 0  # Rate limiting for Gemini API
        self.gemini_cache = {}     # Cache environment classifications
        if GEMINI_AVAILABLE:
            try:
                self.gemini_manager = GeminiAPIManager()
                print("🧠 Gemini environment classifier ready")
            except Exception as e:
                print(f"⚠️ Gemini initialization failed: {e}")
                self.gemini_manager = None
        
        # Object tracking
        self.tracked_objects = {}
        self.object_history = defaultdict(list)
        
        # Performance tracking
        self.detection_count = 0
        self.last_detection_time = 0
        
        self._init_yolo_model()
        
    def _init_yolo_model(self):
        """Initialize YOLO model with latest available version"""
        if not YOLO_AVAILABLE:
            print("❌ YOLO not available - emotion context analysis disabled")
            return
        
        try:
            # Latest YOLO model options (prioritized by performance and accuracy)
            model_options = [
                # YOLOv11 (latest generation - 2024)
                'yolo11x.pt', 'yolo11l.pt', 'yolo11m.pt', 'yolo11s.pt', 'yolo11n.pt',
                
                # YOLOv10 (optimized for speed and accuracy)  
                'yolov10x.pt', 'yolov10l.pt', 'yolov10m.pt', 'yolov10s.pt', 'yolov10n.pt',
                
                # YOLOv9 (advanced architecture)
                'yolov9e.pt', 'yolov9c.pt', 'yolov9m.pt', 'yolov9s.pt',
                
                # Specialized variants
                'yolov8x-worldv2.pt',  # World model - 1000+ classes
                'yolov8x-oiv7.pt',     # Open Images - 600+ classes
                
                # YOLOv8 (proven baseline)
                'yolov8x.pt', 'yolov8l.pt', 'yolov8m.pt', 'yolov8s.pt', 'yolov8n.pt'
            ]
            
            for model_name in model_options:
                try:
                    print(f"🔄 Loading {model_name} for emotion context analysis...")
                    self.yolo_model = YOLO(model_name)
                    
                    # Set model info
                    if 'yolo11' in model_name.lower():
                        self.yolo_model_info = f"{model_name} - Latest 2024 generation"
                    elif 'yolo10' in model_name.lower():
                        self.yolo_model_info = f"{model_name} - Speed optimized"
                    elif 'yolo9' in model_name.lower():
                        self.yolo_model_info = f"{model_name} - Enhanced architecture"
                    elif 'worldv2' in model_name.lower():
                        self.yolo_model_info = f"{model_name} - 1000+ object classes"
                    elif 'oiv7' in model_name.lower():
                        self.yolo_model_info = f"{model_name} - 600+ object classes"
                    else:
                        self.yolo_model_info = f"{model_name} - Standard detection"
                        
                    self.yolo_model_name = model_name
                    print(f"✅ {self.yolo_model_info} - Ready for emotion context analysis")
                    break
                    
                except Exception as e:
                    print(f"⚠️ {model_name} failed: {e}")
                    continue
            
            if not self.yolo_model:
                print("❌ All YOLO models failed to load - emotion context disabled")
        
        except Exception as e:
            print(f"❌ YOLO initialization error: {e}")
    
    def detect_objects_with_emotion_context(self, frame: np.ndarray) -> Tuple[List[Dict], Dict[str, Any]]:
        """Detect objects and analyze emotion context"""
        objects = []
        emotion_context = {}
        
        if not self.yolo_model:
            return objects, emotion_context
        
        try:
            current_time = time.time()
            
            # Run YOLO detection with optimized settings
            results = self.yolo_model(
                frame, 
                verbose=False,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                device=self.config.device
            )
            
            # Priority objects for emotion context (expanded set)
            emotion_relevant_objects = {
                # Human context
                'person', 'face', 'eye', 'nose', 'mouth',
                
                # Technology/work
                'laptop', 'computer', 'keyboard', 'mouse', 'cell phone', 'tablet', 'tv', 'monitor', 'remote',
                
                # Reading/learning
                'book', 'newspaper', 'magazine', 'pen', 'pencil',
                
                # Furniture/comfort
                'chair', 'couch', 'bed', 'desk', 'table', 'pillow', 'blanket',
                
                # Food/social
                'cup', 'bottle', 'wine glass', 'fork', 'knife', 'spoon', 'bowl', 'plate',
                'cake', 'pizza', 'sandwich', 'donut', 'apple', 'banana',
                
                # Transportation
                'car', 'bicycle', 'motorcycle', 'bus', 'train', 'airplane',
                
                # Animals (positive emotion)
                'dog', 'cat', 'bird', 'horse', 'sheep', 'cow',
                
                # Home/decor
                'potted plant', 'vase', 'clock', 'picture frame', 'mirror',
                
                # Personal items
                'backpack', 'handbag', 'suitcase', 'tie', 'hat', 'umbrella',
                
                # Entertainment/sports
                'sports ball', 'frisbee', 'kite', 'surfboard', 'skateboard', 'snowboard',
                'teddy bear', 'game controller',
                
                # Tools/stress indicators
                'scissors', 'knife', 'hammer', 'screwdriver',
                
                # Safety/warning
                'fire hydrant', 'stop sign', 'traffic light'
            }
            
            # Process detections
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        # Calculate object importance for emotion analysis
                        object_area = (x2 - x1) * (y2 - y1)
                        frame_area = frame.shape[0] * frame.shape[1]
                        area_ratio = object_area / frame_area
                        
                        # Enhanced filtering for emotion-relevant objects
                        include_object = False
                        emotion_relevance = 0.0
                        
                        if class_name in emotion_relevant_objects:
                            include_object = confidence > 0.25
                            emotion_relevance = 1.0
                            
                            # Boost relevance for larger objects
                            if area_ratio > 0.1:
                                emotion_relevance *= 1.3
                                
                            # Special relevance for specific objects
                            if class_name in ['person', 'dog', 'cat', 'laptop', 'tv']:
                                emotion_relevance *= 1.5
                                
                        elif confidence > 0.4:
                            include_object = True
                            emotion_relevance = 0.5
                        
                        if include_object:
                            obj_data = {
                                "class": class_name,
                                "confidence": float(confidence),
                                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                "area_ratio": float(area_ratio),
                                "emotion_relevance": emotion_relevance,
                                "priority": class_name in emotion_relevant_objects,
                                "class_id": class_id,
                                "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                "timestamp": current_time
                            }
                            objects.append(obj_data)
            
            # Sort by emotion relevance and confidence
            objects.sort(key=lambda x: (x['emotion_relevance'], x['confidence']), reverse=True)
            
            # Limit to most relevant objects for performance
            objects = objects[:20]
            
            # 🧠 SMART ENVIRONMENT CLASSIFICATION (data-driven, not rule-based)
            if self.config.emotion_context_enabled:
                environment_type = self._classify_environment_smart(objects)
                emotion_context = {
                    'type': environment_type,
                    'ml_enhanced': True,
                    'confidence': self._calculate_environment_confidence(objects)
                }
                emotion_context['detection_info'] = {
                    'model': self.yolo_model_name,
                    'objects_detected': len(objects),
                    'emotion_relevant': len([o for o in objects if o['emotion_relevance'] > 0.5]),
                    'detection_time': current_time - self.last_detection_time if self.last_detection_time > 0 else 0
                }
            
            self.detection_count += 1
            self.last_detection_time = current_time
            
            # Update object tracking if enabled
            if self.config.object_tracking_enabled:
                self._update_object_tracking(objects, current_time)
                
        except Exception as e:
            print(f"⚠️ Enhanced YOLO detection error: {e}")
        
        return objects, emotion_context
    
    def _classify_environment_smart(self, objects: List[Dict]) -> str:
        """🧠 Gemini-powered intelligent environment classification"""
        if not objects:
            return "empty_space"
        
        # Use Gemini for intelligent analysis if available
        if self.gemini_manager:
            return self._classify_environment_with_gemini(objects)
        else:
            # Fallback to simple object-based classification
            return self._classify_environment_fallback(objects)
    
    def _classify_environment_with_gemini(self, objects: List[Dict]) -> str:
        """🧠 Use Gemini AI to intelligently classify the environment (with rate limiting)"""
        try:
            # Prepare object data for Gemini
            object_list = []
            for obj in objects[:10]:  # Limit to top 10 objects for efficiency
                confidence = obj.get('confidence', 0.0)
                class_name = obj.get('class', 'unknown')
                object_list.append(f"{class_name} (confidence: {confidence:.2f})")
            
            objects_text = ", ".join(object_list)
            
            # 🛡️ RATE LIMITING: Only call Gemini every 2 minutes to preserve API quota
            current_time = time.time()
            if current_time - self.last_gemini_call < 120:  # 2 minutes
                # Check cache for similar object combinations
                cache_key = "_".join(sorted([obj.get('class', '') for obj in objects[:5]]))
                if cache_key in self.gemini_cache:
                    cached_result = self.gemini_cache[cache_key]
                    print(f"🔄 Using cached environment classification: {cached_result}")
                    return cached_result
                else:
                    # Return fallback without using API quota
                    fallback = self._classify_environment_fallback(objects)
                    print(f"⏰ Rate limited - using fallback: {fallback}")
                    return fallback
            
            self.last_gemini_call = current_time
            
            prompt = f"""Analyze these detected objects and classify the environment type in 1-3 words:

Objects detected: {objects_text}

Based on these objects, what type of environment/setting is this? Choose the most accurate description from these categories or suggest a better one:

- home_living_room
- home_bedroom  
- home_kitchen
- home_office
- work_office
- outdoor_street
- outdoor_park
- restaurant_cafe
- social_gathering
- tech_workspace
- creative_studio
- fitness_gym
- academic_classroom
- medical_facility
- retail_store
- personal_space
- busy_public_area
- quiet_private_space

Respond with just the environment type (no explanation):"""

            # Get Gemini response using the model
            model = self.gemini_manager.create_model()
            response = model.generate_content(prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            if response_text and len(response_text.strip()) > 0:
                # Clean and validate response
                environment = response_text.strip().lower().replace(' ', '_')
                # Remove any extra text, keep only the classification
                environment = environment.split('\n')[0].split('.')[0].split(',')[0]
                
                print(f"🧠 Gemini classified environment: {environment}")
                
                # Cache the result for similar object combinations
                cache_key = "_".join(sorted([obj.get('class', '') for obj in objects[:5]]))
                self.gemini_cache[cache_key] = environment
                
                return environment
            else:
                print("⚠️ Gemini returned empty response, using fallback")
                return self._classify_environment_fallback(objects)
                
        except Exception as e:
            print(f"⚠️ Gemini environment classification error: {e}")
            return self._classify_environment_fallback(objects)
    
    def _classify_environment_fallback(self, objects: List[Dict]) -> str:
        """Simple fallback classification when Gemini is unavailable"""
        object_names = [obj.get('class', '').lower() for obj in objects]
        
        # Simple pattern matching as fallback
        if 'person' in object_names:
            if any(obj in object_names for obj in ['dining table', 'wine glass', 'cake']):
                return "social_dining"
            elif any(obj in object_names for obj in ['laptop', 'computer', 'book']):
                return "work_meeting"
            else:
                return "personal_space"
        elif any(obj in object_names for obj in ['laptop', 'computer', 'keyboard']):
            return "tech_workspace"
        elif any(obj in object_names for obj in ['car', 'bicycle', 'traffic light']):
            return "outdoor_transport"
        elif any(obj in object_names for obj in ['couch', 'tv', 'bed']):
            return "home_comfort"
        elif len(objects) >= 5:
            return "busy_environment"
        else:
            return "minimal_space"
    
    def _calculate_environment_confidence(self, objects: List[Dict]) -> float:
        """Calculate confidence in environment classification"""
        if not objects:
            return 0.5
        
        # More objects = higher confidence up to a point
        object_count = len(objects)
        count_confidence = min(1.0, object_count / 5.0)
        
        # Higher average confidence in object detection = higher environment confidence
        avg_detection_confidence = sum(obj.get('confidence', 0.5) for obj in objects) / len(objects)
        
        # Combine factors
        final_confidence = (count_confidence * 0.6) + (avg_detection_confidence * 0.4)
        return min(0.95, max(0.3, final_confidence))
    
    def _update_object_tracking(self, objects: List[Dict], timestamp: float):
        """Track objects over time for temporal context analysis"""
        try:
            # Clear old history (keep last 30 seconds)
            for obj_class in list(self.object_history.keys()):
                self.object_history[obj_class] = [
                    entry for entry in self.object_history[obj_class] 
                    if timestamp - entry['timestamp'] < 30.0
                ]
            
            # Add current detections to history
            for obj in objects:
                self.object_history[obj['class']].append({
                    'timestamp': timestamp,
                    'confidence': obj['confidence'],
                    'area_ratio': obj['area_ratio']
                })
                
        except Exception as e:
            print(f"⚠️ Object tracking error: {e}")
    
    def apply_emotion_context(self, base_emotions: Dict[str, float], emotion_context: Dict[str, Any]) -> Dict[str, float]:
        """Apply environmental context modifiers to base emotion predictions"""
        if not emotion_context or not self.config.emotion_context_enabled:
            return base_emotions
        
        try:
            context_modifiers = emotion_context.get('context_modifiers', {})
            enhanced_emotions = {}
            
            for emotion, base_score in base_emotions.items():
                modifier = context_modifiers.get(emotion, 1.0)
                enhanced_score = base_score * modifier
                
                # Ensure scores stay within valid range
                enhanced_emotions[emotion] = max(0.0, min(100.0, enhanced_score))
            
            # Normalize to ensure total doesn't exceed reasonable bounds
            total_score = sum(enhanced_emotions.values())
            if total_score > 100:
                normalization_factor = 100 / total_score
                for emotion in enhanced_emotions:
                    enhanced_emotions[emotion] *= normalization_factor
            
            return enhanced_emotions
            
        except Exception as e:
            print(f"⚠️ Emotion context application error: {e}")
            return base_emotions
    
    def get_temporal_context(self, time_window: float = 10.0) -> Dict[str, Any]:
        """Analyze temporal patterns in object detection for additional context"""
        current_time = time.time()
        temporal_context = {
            'stable_objects': [],
            'transient_objects': [],
            'environment_stability': 0.0
        }
        
        try:
            for obj_class, history in self.object_history.items():
                recent_history = [
                    entry for entry in history 
                    if current_time - entry['timestamp'] < time_window
                ]
                
                if len(recent_history) >= 3:  # Stable if detected multiple times
                    avg_confidence = np.mean([h['confidence'] for h in recent_history])
                    if avg_confidence > 0.5:
                        temporal_context['stable_objects'].append({
                            'class': obj_class,
                            'stability': len(recent_history) / (time_window / 2),  # Normalized stability
                            'avg_confidence': avg_confidence
                        })
                elif len(recent_history) == 1:  # Transient if detected once
                    temporal_context['transient_objects'].append(obj_class)
            
            # Calculate environment stability
            stable_count = len(temporal_context['stable_objects'])
            transient_count = len(temporal_context['transient_objects'])
            total_unique_objects = stable_count + transient_count
            
            if total_unique_objects > 0:
                temporal_context['environment_stability'] = stable_count / total_unique_objects
            
        except Exception as e:
            print(f"⚠️ Temporal context analysis error: {e}")
        
        return temporal_context
    
    def draw_enhanced_objects(self, frame: np.ndarray, objects: List[Dict], emotion_context: Dict[str, Any]):
        """Draw objects with emotion context indicators"""
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            confidence = obj["confidence"]
            class_name = obj["class"]
            emotion_relevance = obj.get("emotion_relevance", 0.0)
            
            # Color coding based on emotion relevance
            if emotion_relevance > 1.0:
                color = (0, 255, 0)  # Bright green for high emotion relevance
                thickness = 3
            elif emotion_relevance > 0.5:
                color = (0, 255, 255)  # Yellow for medium relevance
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for low relevance
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced label with emotion relevance
            label = f"{class_name}: {confidence:.2f}"
            if emotion_relevance > 0.5:
                label += f" ★{emotion_relevance:.1f}"
            
            # Background for text readability
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def draw_emotion_context_info(self, frame: np.ndarray, emotion_context: Dict[str, Any]):
        """Draw emotion context analysis information"""
        if not emotion_context:
            return
        
        y_offset = 30
        color = (255, 255, 255)
        
        # Environment analysis
        env_type = emotion_context.get('type', 'unknown')
        emotion_influence = emotion_context.get('emotion_influence', 'neutral')
        confidence = emotion_context.get('confidence', 0.0)
        
        info_lines = [
            f"🎯 Environment: {env_type.replace('_', ' ').title()}",
            f"Emotion Context: {emotion_influence.upper()} ({confidence:.2f})",
            f"Objects Detected: {emotion_context.get('object_count', 0)}",
        ]
        
        # Add detection info if available
        detection_info = emotion_context.get('detection_info', {})
        if detection_info:
            model = detection_info.get('model', 'Unknown').replace('.pt', '').upper()
            relevant_count = detection_info.get('emotion_relevant', 0)
            info_lines.append(f"Model: {model} | Relevant: {relevant_count}")
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (frame.shape[1] - 400, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded YOLO model"""
        return {
            'model_name': self.yolo_model_name,
            'model_info': self.yolo_model_info,
            'available': self.yolo_model is not None,
            'detection_count': self.detection_count
        }