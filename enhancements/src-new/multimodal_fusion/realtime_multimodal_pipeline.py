"""
⚡ Y.M.I.R Real-time Multimodal Processing Pipeline
================================================
Real-time pipeline that orchestrates facial emotion recognition, text sentiment analysis,
multimodal fusion, and music recommendation in a unified system.

Features:
- Asynchronous processing pipeline
- Real-time emotion fusion and music recommendations  
- WebSocket support for live updates
- Session management and state tracking
- Performance monitoring and analytics
- Error handling and resilience
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np

# Import our services
from multimodal_emotion_service import get_multimodal_processor, process_facial_emotion_input, process_text_emotion_input
from multimodal_music_recommendation_service import get_music_recommendation_engine, recommend_music_for_session

@dataclass
class PipelineMessage:
    """Message format for pipeline communication"""
    session_id: str
    message_type: str  # facial_input, text_input, fusion_request, recommendation_request
    timestamp: datetime
    data: Dict[str, Any]
    message_id: str = None
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class PipelineResult:
    """Result from pipeline processing"""
    session_id: str
    result_type: str  # emotion_processed, fusion_complete, recommendations_ready, error
    timestamp: datetime
    data: Dict[str, Any]
    processing_time_ms: float
    message_id: str = None
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class RealtimeMultimodalPipeline:
    """Main pipeline orchestrator for real-time multimodal processing"""
    
    def __init__(self, max_concurrent_sessions: int = 50):
        self.max_concurrent_sessions = max_concurrent_sessions
        
        # Get service instances
        self.emotion_processor = get_multimodal_processor()
        self.music_engine = get_music_recommendation_engine()
        
        # Pipeline queues and state
        self.input_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.active_sessions = {}
        self.session_callbacks = defaultdict(list)
        
        # Processing stats
        self.stats = {
            'total_messages_processed': 0,
            'facial_inputs_processed': 0,
            'text_inputs_processed': 0,
            'fusions_completed': 0,
            'recommendations_generated': 0,
            'errors_occurred': 0,
            'average_processing_time_ms': 0.0,
            'sessions_created': 0,
            'sessions_active': 0
        }
        
        # Performance monitoring
        self.processing_times = deque(maxlen=1000)
        self.error_log = deque(maxlen=100)
        
        # Pipeline control
        self.is_running = False
        self.worker_tasks = []
        
        # Auto-fusion settings
        self.auto_fusion_enabled = True
        self.auto_fusion_interval = 10  # seconds
        self.auto_recommendation_enabled = True
        
        logging.info("⚡ Real-time Multimodal Pipeline initialized")
    
    async def start(self, num_workers: int = 3):
        """Start the pipeline with specified number of worker threads"""
        if self.is_running:
            logging.warning("Pipeline is already running")
            return
        
        self.is_running = True
        logging.info(f"🚀 Starting pipeline with {num_workers} workers")
        
        # Start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        # Start background tasks
        asyncio.create_task(self._auto_fusion_task())
        asyncio.create_task(self._session_cleanup_task())
        asyncio.create_task(self._stats_update_task())
        
        logging.info("✅ Pipeline started successfully")
    
    async def stop(self):
        """Stop the pipeline gracefully"""
        if not self.is_running:
            return
        
        logging.info("🛑 Stopping pipeline...")
        self.is_running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logging.info("✅ Pipeline stopped")
    
    def create_session(self, session_config: Optional[Dict[str, Any]] = None) -> str:
        """Create a new processing session"""
        session_id = str(uuid.uuid4())
        
        config = session_config or {}
        self.active_sessions[session_id] = {
            'created': datetime.now(),
            'last_activity': datetime.now(),
            'config': config,
            'stats': {
                'facial_inputs': 0,
                'text_inputs': 0,
                'fusions': 0,
                'recommendations': 0
            }
        }
        
        # Create session in services
        self.emotion_processor.create_session(session_id)
        
        self.stats['sessions_created'] += 1
        self.stats['sessions_active'] += 1
        
        logging.info(f"📱 Created session: {session_id}")
        return session_id
    
    async def process_facial_emotion(self, session_id: str, emotion_data: Dict[str, Any]) -> str:
        """Process facial emotion input"""
        message = PipelineMessage(
            session_id=session_id,
            message_type='facial_input',
            timestamp=datetime.now(),
            data=emotion_data
        )
        
        await self.input_queue.put(message)
        return message.message_id
    
    async def process_text_emotion(self, session_id: str, text: str, emotion: str = None, confidence: float = None) -> str:
        """Process text emotion input"""
        message = PipelineMessage(
            session_id=session_id,
            message_type='text_input',
            timestamp=datetime.now(),
            data={
                'text': text,
                'emotion': emotion,
                'confidence': confidence
            }
        )
        
        await self.input_queue.put(message)
        return message.message_id
    
    async def request_fusion(self, session_id: str, strategy: str = 'adaptive') -> str:
        """Request emotion fusion for session"""
        message = PipelineMessage(
            session_id=session_id,
            message_type='fusion_request',
            timestamp=datetime.now(),
            data={'strategy': strategy}
        )
        
        await self.input_queue.put(message)
        return message.message_id
    
    async def request_recommendations(self, session_id: str, strategy: str = 'adaptive', num_tracks: int = 10) -> str:
        """Request music recommendations for session"""
        message = PipelineMessage(
            session_id=session_id,
            message_type='recommendation_request',
            timestamp=datetime.now(),
            data={
                'strategy': strategy,
                'num_tracks': num_tracks
            }
        )
        
        await self.input_queue.put(message)
        return message.message_id
    
    def register_session_callback(self, session_id: str, callback: Callable[[PipelineResult], None]):
        """Register callback for session results"""
        self.session_callbacks[session_id].append(callback)
    
    async def _worker(self, worker_name: str):
        """Worker task for processing pipeline messages"""
        logging.info(f"🔧 Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get message from queue with timeout
                try:
                    message = await asyncio.wait_for(self.input_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process message
                start_time = time.time()
                result = await self._process_message(message)
                processing_time = (time.time() - start_time) * 1000
                
                # Update stats
                self._update_processing_stats(message.message_type, processing_time)
                
                # Add processing time to result
                result.processing_time_ms = processing_time
                
                # Send result to callbacks
                await self._send_result(result)
                
                # Mark task done
                self.input_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in worker {worker_name}: {e}")
                self.stats['errors_occurred'] += 1
                self.error_log.append({
                    'timestamp': datetime.now(),
                    'worker': worker_name,
                    'error': str(e)
                })
        
        logging.info(f"🔧 Worker {worker_name} stopped")
    
    async def _process_message(self, message: PipelineMessage) -> PipelineResult:
        """Process individual pipeline message"""
        try:
            session_id = message.session_id
            
            # Update session activity
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['last_activity'] = datetime.now()
            
            if message.message_type == 'facial_input':
                return await self._process_facial_input(message)
            elif message.message_type == 'text_input':
                return await self._process_text_input(message)
            elif message.message_type == 'fusion_request':
                return await self._process_fusion_request(message)
            elif message.message_type == 'recommendation_request':
                return await self._process_recommendation_request(message)
            else:
                return PipelineResult(
                    session_id=session_id,
                    result_type='error',
                    timestamp=datetime.now(),
                    data={'error': f'Unknown message type: {message.message_type}'},
                    processing_time_ms=0.0
                )
                
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            return PipelineResult(
                session_id=message.session_id,
                result_type='error',
                timestamp=datetime.now(),
                data={'error': str(e)},
                processing_time_ms=0.0
            )
    
    async def _process_facial_input(self, message: PipelineMessage) -> PipelineResult:
        """Process facial emotion input"""
        session_id = message.session_id
        emotion_data = message.data
        
        # Process through emotion service
        multimodal_reading = process_facial_emotion_input(session_id, emotion_data)
        
        # Update session stats
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['stats']['facial_inputs'] += 1
        
        return PipelineResult(
            session_id=session_id,
            result_type='facial_emotion_processed',
            timestamp=datetime.now(),
            data={
                'emotion': multimodal_reading.facial_emotion,
                'confidence': multimodal_reading.facial_confidence,
                'quality_score': multimodal_reading.face_quality_score,
                'emotions_raw': multimodal_reading.facial_emotions_raw
            },
            processing_time_ms=0.0
        )
    
    async def _process_text_input(self, message: PipelineMessage) -> PipelineResult:
        """Process text emotion input"""
        session_id = message.session_id
        data = message.data
        
        # Process through emotion service
        multimodal_reading = process_text_emotion_input(
            session_id, 
            data['text'], 
            data.get('emotion'), 
            data.get('confidence', 0.0)
        )
        
        # Update session stats
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['stats']['text_inputs'] += 1
        
        return PipelineResult(
            session_id=session_id,
            result_type='text_emotion_processed',
            timestamp=datetime.now(),
            data={
                'emotion': multimodal_reading.text_emotion,
                'confidence': multimodal_reading.text_confidence,
                'text_content': multimodal_reading.text_content
            },
            processing_time_ms=0.0
        )
    
    async def _process_fusion_request(self, message: PipelineMessage) -> PipelineResult:
        """Process emotion fusion request"""
        session_id = message.session_id
        strategy = message.data.get('strategy', 'adaptive')
        
        # Request fusion from emotion processor
        fused_reading = self.emotion_processor.fuse_recent_emotions(
            session_id, 
            time_window_seconds=30,
            strategy=strategy
        )
        
        if fused_reading:
            # Update session stats
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['stats']['fusions'] += 1
            
            return PipelineResult(
                session_id=session_id,
                result_type='fusion_complete',
                timestamp=datetime.now(),
                data={
                    'fused_emotion': fused_reading.fused_emotion,
                    'fused_confidence': fused_reading.fused_confidence,
                    'fusion_weights': fused_reading.fusion_weights,
                    'strategy': strategy,
                    'sources': {
                        'facial_available': fused_reading.facial_emotion is not None,
                        'text_available': fused_reading.text_emotion is not None
                    }
                },
                processing_time_ms=0.0
            )
        else:
            return PipelineResult(
                session_id=session_id,
                result_type='error',
                timestamp=datetime.now(),
                data={'error': 'No emotions available for fusion'},
                processing_time_ms=0.0
            )
    
    async def _process_recommendation_request(self, message: PipelineMessage) -> PipelineResult:
        """Process music recommendation request"""
        session_id = message.session_id
        strategy = message.data.get('strategy', 'adaptive')
        num_tracks = message.data.get('num_tracks', 10)
        
        # Get recommendations from music engine
        rec_set = self.music_engine.get_recommendations_for_emotion(
            session_id, strategy, num_tracks
        )
        
        if rec_set:
            # Update session stats
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['stats']['recommendations'] += 1
            
            return PipelineResult(
                session_id=session_id,
                result_type='recommendations_ready',
                timestamp=datetime.now(),
                data=rec_set.to_dict(),
                processing_time_ms=0.0
            )
        else:
            return PipelineResult(
                session_id=session_id,
                result_type='error',
                timestamp=datetime.now(),
                data={'error': 'No recommendations available'},
                processing_time_ms=0.0
            )
    
    async def _send_result(self, result: PipelineResult):
        """Send result to registered callbacks"""
        session_id = result.session_id
        
        # Send to session-specific callbacks
        callbacks = self.session_callbacks.get(session_id, [])
        for callback in callbacks:
            try:
                callback(result)
            except Exception as e:
                logging.error(f"Error in session callback: {e}")
        
        # Also put in result queue for polling
        await self.result_queue.put(result)
    
    async def _auto_fusion_task(self):
        """Background task for automatic emotion fusion"""
        while self.is_running:
            try:
                if self.auto_fusion_enabled:
                    # Check each active session for auto-fusion
                    for session_id in list(self.active_sessions.keys()):
                        try:
                            # Request fusion for sessions with recent activity
                            last_activity = self.active_sessions[session_id]['last_activity']
                            if datetime.now() - last_activity < timedelta(seconds=60):
                                await self.request_fusion(session_id, 'adaptive')
                                
                                # Also request recommendations if enabled
                                if self.auto_recommendation_enabled:
                                    await asyncio.sleep(0.1)  # Small delay
                                    await self.request_recommendations(session_id, 'adaptive', 5)
                        except Exception as e:
                            logging.error(f"Error in auto-fusion for session {session_id}: {e}")
                
                await asyncio.sleep(self.auto_fusion_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in auto-fusion task: {e}")
    
    async def _session_cleanup_task(self):
        """Background task for cleaning up inactive sessions"""
        while self.is_running:
            try:
                cutoff_time = datetime.now() - timedelta(minutes=30)
                expired_sessions = []
                
                for session_id, session_data in self.active_sessions.items():
                    if session_data['last_activity'] < cutoff_time:
                        expired_sessions.append(session_id)
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    await self._cleanup_session(session_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in session cleanup task: {e}")
    
    async def _cleanup_session(self, session_id: str):
        """Clean up a specific session"""
        try:
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                self.stats['sessions_active'] -= 1
            
            # Remove callbacks
            if session_id in self.session_callbacks:
                del self.session_callbacks[session_id]
            
            # Clean up in services
            self.emotion_processor.cleanup_old_sessions()
            self.music_engine.clear_session_data(session_id)
            
            logging.info(f"🧹 Cleaned up session: {session_id}")
            
        except Exception as e:
            logging.error(f"Error cleaning up session {session_id}: {e}")
    
    async def _stats_update_task(self):
        """Background task for updating statistics"""
        while self.is_running:
            try:
                # Update processing time average
                if self.processing_times:
                    self.stats['average_processing_time_ms'] = np.mean(list(self.processing_times))
                
                # Update active sessions count
                self.stats['sessions_active'] = len(self.active_sessions)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in stats update task: {e}")
    
    def _update_processing_stats(self, message_type: str, processing_time_ms: float):
        """Update processing statistics"""
        self.stats['total_messages_processed'] += 1
        
        if message_type == 'facial_input':
            self.stats['facial_inputs_processed'] += 1
        elif message_type == 'text_input':
            self.stats['text_inputs_processed'] += 1
        elif message_type == 'fusion_request':
            self.stats['fusions_completed'] += 1
        elif message_type == 'recommendation_request':
            self.stats['recommendations_generated'] += 1
        
        # Update processing times
        self.processing_times.append(processing_time_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics"""
        return {
            **self.stats,
            'queue_size': self.input_queue.qsize(),
            'active_sessions': len(self.active_sessions),
            'worker_count': len(self.worker_tasks),
            'is_running': self.is_running,
            'recent_errors': list(self.error_log)[-10:]  # Last 10 errors
        }
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session"""
        if session_id not in self.active_sessions:
            return None
        
        session_data = self.active_sessions[session_id]
        
        return {
            'session_id': session_id,
            'created': session_data['created'].isoformat(),
            'last_activity': session_data['last_activity'].isoformat(),
            'config': session_data['config'],
            'stats': session_data['stats'],
            'emotion_state': self.emotion_processor.get_current_emotion_state(session_id),
            'callback_count': len(self.session_callbacks.get(session_id, []))
        }

# Singleton instance
realtime_pipeline = RealtimeMultimodalPipeline()

def get_realtime_pipeline() -> RealtimeMultimodalPipeline:
    """Get the global realtime pipeline instance"""
    return realtime_pipeline

# Convenience functions for external use
async def start_pipeline(num_workers: int = 3):
    """Start the real-time pipeline"""
    pipeline = get_realtime_pipeline()
    await pipeline.start(num_workers)

async def stop_pipeline():
    """Stop the real-time pipeline"""
    pipeline = get_realtime_pipeline()
    await pipeline.stop()

def create_processing_session(config: Optional[Dict[str, Any]] = None) -> str:
    """Create a new processing session"""
    pipeline = get_realtime_pipeline()
    return pipeline.create_session(config)

async def process_facial_input(session_id: str, emotion_data: Dict[str, Any]) -> str:
    """Process facial emotion input through pipeline"""
    pipeline = get_realtime_pipeline()
    return await pipeline.process_facial_emotion(session_id, emotion_data)

async def process_text_input(session_id: str, text: str, emotion: str = None, confidence: float = None) -> str:
    """Process text input through pipeline"""
    pipeline = get_realtime_pipeline()
    return await pipeline.process_text_emotion(session_id, text, emotion, confidence)

async def get_music_recommendations(session_id: str, strategy: str = 'adaptive', num_tracks: int = 10) -> str:
    """Get music recommendations through pipeline"""
    pipeline = get_realtime_pipeline()
    return await pipeline.request_recommendations(session_id, strategy, num_tracks)

# Demo/Testing main function
async def main_demo():
    """Demo function to test the multimodal pipeline"""
    print("🚀 Starting Y.M.I.R Multimodal Pipeline Demo")
    print("=" * 50)
    
    try:
        # Start pipeline
        await start_pipeline(num_workers=2)
        
        # Create session
        session_id = create_processing_session({'demo': True})
        print(f"📱 Created demo session: {session_id}")
        
        # Demo facial input
        facial_data = {
            'emotions': {'joy': 0.8, 'neutral': 0.2},
            'confidence': 0.85,
            'quality_score': 0.9
        }
        await process_facial_input(session_id, facial_data)
        print("😊 Processed facial emotion: joy (85% confidence)")
        
        # Demo text input  
        await process_text_input(session_id, "I'm feeling great today!", "joy", 0.75)
        print("💬 Processed text emotion: joy (75% confidence)")
        
        # Get recommendations
        recommendations = await get_music_recommendations(session_id, 'adaptive', 5)
        print("🎵 Generated music recommendations")
        
        # Get pipeline stats
        pipeline = get_realtime_pipeline()
        stats = pipeline.get_stats()
        print(f"📊 Pipeline stats: {stats['total_messages_processed']} messages processed")
        
        # Wait a bit for processing
        await asyncio.sleep(2)
        
        # Stop pipeline
        await stop_pipeline()
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧠 Y.M.I.R Multimodal Pipeline")
    print("Running standalone demo...")
    asyncio.run(main_demo())