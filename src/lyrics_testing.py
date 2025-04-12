import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import Counter, deque
import re
import random
import nltk
from nltk.corpus import cmudict
import syllables
import json
import os
global device

# ========= ENVIRONMENT SETUP =========
# Try to download required NLTK data if needed
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict')

# Initialize pronunciation dictionary
pronounce_dict = cmudict.dict()

# Set up device with proper error handling
if torch.cuda.is_available():
    try:
        # Test CUDA capability before fully committing
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Set memory management for better stability
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception as e:
        print(f"⚠️ CUDA initialization failed: {e}")
        print("Falling back to CPU")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    print("✅ Using CPU")

# ======= HYPERPARAMETERS =======
CONFIG = {
    "vocab_size": 10000,
    "max_seq_len": 64,
    "embed_dim": 256,        # Reduced from 512
    "hidden_dim": 512,       # Reduced from 1024
    "num_layers": 2,         # Reduced from 4
    "dropout": 0.2,
    "num_heads": 4,          # Reduced from 8
    "batch_size": 64,        # Reduced from 128
    "learning_rate": 0.001,
    "epochs": 30,
    "patience": 7,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "max_song_length": 300,
    "min_line_length": 3,
    "max_line_length": 12,
    "rhyme_boost": 3.0,
    "gradient_accumulation_steps": 2,  # New: accumulate gradients over multiple batches
    "structure_weights": {
        "verse": 1.0,
        "chorus": 1.2,
        "bridge": 1.1,
        "pre-chorus": 1.0,
        "outro": 0.9
    }
}

# ======= DATA PREPROCESSING =======
class LyricsPreprocessor:
    def __init__(self, config):
        self.config = config
        self.word_to_index = {}
        self.index_to_word = {}
        self.emotion_to_idx = {}
        self.idx_to_emotion = {}
        self.structure_markers = [
            "verse", "chorus", "bridge", "pre-chorus", "hook", 
            "intro", "outro", "refrain", "interlude"
        ]
        self.rhyme_cache = {}
        
        
    def load_data(self, filepath):
        """Load and preprocess the dataset"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        # Handle missing values
        df = df.dropna(subset=["lyrics", "emotion"])
        df["lyrics"] = df["lyrics"].str.lower()
        
        # Enhanced cleaning
        df["lyrics"] = df["lyrics"].apply(self.clean_text)
        return df
    
    def clean_text(self, text):
        """Enhanced text cleaning"""
        if not isinstance(text, str):
            return ""
            
        # Standardize structure markers
        for marker in self.structure_markers:
            pattern = rf'\[{marker}\s*\d*\]|\({marker}\s*\d*\)'
            replacement = f"[{marker}]"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove unwanted characters but keep basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\s.,!?\'\"\[\]]", "", text)
        
        # Standardize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def build_vocabulary(self, df):
        """Build enhanced vocabulary with special tokens"""
        if "lyrics" not in df.columns:
            raise ValueError("DataFrame must contain a 'lyrics' column")
            
        word_counter = Counter()
        for text in df["lyrics"]:
            if isinstance(text, str):
                word_counter.update(text.split())

        # Keep most common words
        most_common_words = [word for word, _ in word_counter.most_common(self.config["vocab_size"] - 6)]
        
        # Special tokens
        self.word_to_index = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<EOS>": 2,
            "<NEWLINE>": 3,
            "<START>": 4,
            "<END>": 5
        }
        
        # Add words to vocabulary
        for idx, word in enumerate(most_common_words):
            self.word_to_index[word] = idx + 6
            
        # Reverse mapping
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
        # Emotion mappings
        if "emotion" in df.columns:
            self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(sorted(df["emotion"].unique()))}
            self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
        else:
            print("⚠️ Warning: No 'emotion' column found in DataFrame")
            self.emotion_to_idx = {"neutral": 0}
            self.idx_to_emotion = {0: "neutral"}
    
    def encode_text(self, text):
        """Enhanced text encoding with structure markers"""
        if not isinstance(text, str):
            return [self.word_to_index["<UNK>"]]
            
        # Replace newlines with special token
        text = text.replace("\n", " <NEWLINE> ")
        
        # Enhanced structure marker handling
        for marker in self.structure_markers:
            pattern = rf'\[{marker}\]'
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, f" <{marker.upper()}> ", text, flags=re.IGNORECASE)
        
        tokens = text.split()
        return [self.word_to_index.get(token, self.word_to_index["<UNK>"]) for token in tokens]
    
    def prepare_training_data(self, df):
        """Prepare training data with enhanced features"""
        print("Preparing training data...")
        
        df["tokenized"] = df["lyrics"].apply(self.encode_text)
        
        input_sequences = []
        target_words = []
        emotion_labels = []
        line_positions = []
        line_lengths = []
        
        sequence_count = 0
        
        for idx, row in df.iterrows():
            tokens = row["tokenized"]
            emotion = self.emotion_to_idx[row["emotion"]]
            
            current_position = 0  # 0: start, 1: middle, 2: end
            current_line_length = 0
            
            for i in range(1, len(tokens)):
                # Update line position tracking
                if tokens[i-1] == self.word_to_index["<NEWLINE>"]:
                    current_position = 0
                    current_line_length = 0
                elif tokens[i] == self.word_to_index["<NEWLINE>"]:
                    current_position = 2
                else:
                    current_position = 1
                    current_line_length += 1
                
                input_seq = tokens[:i][-self.config["max_seq_len"]:]
                target_word = tokens[i]
                
                # Pad the input sequence
                if len(input_seq) < self.config["max_seq_len"]:
                    padding = [self.word_to_index["<PAD>"]] * (self.config["max_seq_len"] - len(input_seq))
                    input_seq = padding + input_seq
                
                input_sequences.append(input_seq)
                target_words.append(target_word)
                emotion_labels.append(emotion)
                line_positions.append(current_position)
                line_lengths.append(min(current_line_length, 19))  # Cap at 19 for embedding
                
                sequence_count += 1
                if sequence_count % 10000 == 0:
                    print(f"Processed {sequence_count} sequences...")
        
        print(f"Finished preparing {len(input_sequences)} training sequences")
        
        return {
            "input_sequences": torch.tensor(input_sequences, dtype=torch.long),
            "target_words": torch.tensor(target_words, dtype=torch.long),
            "emotions": torch.tensor(emotion_labels, dtype=torch.long),
            "positions": torch.tensor(line_positions, dtype=torch.long),
            "lengths": torch.tensor(line_lengths, dtype=torch.long)
        }

# ======= SIMPLIFIED MODEL ARCHITECTURE =======
class SimplifiedLyricsGenerator(nn.Module):
    def __init__(self, config, vocab_size, num_emotions):
        super(SimplifiedLyricsGenerator, self).__init__()
        self.config = config
        
        # Simplified embeddings
        self.word_embedding = nn.Embedding(vocab_size, config["embed_dim"], padding_idx=0)
        self.emotion_embedding = nn.Embedding(num_emotions, config["embed_dim"])
        self.position_embedding = nn.Embedding(3, config["embed_dim"])  # line position
        self.length_embedding = nn.Embedding(20, config["embed_dim"])  # line length (0-19)
        
        # Embedding combiner
        self.embed_combiner = nn.Sequential(
            nn.Linear(config["embed_dim"] * 4, config["embed_dim"]),
            nn.LayerNorm(config["embed_dim"]),
            nn.GELU()
        )
        
        # Use a unidirectional LSTM (less memory intensive)
        self.lstm = nn.LSTM(
            config["embed_dim"],
            config["hidden_dim"],
            num_layers=config["num_layers"],
            batch_first=True,
            dropout=config["dropout"] if config["num_layers"] > 1 else 0,
            bidirectional=False  # Changed to unidirectional
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(config["hidden_dim"]),
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], vocab_size)
        )
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for better training stability"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'embedding' in name:
                    nn.init.normal_(param, mean=0, std=0.1)
                else:
                    # Check dimensions before applying Xavier initialization
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
    
    def forward(self, x, emotion, position=None, length=None):
        """Forward pass with error handling for dimensions"""
        # Get batch size and sequence length
        batch_size, seq_len = x.size()
        
        # Default values if not provided
        if position is None:
            position = torch.ones(batch_size, dtype=torch.long, device=x.device)
        if length is None:
            length = torch.ones(batch_size, dtype=torch.long, device=x.device) * 5
        
        # Ensure proper dimensions and valid ranges
        position = position.view(-1)
        length = length.view(-1)
        
        if position.size(0) != batch_size:
            position = position.repeat(batch_size)[:batch_size]
        if length.size(0) != batch_size:
            length = length.repeat(batch_size)[:batch_size]
            
        # Clip values to valid ranges
        position = torch.clamp(position, 0, 2)  # Valid range: 0-2
        length = torch.clamp(length, 0, 19)    # Valid range: 0-19
        
        # Embeddings
        word_emb = self.word_embedding(x)
        emotion_emb = self.emotion_embedding(emotion).unsqueeze(1).expand(-1, seq_len, -1)
        pos_emb = self.position_embedding(position).unsqueeze(1).expand(-1, seq_len, -1)
        len_emb = self.length_embedding(length).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine embeddings
        combined = torch.cat([word_emb, emotion_emb, pos_emb, len_emb], dim=-1)
        combined = self.embed_combiner(combined)
        
        # LSTM processing with error handling
        try:
            lstm_out, _ = self.lstm(combined)
        except RuntimeError as e:
            print(f"LSTM error: {e}")
            print(f"Input shape: {combined.shape}")
            # Try to recover with a smaller batch
            if batch_size > 1:
                half_batch = batch_size // 2
                first_half = self.lstm(combined[:half_batch])[0]
                second_half = self.lstm(combined[half_batch:])[0]
                lstm_out = torch.cat([first_half, second_half], dim=0)
            else:
                raise  # Cannot recover
        
        # Output projection
        output = self.output_layer(lstm_out)
        return output

# ======= MEMORY-EFFICIENT TRAINING =======
class LyricsTrainer:
    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """Memory-efficient training with gradient accumulation"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Zero gradients once at the beginning
        self.optimizer.zero_grad()
        
        # Gradient accumulation steps
        steps_per_update = self.config.get("gradient_accumulation_steps", 1)
        step_count = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move tensors to device one by one to better manage memory
                batch_X = batch[0].to(device)
                batch_y = batch[1].to(device)
                batch_emotion = batch[2].to(device)
                batch_position = batch[3].to(device)
                batch_length = batch[4].to(device)
                
                # Forward pass
                output = self.model(
                    batch_X, 
                    batch_emotion, 
                    batch_position, 
                    batch_length
                )
                
                # Take the prediction for the target word
                output = output[:, -1, :]  # Shape: [batch_size, vocab_size]
                
                # Calculate loss
                loss = self.criterion(output, batch_y) / steps_per_update
                
                # Backward pass
                loss.backward()
                
                # Update step counter
                step_count += 1
                
                # Only update weights after accumulating gradients
                if step_count % steps_per_update == 0:
                    # Clip gradients to prevent explosion
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Track metrics
                total_loss += loss.item() * steps_per_update
                
                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                
                # Clean up GPU memory
                del batch_X, batch_y, batch_emotion, batch_position, batch_length, output, loss
                
                # Periodically clear cache to prevent fragmentation
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                
                # If CUDA out of memory, try to recover
                if "CUDA out of memory" in str(e) or "CUDNN_STATUS_" in str(e):
                    print("Attempting to recover from CUDA error...")
                    
                    # Clear GPU memory
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad = None
                    
                    torch.cuda.empty_cache()
                    
                    # Skip this batch and continue
                    continue
                else:
                    # Other errors may need to be addressed differently
                    raise
        
        # Handle any remaining gradients
        if step_count % steps_per_update != 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Calculate averages
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model with memory-efficient processing"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    # Process in smaller chunks if needed
                    batch_X = batch[0].to(device)
                    batch_y = batch[1].to(device)
                    batch_emotion = batch[2].to(device)
                    batch_position = batch[3].to(device)
                    batch_length = batch[4].to(device)
                    
                    # Forward pass
                    output = self.model(
                        batch_X, 
                        batch_emotion, 
                        batch_position, 
                        batch_length
                    )
                    
                    # Take the prediction for the target word
                    output = output[:, -1, :]
                    
                    # Calculate loss
                    loss = self.criterion(output, batch_y)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
                    
                    # Clean up memory
                    del batch_X, batch_y, batch_emotion, batch_position, batch_length, output, loss
                    
                except RuntimeError as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Calculate averages
        avg_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else float('inf')
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self):
        """Full training loop with error handling"""
        print(f"Starting training with {self.config['epochs']} epochs")
        
        try:
            for epoch in range(self.config["epochs"]):
                print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
                
                # Set deterministic algorithms for reproducibility
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                try:
                    # Training phase
                    train_loss, train_acc = self.train_epoch()
                    
                    # Validation phase
                    val_loss, val_acc = self.validate()
                    
                    # Update learning rate
                    self.scheduler.step(val_loss)
                    
                    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                    # Early stopping and model saving
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_model()
                        print(f"✅ Model saved (val_loss: {val_loss:.4f})")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config["patience"]:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                            break
                
                except RuntimeError as e:
                    if "CUDA" in str(e) or "CUDNN" in str(e):
                        print(f"⚠️ CUDA error in epoch {epoch+1}: {e}")
                        print("Trying to recover...")
                        
                        # Clear GPU memory
                        torch.cuda.empty_cache()
                        
                        # Reduce batch size if possible
                        if self.config["batch_size"] > 16:
                            self.config["batch_size"] //= 2
                            print(f"Reduced batch size to {self.config['batch_size']}")
                            
                            # Recreate data loaders with new batch size
                            self.train_loader = self._recreate_dataloader(
                                self.train_loader.dataset, 
                                self.config["batch_size"],
                                shuffle=True
                            )
                            
                            self.val_loader = self._recreate_dataloader(
                                self.val_loader.dataset, 
                                self.config["batch_size"],
                                shuffle=False
                            )
                            
                            # Skip this epoch and continue
                            continue
                        else:
                            print("Batch size already minimal. Trying CPU fallback...")
                            # Fall back to CPU
                            if device.type == "cuda":
                                self.model = self.model.cpu()
                                # Update device globally
                                device = torch.device("cpu")
                                
                                # Recreate data loaders
                                self.train_loader = self._recreate_dataloader(
                                    self.train_loader.dataset, 
                                    16,  # Smaller batch for CPU
                                    shuffle=True
                                )
                                
                                self.val_loader = self._recreate_dataloader(
                                    self.val_loader.dataset, 
                                    16,
                                    shuffle=False
                                )
                                continue
                            else:
                                print("Already using CPU. Cannot recover.")
                                raise
                    else:
                        print(f"Non-CUDA error: {e}")
                        raise
                        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_model("interrupted_model.pth")
            print("Model saved to 'interrupted_model.pth'")
    
    def _recreate_dataloader(self, dataset, batch_size, shuffle=False):
        """Helper to recreate dataloaders with new batch size"""
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=(device.type == "cuda")
        )
        
    def save_model(self, filename="lyrics_generator.pth"):
        """Save model and its metadata"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }, filename)

# ======= GENERATION CLASS =======
class LyricsGenerator:
    def __init__(self, model, preprocessor, config):
        self.model = model
        self.model.device = next(model.parameters()).device  # Get device from model
        self.preprocessor = preprocessor
        self.config = config
        self.rhyme_cache = {}
        self.syllable_cache = {}
        self.recent_words = deque(maxlen=10)  # Track recent words
        
    def get_rhyme_score(self, word1, word2):
        """Calculate rhyming score between two words"""
        if word1 == word2:
            return 0.0  # Don't count same word as rhyme
        
        # Check cache first
        cache_key = (word1, word2)
        if cache_key in self.rhyme_cache:
            return self.rhyme_cache[cache_key]
        
        # Clean words
        word1 = re.sub(r'[^\w\s]', '', word1.lower())
        word2 = re.sub(r'[^\w\s]', '', word2.lower())
        
        # Handle empty strings
        if not word1 or not word2:
            return 0.0
        
        # Simple suffix matching (fallback)
        if word1 not in pronounce_dict or word2 not in pronounce_dict:
            score = 0.5 if len(word1) > 2 and len(word2) > 2 and word1[-3:] == word2[-3:] else 0.0
            self.rhyme_cache[cache_key] = score
            return score
        
        # Get pronunciations
        pron1 = pronounce_dict[word1][0]
        pron2 = pronounce_dict[word2][0]
        
        # Extract vowel sounds
        vowel_sounds1 = [sound for sound in pron1 if any(vowel in sound for vowel in '012')]
        vowel_sounds2 = [sound for sound in pron2 if any(vowel in sound for vowel in '012')]
        
        # Calculate rhyme score
        score = 0.0
        if vowel_sounds1 and vowel_sounds2:
            # Perfect rhyme if last vowel and following sounds match
            if vowel_sounds1[-1] == vowel_sounds2[-1]:
                score = 1.0
            # Slant rhyme if vowel sounds are similar
            elif vowel_sounds1[-1][0] == vowel_sounds2[-1][0]:
                score = 0.7
        
        self.rhyme_cache[cache_key] = score
        return score
    
    def generate_line(self, current_words, emotion_idx, line_length, rhyme_word=None, section_type="verse"):
        """Generate a single line of lyrics with the specified properties"""
        # Start with the current context
        if not current_words:
            context = ["<START>"]
        else:
            # Take the last few words as context
            context = current_words[-self.config["max_seq_len"]+1:] if len(current_words) > 0 else ["<START>"]
        
        # Initialize line with existing context if appropriate
        line = []
        syllable_count = 0
        target_syllables = line_length * 2  # Approximate target
        
        # Position tracking
        line_position = 0  # Start of line
        
        # Maximum words to generate to prevent infinite loops
        max_words = self.config["max_line_length"] * 2
        
        # Generate words until we reach desired length or hit special token
        for _ in range(max_words):
            # Convert context to tensor
            tokens = [self.preprocessor.word_to_index.get(w, self.preprocessor.word_to_index["<UNK>"]) for w in context]
            input_tensor = torch.tensor([tokens], device=self.model.device).long()
            
            # Prepare position and length tensors
            position_tensor = torch.tensor([line_position], device=self.model.device).long()
            length_tensor = torch.tensor([len(line)], device=self.model.device).long()
            emotion_tensor = torch.tensor([emotion_idx], device=self.model.device).long()
            
            # Generate logits for next word
            with torch.no_grad():
                logits = self.model(input_tensor, emotion_tensor, position_tensor, length_tensor)
                logits = logits[0, -1, :]  # Get the last position
                
            # Apply temperature
            logits = logits / self.config["temperature"]
            
            # Apply repetition penalty
            for word in self.recent_words:
                word_idx = self.preprocessor.word_to_index.get(word, self.preprocessor.word_to_index["<UNK>"])
                logits[word_idx] /= self.config["repetition_penalty"]
            
            # If we have a rhyme word and are near the end of line
            if rhyme_word and syllable_count >= target_syllables * 0.7:
                # Boost probabilities of rhyming words
                for word, idx in self.preprocessor.word_to_index.items():
                    if word not in ["<PAD>", "<UNK>", "<EOS>", "<NEWLINE>", "<START>", "<END>"]:
                        rhyme_score = self.get_rhyme_score(word, rhyme_word)
                        if rhyme_score > 0.5:
                            logits[idx] *= self.config["rhyme_boost"]
            
            # Apply Top-K filtering
            top_k = min(self.config["top_k"], logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
            # Apply Top-P filtering (nucleus sampling)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config["top_p"]
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            next_word = self.preprocessor.index_to_word[next_token]
            
            # Check if we should end the line
            if next_word == "<NEWLINE>" or next_word == "<EOS>" or next_word == "<END>":
                break
                
            # Add to line if not a special token
            if not next_word.startswith('<') and next_word != '<PAD>' and next_word != '<UNK>':
                line.append(next_word)
                self.recent_words.append(next_word)
                
                # Update syllable count
                syllable_count += self.count_syllables(next_word)
                
                # Update position (0: start, 1: middle, 2: end)
                if len(line) == 1:
                    line_position = 0  # Start
                else:
                    line_position = 1  # Middle
            
            # Check if we've reached target length 
            if syllable_count >= target_syllables:
                break
                
            # Update context with new word
            context.append(next_word)
            if len(context) > self.config["max_seq_len"]:
                context = context[-self.config["max_seq_len"]:]
        
        return line

    def count_syllables(self, word):
        """Count syllables in a word"""
        if not word:
            return 1
            
        # Check cache first
        if word in self.syllable_cache:
            return self.syllable_cache[word]
        
        try:
            count = syllables.estimate(word)
            self.syllable_cache[word] = count
            return count
        except:
            # Fallback: estimate syllables by vowel groups
            word = word.lower()
            count = max(1, len(re.findall(r'[aeiouy]+', word)))
            self.syllable_cache[word] = count
            return count
    
    def generate_song(self, seed="", emotion="happy", song_structure=None):
        """Generate a complete song with specified structure"""
        if not seed:
            seed = "<START>"
            
        # Default structure if none provided
        if not song_structure:
            song_structure = [
                "verse", "verse", 
                "chorus", 
                "verse", 
                "chorus",
                "bridge",
                "chorus", "chorus"
            ]
        
        # Convert emotion to index
        emotion_idx = self.preprocessor.emotion_to_idx.get(emotion.lower(), 0)
        
        # Initialize song components
        song_parts = {}
        current_words = seed.lower().split()
        
        # Generate each section
        for section in song_structure:
            print(f"Generating {section}...")
            
            # Number of lines per section
            if section == "chorus":
                num_lines = 4
            elif section == "bridge":
                num_lines = 3
            elif section == "verse":
                num_lines = 5
            else:
                num_lines = 4
                
            lines = []
            rhyme_pattern = self.get_rhyme_pattern(section, num_lines)
            rhyme_words = {}
            
            # Generate each line in the section
            for i in range(num_lines):
                # Line length varies by section type
                line_length = random.randint(
                    self.config["min_line_length"],
                    self.config["max_line_length"]
                )
                
                # Adjust by structure weighting
                weight = self.config["structure_weights"].get(section, 1.0)
                if weight != 1.0:
                    line_length = int(line_length * weight)
                
                # Get rhyme word if needed
                rhyme_word = None
                if i > 0 and rhyme_pattern[i] in rhyme_pattern[:i]:
                    # Find the previous line with matching rhyme pattern
                    for j in range(i):
                        if rhyme_pattern[j] == rhyme_pattern[i]:
                            # Get last word of that line
                            if lines[j]:  # Check if line exists and is not empty
                                rhyme_word = lines[j][-1]
                                break
                
                # Generate the line
                line = self.generate_line(
                    current_words, 
                    emotion_idx, 
                    line_length, 
                    rhyme_word,
                    section
                )
                
                lines.append(line)
                
                # Update current words with the new line
                if line:
                    current_words.extend(line)
                    
                    # Keep track of rhyme words
                    if rhyme_pattern[i] != 'X':
                        rhyme_words[rhyme_pattern[i]] = line[-1] if line else ""
            
            # Store the section
            if section not in song_parts:
                song_parts[section] = []
            
            song_parts[section].append(lines)
        
        # Format the song
        formatted_song = self.format_song(song_parts, song_structure)
        return formatted_song
    
    def get_rhyme_pattern(self, section_type, num_lines):
        """Generate appropriate rhyme pattern for section"""
        if section_type == "chorus":
            if num_lines == 4:
                return ['A', 'B', 'A', 'B']
            else:
                return ['A', 'B', 'A', 'B', 'C']
        elif section_type == "verse":
            if num_lines == 4:
                return ['A', 'A', 'B', 'B']
            else:
                return ['A', 'A', 'B', 'B', 'C']
        elif section_type == "bridge":
            return ['X', 'X', 'X']
        else:
            # Default pattern
            return ['A', 'B'] * (num_lines // 2) + (['C'] if num_lines % 2 else [])
    
    def format_song(self, song_parts, song_structure):
        """Format the generated song with section labels"""
        formatted_song = ""
        
        for section in song_structure:
            if section in song_parts and song_parts[section]:
                formatted_song += f"[{section.upper()}]\n"
                
                # Get the next section of this type
                lines = song_parts[section][0]
                song_parts[section].pop(0)
                
                # Format each line
                for line in lines:
                    if line:
                        formatted_song += " ".join(line) + "\n"
                    else:
                        formatted_song += "\n"
                
                formatted_song += "\n"
        
        return formatted_song.strip()

# ======= MAIN EXECUTION =======
def main():
    """Main execution function"""
    print("🎵 Initializing Lyrics Generator 🎵")
    
    try:
        # Load dataset
        dataset_path = "lyrics_with_emotions.csv"
        preprocessor = LyricsPreprocessor(CONFIG)
        df = preprocessor.load_data(dataset_path)
        
        # Build vocabulary
        preprocessor.build_vocabulary(df)
        print(f"✅ Vocabulary built with {len(preprocessor.word_to_index)} words")
        
        # Prepare training data
        training_data = preprocessor.prepare_training_data(df)
        
        # Create dataset
        dataset = TensorDataset(
            training_data["input_sequences"],
            training_data["target_words"],
            training_data["emotions"],
            training_data["positions"],
            training_data["lengths"]
        )
        
        # Split into train/validation
        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            pin_memory=(device.type == "cuda")
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            pin_memory=(device.type == "cuda")
        )
        
        # Initialize model
        num_emotions = len(preprocessor.emotion_to_idx)
        model = SimplifiedLyricsGenerator(
            CONFIG, 
            len(preprocessor.word_to_index),
            num_emotions
        ).to(device)
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model initialized with {total_params:,} parameters")
        
        # Train model or load pretrained model
        try:
            # Try to load existing model
            checkpoint = torch.load('lyrics_generator.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded pretrained model")
            
        except (FileNotFoundError, RuntimeError):
            print("⚠️ No pretrained model found or model incompatible. Training new model...")
            trainer = LyricsTrainer(CONFIG, model, train_loader, val_loader)
            trainer.train()
        
        # Initialize generator
        generator = LyricsGenerator(model, preprocessor, CONFIG)
        
        # Ask for user input instead of hardcoding
        print("\n🎵 Let's generate a song!")
        seed = input("Enter seed word(s) (or press Enter for default): ").strip()
        if not seed:
            seed = "die"  # Default seed
            
        print("\nAvailable emotions:")
        for emotion in preprocessor.emotion_to_idx:
            print(f"- {emotion}")
            
        emotion = input("\nEnter emotion from the list above: ").strip().lower()
        if emotion not in preprocessor.emotion_to_idx:
            print(f"Emotion '{emotion}' not recognized. Using 'sadness' as default.")
            emotion = "sadness"  # Default emotion
        
        print("\n🎵 Generating your song...\n")
        example_song = generator.generate_song(
            seed=seed,
            emotion=emotion
        )
        
        print(example_song)
        
        # Save example
        with open("example_song.txt", "w") as f:
            f.write(example_song)
            
        print("\n✅ Song saved to 'example_song.txt'")
            
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()