"""
🎵 Y.M.I.R Production-Ready Music Recommendation Model
===================================================
Fixed major issues from original training:
- ❌ Removed data leakage (mood as input feature)
- ✅ Proper class imbalance handling
- ✅ Robust cross-validation strategy
- ✅ Regularized ensemble to prevent overfitting
- ✅ Realistic performance targets (65-75% accuracy)
- ✅ Feature engineering for mood-audio relationships
- ✅ Production-ready evaluation framework
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import contextlib
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, 
    cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import os

# Fix Windows joblib CPU detection issues
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set fixed CPU count

# Suppress all ML library warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
# Suppress specific sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='xgboost')
warnings.filterwarnings('ignore', module='lightgbm')
warnings.filterwarnings('ignore', module='sklearn')

class ProductionMusicRecommender:
    """Production-ready music recommendation system with robust ML pipeline"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        self.training_history = []
        
        # Realistic performance expectations
        self.target_metrics = {
            'accuracy': (0.65, 0.75),
            'f1_weighted': (0.60, 0.70),
            'f1_macro': (0.40, 0.55),
            'roc_auc': (0.80, 0.85)
        }
        
        print("🎵 Production-Ready Music Recommender initialized")
    
    def load_and_analyze_data(self) -> pd.DataFrame:
        """Load data with comprehensive analysis"""
        print("📊 Loading and analyzing dataset...")
        
        df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {df.shape[0]} tracks, {df.shape[1]} features")
        
        # Data quality checks
        print("\n🔍 Data Quality Analysis:")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicate tracks: {df.duplicated().sum()} ({df.duplicated().mean():.1%})")
        
        # Target distribution analysis
        print("\n📈 Mood Distribution:")
        mood_counts = df['Mood_Label'].value_counts()
        for mood, count in mood_counts.items():
            print(f"  {mood}: {count} ({count/len(df)*100:.1f}%)")
        
        # Identify severely underrepresented classes
        self.minority_classes = mood_counts[mood_counts < 20].index.tolist()
        print(f"\n⚠️  Severely underrepresented classes: {self.minority_classes}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create mood-predictive features from audio characteristics"""
        print("⚗️ Engineering features for mood prediction...")
        
        df_enhanced = df.copy()
        
        # Remove duplicates
        df_enhanced = df_enhanced.drop_duplicates()
        print(f"After deduplication: {len(df_enhanced)} tracks")
        
        # Core audio features (NO MOOD AS INPUT!)
        self.audio_features = [
            "Danceability", "Energy", "Key", "Loudness", "Mode",
            "Speechiness", "Acousticness", "Instrumentalness", 
            "Liveness", "Valence", "Tempo", "Duration (ms)",
            "Artist Popularity", "Track Popularity"
        ]
        
        # Engineered features for better mood prediction
        df_enhanced['energy_valence'] = df_enhanced['Energy'] * df_enhanced['Valence']
        df_enhanced['acoustic_energy_ratio'] = df_enhanced['Acousticness'] / (df_enhanced['Energy'] + 0.01)
        df_enhanced['mood_intensity'] = (df_enhanced['Energy'] + df_enhanced['Danceability'] + df_enhanced['Valence']) / 3
        df_enhanced['rhythmic_complexity'] = df_enhanced['Tempo'] * df_enhanced['Danceability']
        df_enhanced['emotional_depth'] = df_enhanced['Acousticness'] * df_enhanced['Valence']
        
        # Tempo categories
        df_enhanced['tempo_slow'] = (df_enhanced['Tempo'] < 80).astype(int)
        df_enhanced['tempo_moderate'] = ((df_enhanced['Tempo'] >= 80) & (df_enhanced['Tempo'] < 120)).astype(int)
        df_enhanced['tempo_fast'] = (df_enhanced['Tempo'] >= 120).astype(int)
        
        # Energy categories  
        df_enhanced['energy_low'] = (df_enhanced['Energy'] < 0.4).astype(int)
        df_enhanced['energy_high'] = (df_enhanced['Energy'] > 0.7).astype(int)
        
        # Valence categories
        df_enhanced['valence_negative'] = (df_enhanced['Valence'] < 0.4).astype(int)
        df_enhanced['valence_positive'] = (df_enhanced['Valence'] > 0.6).astype(int)
        
        # Final feature set
        self.feature_columns = self.audio_features + [
            'energy_valence', 'acoustic_energy_ratio', 'mood_intensity',
            'rhythmic_complexity', 'emotional_depth',
            'tempo_slow', 'tempo_moderate', 'tempo_fast',
            'energy_low', 'energy_high', 'valence_negative', 'valence_positive'
        ]
        
        print(f"Total features: {len(self.feature_columns)}")
        return df_enhanced
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data with robust handling of severely imbalanced classes"""
        print("🔧 Preparing data with imbalanced class handling...")
        
        # Features (NO MOOD LABELS!)
        X = df[self.feature_columns].copy()
        y = df['Mood_Label'].copy()
        
        # Check class distribution before encoding
        class_counts = y.value_counts()
        print(f"Classes with <5 samples: {class_counts[class_counts < 5].to_dict()}")
        
        # Remove classes with too few samples (< 6) for robust train/val/test splitting
        classes_to_remove = class_counts[class_counts < 6].index.tolist()
        if classes_to_remove:
            print(f"🗑️ Removing classes with <6 samples: {classes_to_remove}")
            print(f"Removed classes and counts: {class_counts[class_counts < 6].to_dict()}")
            mask = ~y.isin(classes_to_remove)
            X = X[mask]
            y = y[mask]
            print(f"Dataset after removal: {len(X)} samples")
            
            # Update class counts
            class_counts = y.value_counts()
            print(f"Remaining classes: {len(class_counts)}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Update minority classes after removal
        remaining_counts = y.value_counts()
        self.minority_classes = remaining_counts[remaining_counts < 20].index.tolist()
        
        # Check if stratified splitting is possible - need at least 6 samples per class
        # (2 for train, 2 for val, 2 for test with 70/15/15 split)
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        min_class_count = min(class_counts)
        
        print(f"Minimum class count after removal: {min_class_count}")
        
        if min_class_count >= 6:
            # Use stratified splitting when possible
            print("✅ Using stratified splitting")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
        elif min_class_count >= 3:
            # Try single stratified split then random for validation/test
            print("⚠️ Using mixed splitting strategy")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
            # Random split for validation and test since stratification may fail
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
        else:
            # Fallback to random splitting for very small classes
            print("⚠️ Using random splitting due to extremely small classes")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
        
        print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        # Scale features AFTER splitting (prevent data leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)  
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance with conservative SMOTE
        print("⚖️ Handling class imbalance with SMOTE...")
        
        # Only apply SMOTE to classes that have enough samples in training set
        train_class_counts = np.bincount(y_train)
        
        sampling_strategy = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            if class_name in self.minority_classes and i < len(train_class_counts):
                current_count = train_class_counts[i]
                if current_count >= 2:  # Need at least 2 samples for SMOTE
                    target_count = min(30, current_count * 2)  # Very conservative increase
                    if target_count > current_count:
                        sampling_strategy[i] = target_count
        
        if sampling_strategy:
            # Use k_neighbors based on smallest class size
            min_samples = min([train_class_counts[i] for i in sampling_strategy.keys()])
            k_neighbors = min(3, min_samples - 1)  # k must be < min class size
            
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=max(1, k_neighbors),
                random_state=42
            )
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: {len(X_train_balanced)} training samples")
        else:
            print("⏭️ Skipping SMOTE - no suitable classes for oversampling")
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        return X_train_balanced, X_val_scaled, X_test_scaled, y_train_balanced, y_val, y_test
    
    def create_robust_models(self) -> Dict[str, Any]:
        """Create regularized models to prevent overfitting"""
        print("🤖 Creating regularized ensemble models...")
        
        # Stratified CV for proper evaluation
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,         # Limited depth to prevent overfitting
                min_samples_split=10, # Increased to prevent overfitting
                min_samples_leaf=5,   # Increased to prevent overfitting
                max_features='sqrt',  # Reduce feature randomness
                class_weight='balanced',
                n_jobs=4,            # Fixed CPU count to avoid detection issues
                random_state=42
            ),
            
            "XGBoost": xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,   # Lower learning rate
                max_depth=6,          # Limited depth
                min_child_weight=5,   # Regularization
                subsample=0.8,        # Regularization
                colsample_bytree=0.8, # Regularization
                reg_alpha=0.1,        # L1 regularization
                reg_lambda=1.0,       # L2 regularization
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0          # Suppress XGBoost warnings
            ),
            
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=100,      # Reduced to prevent overfitting
                learning_rate=0.1,     # Increased for faster convergence
                max_depth=4,           # Reduced depth
                num_leaves=15,         # Reduced leaves
                min_child_samples=50,  # Increased regularization
                min_child_weight=1e-2, # Increased regularization
                reg_alpha=0.3,         # Increased L1 regularization
                reg_lambda=0.3,        # Increased L2 regularization
                class_weight='balanced',
                verbosity=-1,          # Suppress warnings
                random_state=42
            ),
            
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=20,  # Regularization
                min_samples_leaf=10,   # Regularization  
                subsample=0.8,         # Regularization
                random_state=42
            ),
            
            "SVM": SVC(
                C=1.0,                 # Moderate regularization
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,      # For ROC-AUC
                random_state=42
            ),
            
            "MLP": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                learning_rate_init=0.001,
                alpha=0.01,            # L2 regularization
                max_iter=500,
                early_stopping=True,  # Prevent overfitting
                validation_fraction=0.2,
                random_state=42
            )
        }
        
        return models, cv_strategy
    
    def evaluate_model_robust(self, model, X_train, X_val, X_test, y_train, y_val, y_test) -> Dict[str, Dict[str, float]]:
        """Comprehensive evaluation with realistic metrics"""
        
        # Fit model with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        
        # Predictions with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            val_proba = model.predict_proba(X_val)
            test_proba = model.predict_proba(X_test)
        
        results = {
            'validation': {
                'accuracy': accuracy_score(y_val, val_pred),
                'f1_weighted': f1_score(y_val, val_pred, average='weighted'),
                'f1_macro': f1_score(y_val, val_pred, average='macro'),
                'roc_auc': roc_auc_score(y_val, val_proba, multi_class='ovr', average='weighted')
            },
            'test': {
                'accuracy': accuracy_score(y_test, test_pred), 
                'f1_weighted': f1_score(y_test, test_pred, average='weighted'),
                'f1_macro': f1_score(y_test, test_pred, average='macro'),
                'roc_auc': roc_auc_score(y_test, test_proba, multi_class='ovr', average='weighted')
            }
        }
        
        return results
    
    def train_and_evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test) -> Dict[str, Any]:
        """Train all models and create production ensemble"""
        print("🏋️ Training ensemble models with robust evaluation...")
        
        models, cv_strategy = self.create_robust_models()
        results = {}
        
        # Train individual models with progress tracking
        model_names = list(models.keys())
        
        with tqdm(total=len(model_names), desc="🤖 Training ML Models", unit="model") as pbar:
            for name, model in models.items():
                pbar.set_description(f"🤖 Training {name}")
                
                # Cross-validation on training set
                print(f"\n📈 Training {name}...")
                print("  ⏳ Running cross-validation...")
                
                # Suppress warnings during training
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_validate(
                        model, X_train, y_train,
                        cv=cv_strategy,
                        scoring=['accuracy', 'f1_weighted', 'f1_macro'],
                        n_jobs=4  # Fixed CPU count
                    )
                
                print(f"  ✅ CV Accuracy: {cv_scores['test_accuracy'].mean():.3f} ± {cv_scores['test_accuracy'].std():.3f}")
                print(f"  ✅ CV F1-Weighted: {cv_scores['test_f1_weighted'].mean():.3f} ± {cv_scores['test_f1_weighted'].std():.3f}")
                
                # Full evaluation
                print("  ⏳ Running full evaluation...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_results = self.evaluate_model_robust(model, X_train, X_val, X_test, y_train, y_val, y_test)
                results[name] = model_results
                
                # Store trained model
                self.models[name] = model
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = np.abs(model.coef_).mean(axis=0)
                
                # Update progress bar
                pbar.set_postfix({
                    'Current': name,
                    'CV_Acc': f"{cv_scores['test_accuracy'].mean():.3f}",
                    'Test_Acc': f"{model_results['test']['accuracy']:.3f}"
                })
                pbar.update(1)
        
        # Create stacking ensemble
        print("\n🎯 Creating stacking ensemble...")
        
        # Select top 3 models based on validation F1-weighted
        model_performance = [(name, results[name]['validation']['f1_weighted']) for name in results.keys()]
        model_performance.sort(key=lambda x: x[1], reverse=True)
        top_models = [name for name, _ in model_performance[:3]]
        
        print(f"✨ Top models for ensemble: {top_models}")
        
        # Stacking ensemble with progress tracking
        with tqdm(total=3, desc="🔧 Building Ensemble", unit="step") as ens_pbar:
            ens_pbar.set_description("🔧 Configuring base models...")
            base_estimators = [(name, models[name]) for name in top_models]
            meta_learner = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                C=0.1  # Regularization
            )
            ens_pbar.update(1)
            
            ens_pbar.set_description("🔧 Training stacking ensemble...")
            stacking_ensemble = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=cv_strategy,
                n_jobs=4  # Fixed CPU count
            )
            ens_pbar.update(1)
            
            ens_pbar.set_description("🔧 Evaluating ensemble...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ensemble_results = self.evaluate_model_robust(
                    stacking_ensemble, X_train, X_val, X_test, y_train, y_val, y_test
                )
            results['StackingEnsemble'] = ensemble_results
            self.models['StackingEnsemble'] = stacking_ensemble
            ens_pbar.set_postfix({
                'Ensemble_Acc': f"{ensemble_results['test']['accuracy']:.3f}"
            })
            ens_pbar.update(1)
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]):
        """Analyze results with realistic expectations"""
        print("\n📊 Performance Analysis:")
        print("=" * 60)
        
        for model_name, model_results in results.items():
            print(f"\n{model_name}:")
            val_results = model_results['validation']
            test_results = model_results['test']
            
            print(f"  Validation - Acc: {val_results['accuracy']:.3f}, F1-W: {val_results['f1_weighted']:.3f}, F1-M: {val_results['f1_macro']:.3f}, AUC: {val_results['roc_auc']:.3f}")
            print(f"  Test       - Acc: {test_results['accuracy']:.3f}, F1-W: {test_results['f1_weighted']:.3f}, F1-M: {test_results['f1_macro']:.3f}, AUC: {test_results['roc_auc']:.3f}")
            
            # Check if performance is realistic
            test_acc = test_results['accuracy']
            if test_acc > 0.85:
                print(f"  ⚠️ HIGH ACCURACY ({test_acc:.1%}) - Potential overfitting!")
            elif self.target_metrics['accuracy'][0] <= test_acc <= self.target_metrics['accuracy'][1]:
                print(f"  ✅ REALISTIC PERFORMANCE ({test_acc:.1%}) - Good for production")
            else:
                print(f"  📉 Below target performance ({test_acc:.1%})")
        
        # Recommend best model
        best_model = max(results.keys(), 
                        key=lambda k: results[k]['test']['f1_weighted'])
        print(f"\n🏆 Best model: {best_model}")
        print(f"   Test F1-Weighted: {results[best_model]['test']['f1_weighted']:.3f}")
        
        return best_model
    
    def save_production_model(self, best_model_name: str):
        """Save production-ready model and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = f"models/music_recommender_{best_model_name.lower()}_{timestamp}.pkl"
        Path("models").mkdir(exist_ok=True)
        
        production_package = {
            'model': self.models[best_model_name],
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_name': best_model_name,
            'training_date': timestamp,
            'target_metrics': self.target_metrics
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(production_package, f)
        
        print(f"✅ Production model saved: {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': best_model_name,
            'training_date': timestamp,
            'feature_count': len(self.feature_columns),
            'target_classes': self.label_encoder.classes_.tolist(),
            'performance_targets': self.target_metrics
        }
        
        metadata_path = f"models/metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path

def main():
    """Main training pipeline"""
    print("🚀 Y.M.I.R Production Music Recommendation Training")
    print("=" * 60)
    
    # Initialize system
    data_path = "datasets/therapeutic_music_enriched.csv"
    recommender = ProductionMusicRecommender(data_path)
    
    # Load and analyze data
    df = recommender.load_and_analyze_data()
    
    # Feature engineering
    df_enhanced = recommender.engineer_features(df)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = recommender.prepare_data(df_enhanced)
    
    # Train and evaluate
    results = recommender.train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Analyze results
    best_model = recommender.analyze_results(results)
    
    # Save production model
    model_path = recommender.save_production_model(best_model)
    
    print(f"\n🎉 Training complete! Production model saved to: {model_path}")
    print("\n📋 Next steps:")
    print("1. Test model with real emotion detection input")
    print("2. Deploy to production environment")
    print("3. Monitor performance and retrain as needed")
    
    return recommender, results

if __name__ == "__main__":
    recommender, results = main()