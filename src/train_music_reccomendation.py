# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import keras
# import os
# import joblib
# import json

# from keras._tf_keras.keras.models import load_model
# from keras._tf_keras.keras.layers import Embedding, Dense, Flatten, Input, Concatenate, Dropout, BatchNormalization
# from keras._tf_keras.keras.models import Model
# from keras._tf_keras.keras.optimizers import Adam
# from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split

# # ---------------------------- #
# # 1️⃣ Load and Preprocess Data  #
# # ---------------------------- #

# def load_and_preprocess_data(csv_path="datasets/therapeutic_music_enriched.csv"):
#     """Loads and preprocesses dataset for training."""
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"Dataset not found at {csv_path}")

#     df = pd.read_csv(csv_path)

#     # Encode Mood Labels into Numbers
#     mood_encoder = LabelEncoder()
#     df["Mood_Label"] = mood_encoder.fit_transform(df["Mood_Label"])

#     # Normalize Musical Features
#     feature_cols = ["Tempo", "Energy", "Danceability", "Valence"]
#     scaler = MinMaxScaler()
#     df[feature_cols] = scaler.fit_transform(df[feature_cols])

#     return df, mood_encoder, scaler, feature_cols


# # ---------------------------- #
# # 3️⃣ Build Models             #
# # ---------------------------- #

# # Neural Network Model
# def create_nn_model(num_moods, num_features):
#     """Creates a deep learning-based recommendation model."""
#     mood_input = Input(shape=(1,), name="mood_input")
#     mood_embedding = Embedding(input_dim=num_moods, output_dim=10)(mood_input)
#     mood_embedding = Flatten()(mood_embedding)

#     feature_input = Input(shape=(num_features,), name="feature_input")
#     merged = Concatenate()([mood_embedding, feature_input])

#     hidden = Dense(128, activation="relu")(merged)
#     hidden = BatchNormalization()(hidden)
#     hidden = Dropout(0.3)(hidden)
    
#     hidden = Dense(64, activation="relu")(hidden)
#     hidden = BatchNormalization()(hidden)
#     hidden = Dropout(0.3)(hidden)

#     output = Dense(num_moods, activation="softmax")(hidden)

#     model = Model(inputs=[mood_input, feature_input], outputs=output)
#     model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#     return model

# # Other ML Models
# def create_rf_model():
#     return RandomForestClassifier(n_estimators=100, random_state=42)

# def create_xgb_model():
#     return XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# def create_svm_model():
#     return SVC(probability=True, kernel="rbf", C=1.0)

# # ---------------------------- #
# # 4️⃣ Train and Save Models    #
# # ---------------------------- #

# def train_and_save_models(df, mood_encoder, scaler, feature_cols):
#     """Trains multiple models and saves them."""
#     X_moods = df["Mood_Label"].values.reshape(-1, 1)  # Reshaped for correct input
#     X_features = df[feature_cols].values
#     y_songs = df["Mood_Label"].values  

#     # Split dataset
#     X_m_train, X_m_test, X_f_train, X_f_test, y_train, y_test = train_test_split(
#         X_moods, X_features, y_songs, test_size=0.2, random_state=42
#     )

#     # Neural Network Training
#     nn_model = create_nn_model(num_moods=len(mood_encoder.classes_), num_features=len(feature_cols))
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

#     print("\n🚀 Training Neural Network...")
#     nn_model.fit([X_m_train, X_f_train], y_train, epochs=55, batch_size=32, validation_data=([X_m_test, X_f_test], y_test),
#                  callbacks=[early_stopping, reduce_lr])
    
#     nn_model.save("music_recommender_nn.h5")

#     # Train Other Models
#     print("\n🌲 Training Random Forest...")
#     rf_model = create_rf_model()
#     rf_model.fit(np.hstack((X_m_train, X_f_train)), y_train)
#     joblib.dump(rf_model, "music_recommender_rf.pkl")

#     print("\n🔥 Training XGBoost...")
#     xgb_model = create_xgb_model()
#     xgb_model.fit(np.hstack((X_m_train, X_f_train)), y_train)
#     joblib.dump(xgb_model, "music_recommender_xgb.pkl")

#     print("\n⚡ Training SVM...")
#     svm_model = create_svm_model()
#     svm_model.fit(np.hstack((X_m_train, X_f_train)), y_train)
#     joblib.dump(svm_model, "music_recommender_svm.pkl")

#     print("\n✅ All models trained and saved successfully!")


# if __name__ == "__main__":
#     # Step 1: Load and preprocess data
#     print("\n📂 Loading and preprocessing data...")
#     df, mood_encoder, scaler, feature_cols = load_and_preprocess_data()

#     # Step 2: Train and save models (if not already trained)
#     print("\n🚀 Training models...")
#     train_and_save_models(df, mood_encoder, scaler, feature_cols)








































import os
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# === 📌 Load Dataset ===
file_path = r"datasets/therapeutic_music_enriched.csv"
df = pd.read_csv(file_path)

# === 📌 Define Opposite Moods for Uplifting Recommendation ===
opposite_moods = {
    "sad": ["Optimism", "Excitement"],
    "Grief": ["Serenity", "Optimism"],
    "Anger": ["Calmness", "Serenity"],
    "Fear": ["Reassurance", "Emotional Stability"],
    "Anxiety": ["Relaxation", "Stress Relief"],
    "Loneliness": ["Connection", "Upliftment"],
    "Frustration": ["Positivity", "Stability"],
    "neutral": ["neutral"]
}

# === 📌 Data Preprocessing ===
features = ["Danceability", "Energy", "Key", "Loudness", "Speechiness", 
            "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo"]
X = df[features]

# Encode Mood Labels
le = LabelEncoder()
df["Mood_Label_Encoded"] = le.fit_transform(df["Mood_Label"])
y = df["Mood_Label_Encoded"]

# Normalize dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction (retain 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Handle class imbalance
smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=1)  # Reduce k_neighbors
X_balanced, y_balanced = smote.fit_resample(X_pca, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Create a directory to store models
os.makedirs("models", exist_ok=True)

# === 📌 Hyperparameter Tuning ===
def tune_model(model, param_grid, name):
    search = RandomizedSearchCV(model, param_grid, n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"✅ Best parameters for {name}: {search.best_params_}")
    return search.best_estimator_

# Define Models & Tune Hyperparameters
models = {
    "RandomForest": tune_model(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None], "min_samples_split": [2, 5, 10]},
        "RandomForest"
    ),
    "XGBoost": tune_model(
        XGBClassifier(eval_metric="mlogloss", random_state=42),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 6, 9]},
        "XGBoost"
    ),
    "LightGBM": tune_model(
        LGBMClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "num_leaves": [10, 20, 31]},
        "LightGBM"
    ),
    "SVM": tune_model(
        SVC(kernel="rbf", probability=True),
        {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        "SVM"
    ),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
}

trained_models = {}
performance_results = {}

# Train and Evaluate Models
for name, model in models.items():
    print(f"🔄 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Compute Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr") if y_proba is not None else None

    print(f"✅ {name} Accuracy: {accuracy:.4f}")

    # Save the trained model
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    trained_models[name] = model
    performance_results[name] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "roc_auc": roc_auc}

# Save performance results
with open("models/model_performance.json", "w") as f:
    json.dump(performance_results, f, indent=4)

# Save the label encoder, scaler, and PCA
for obj, name in [(le, "label_encoder"), (scaler, "scaler"), (pca, "pca")]:
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

# === 📌 Voting Classifier (Using Top 3 Models) ===
top_models = sorted(performance_results.keys(), key=lambda x: performance_results[x]["f1_score"], reverse=True)[:3]

ensemble_model = VotingClassifier(
    estimators=[(name, trained_models[name]) for name in top_models],
    voting="soft"
)

# Train & Save Ensemble Model
ensemble_model.fit(X_train, y_train)
with open("models/ensemble_model.pkl", "wb") as f:
    pickle.dump(ensemble_model, f)

# Evaluate Ensemble Model
y_pred_ensemble = ensemble_model.predict(X_test)
y_proba_ensemble = ensemble_model.predict_proba(X_test)
ensemble_performance = {
    "accuracy": accuracy_score(y_test, y_pred_ensemble),
    "precision": precision_score(y_test, y_pred_ensemble, average="weighted", zero_division=0),
    "recall": recall_score(y_test, y_pred_ensemble, average="weighted", zero_division=0),
    "f1_score": f1_score(y_test, y_pred_ensemble, average="weighted", zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba_ensemble, multi_class="ovr")
}

# Save final performance
performance_results["EnsembleModel"] = ensemble_performance
with open("models/model_performance.json", "w") as f:
    json.dump(performance_results, f, indent=4)

print("\n✅ All models trained, optimized, and evaluated successfully!")
