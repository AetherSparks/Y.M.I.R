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
import re
import pickle  # For saving and loading models
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 📌 Load Dataset ===
file_path = r"datasets/therapeutic_music_enriched.csv"  # Update the path if needed
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
# Select features for training
features = ["Danceability", "Energy", "Key", "Loudness", "Speechiness", 
            "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo"]
X = df[features]

# Encode Mood Labels
le = LabelEncoder()
df["Mood_Label_Encoded"] = le.fit_transform(df["Mood_Label"])
y = df["Mood_Label_Encoded"]

# Normalize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a directory to store models
os.makedirs("models", exist_ok=True)

# === 📌 Train and Save Multiple Models ===
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, min_data_in_leaf=5, min_split_gain=0.01, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "LogisticRegression": LogisticRegression(max_iter=500),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

trained_models = {}

# Train each model, store it, and save to file
for name, model in models.items():
    print(f"🔄 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ {name} Accuracy: {accuracy:.4f}")
    
    # Save the trained model
    model_path = f"models/{name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    trained_models[name] = model

# Save the label encoder and scaler
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# === 📌 Evaluate Individual Models ===
def evaluate_model(model, name, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Performance of {name}:")

    # Get only the labels that are actually present in y_test
    unique_labels = np.unique(y_test)
    label_names = le.inverse_transform(unique_labels)

    # Use only present labels in classification_report
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=label_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# === 📌 Evaluate All Individual Models ===
for model_name, model in trained_models.items():
    y_pred = model.predict(X_test)
    evaluate_model(model, model_name, y_pred)

# === 📌 Voting Classifier (Combining Top Models) ===
ensemble_model = VotingClassifier(
    estimators=[("rf", trained_models["RandomForest"]), 
                ("xgb", trained_models["XGBoost"]), 
                ("lgbm", trained_models["LightGBM"]),
                ("mlp", trained_models["MLP"])], 
    voting="soft"
)

# Train and save ensemble model
ensemble_model.fit(X_train, y_train)
with open("models/ensemble_model.pkl", "wb") as f:
    pickle.dump(ensemble_model, f)
