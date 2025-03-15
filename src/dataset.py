


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"datasets/therapeutic_music_enriched.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Display basic information
print("Basic Information:")
print(df.info())


print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display unique mood labels
print("\nUnique Mood Labels:")
print(df["Mood_Label"].unique())

# Count occurrences of each mood
print("\nMood Distribution:")
print(df["Mood_Label"].value_counts())

# Display unique mental health benefits
print("\nUnique Mental Health Benefits:")
print(df["Mental_Health_Benefit"].unique())

# Display unique musical features
print("\nUnique Musical Features:")
print(df["Musical_Features"].unique())

# Summary statistics for numerical columns
print("\nStatistical Summary:")
print(df.describe())

# ------ VISUALIZATIONS ------

# 1️⃣ Mood Label Distribution (Bar Chart)
plt.figure(figsize=(12, 6))
sns.countplot(y=df["Mood_Label"], order=df["Mood_Label"].value_counts().index, palette="coolwarm", hue=df["Mood_Label"], legend=False)
plt.title("Mood Label Distribution")
plt.xlabel("Count")
plt.ylabel("Mood Label")
plt.show()

# 2️⃣ Mental Health Benefit Distribution
plt.figure(figsize=(12, 6))
sns.countplot(y=df["Mental_Health_Benefit"], order=df["Mental_Health_Benefit"].value_counts().index, palette="viridis")
plt.title("Mental Health Benefit Distribution")
plt.xlabel("Count")
plt.ylabel("Mental Health Benefit")
plt.show()

# 3️⃣ Correlation Heatmap (Numerical Features)
numeric_df = df.select_dtypes(include=["number"])  # Only numeric columns
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# 4️⃣ Boxplots for Outlier Detection
num_features = ["Danceability", "Energy", "Valence", "Tempo", "Duration (ms)"]
plt.figure(figsize=(14, 8))
df[num_features].boxplot()
plt.title("Outlier Detection in Numerical Features")
plt.xticks(rotation=45)
plt.show()

# 5️⃣ Pairplot (To See Relationships Between Key Features)
sns.pairplot(df[num_features], corner=True)
plt.show()

# 6️⃣ Danceability vs. Energy (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["Danceability"], y=df["Energy"], hue=df["Mood_Label"], palette="tab10", alpha=0.7)
plt.title("Danceability vs. Energy")
plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.show()

