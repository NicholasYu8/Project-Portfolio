from google.cloud import bigquery
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Initialize BigQuery client
client = bigquery.Client()

# Query to load your dataset (example)
query = """
    SELECT
        popularity,
        danceability,
        energy,
        loudness,
        tempo,
        genre
    FROM `nicholas-project-441610.spotify_dataset.spotify_tracks`
    LIMIT 1000
"""

# Run the query to get data
query_job = client.query(query)
df = query_job.to_dataframe()  # Load the data into the DataFrame

# Data Cleaning
df = df.dropna()  # Remove missing values
df = df.drop_duplicates()  # Remove duplicates

# Convert genre to a numeric value
label_encoder = LabelEncoder()
df['genre'] = label_encoder.fit_transform(df['genre'])

# Split the dataset into features (X) and target variable (y)
X = df[['danceability', 'energy', 'loudness', 'tempo', 'genre']]
y = df['popularity']  # Target variable is 'popularity'

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression Model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Predictions: {y_pred[:5]}")  # Display some predictions for a quick check

# Adjusting the threshold for classification (e.g., changing the threshold to 60)
hit_threshold = 60  # Lower the threshold
y_pred_class = [1 if val > hit_threshold else 0 for val in y_pred]
y_test_class = [1 if val > hit_threshold else 0 for val in y_test]

# saving Classification Report

# Convert the target variable and predictions into binary labels for classification
# Define a threshold for popularity (e.g., 50)
threshold = 50

# Apply binning to the true labels and predictions
y_test_binned = (y_test > threshold).astype(int)  # 0 for not popular, 1 for popular
y_pred_binned = (y_pred > threshold).astype(int)  # Same threshold applied to predictions

# Generate the classification report
report = classification_report(y_test_binned, y_pred_binned)

# Print the classification report
print("Classification Report:\n", report)

# Save the classification report to a file
output_path = "classification_report.txt"
with open(output_path, "w") as file:
    file.write(report)

print(f"Classification report saved as '{output_path}'.")


# Visualization 1: Actual vs Predicted Popularity (Scatter Plot)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line of equality
plt.title("Actual vs Predicted Popularity")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.legend()
plt.grid(True)
plt.show()

# Visualization 2: Feature Importance (Bar Plot)
# If you're using a tree-based model like Random Forest or Gradient Boosting, you can plot feature importance
import seaborn as sns

# For a linear model (e.g., Linear Regression), we can use the coefficients (model.coef_)
if hasattr(model, 'coef_'):
    feature_importance = model.coef_
elif hasattr(model, 'feature_importances_'):
    feature_importance = model.feature_importances_

# Get the names of the features
feature_names = ['danceability', 'energy', 'loudness', 'tempo', 'genre']

# Create a DataFrame for easier plotting
import pandas as pd
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort values in descending order to make the plot more readable
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', hue='Feature')
plt.title("Feature Importance in Predicting Popularity")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

df_objective4 = df.copy()  # Create a copy of the cleaned dataframe
df_objective4.to_csv("spotify_data_objective4_cleaned.csv", index=False)