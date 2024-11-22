from google.cloud import bigquery
import pandas as pd

# Initialize BigQuery Client
client = bigquery.Client()

# Define your BigQuery SQL query
query = """
    SELECT
        *
    FROM
        `nicholas-project-441610.spotify_dataset.spotify_tracks`
"""

# Run the query and convert the result into a DataFrame
df_cleaned = client.query(query).to_dataframe()

# Perform your data cleaning operations
# Drop rows with missing values
df_cleaned = df_cleaned.dropna()

# Remove duplicate rows
df_cleaned = df_cleaned.drop_duplicates()

# Ensure columns are of the correct data type (if necessary)
df_cleaned['genre'] = df_cleaned['genre'].astype(str)
df_cleaned['popularity'] = df_cleaned['popularity'].astype(float)

# Save the final cleaned dataset to a new CSV file
df_cleaned.to_csv("final_cleaned_spotify_dataset.csv", index=False)

print("Final cleaned dataset saved as 'final_cleaned_spotify_dataset.csv'")
