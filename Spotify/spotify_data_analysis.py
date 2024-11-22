# Objective 1: Analyze the Relationship Between Popularity and Audio Features#


from google.cloud import bigquery
import pandas as pd

# Initialize BigQuery client
client = bigquery.Client()

# Your SQL query to fetch the data 
query = """
    SELECT
        year,
        popularity,
        danceability,
        energy,
        loudness,
        tempo
    FROM `nicholas-project-441610.spotify_dataset.spotify_tracks`
    LIMIT 1000
"""

# Run the query
query_job = client.query(query)

# Get results into a pandas dataframe
df = query_job.to_dataframe()

# Check the first few rows of the dataframe to verify data
print(df.head())

# Data Exploration
print(f"Data Types: {df.dtypes}")
print(f"Summary Statistics: {df.describe()}")

# Data Cleaning
# Check for missing values
print(df.isnull().sum())  # Check for missing values

# Drop rows with missing values or fill them (choose your cleaning strategy)
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Ensure correct data types
df['popularity'] = df['popularity'].astype(float)
df['danceability'] = df['danceability'].astype(float)
df['energy'] = df['energy'].astype(float)
df['tempo'] = df['tempo'].astype(float)
df['year'] = df['year'].astype(int)  # Ensure 'year' is treated as integer

# Final data check
print(df.head())
print(df.dtypes)  # Verify data types

df_objective1 = df.copy() # Create a copy of the cleaned dataframe
df_objective1.to_csv("spotify_data_objective1_cleaned.csv", index=False)

# Now that we have cleaned the data, let's explore the relationship between popularity and audio features
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation matrix
correlation = df[['popularity', 'danceability', 'energy', 'loudness', 'tempo']].corr()
print(correlation)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Relationship Between Popularity and Audio Features")
plt.show()

# Visualize distribution of popularity vs. audio features
plt.figure(figsize=(12, 8))
sns.pairplot(df[['popularity', 'danceability', 'energy', 'loudness', 'tempo']])
plt.suptitle('Popularity vs. Audio Features', y=1.02)
plt.show()


