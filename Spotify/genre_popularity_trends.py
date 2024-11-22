# Objective 2: Genre Analysis and Popularity Trends Over Time #

from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize BigQuery client
client = bigquery.Client()

# SQL Query to extract data
query = """
    SELECT
        year,
        popularity,
        genre
    FROM `nicholas-project-441610.spotify_dataset.spotify_tracks`
    LIMIT 1000
"""

# Run the query
query_job = client.query(query)

# Get results into a pandas dataframe
df = query_job.to_dataframe()

# Data Cleaning (Handle missing values, duplicates, and data types)
df = df.dropna()  # Remove missing values
df = df.drop_duplicates()  # Remove duplicates
df['year'] = df['year'].astype(int)
df['popularity'] = df['popularity'].astype(float)
df['genre'] = df['genre'].astype(str)  # Ensure genre is treated as string

# Explore Popularity Over Time
popularity_over_time = df.groupby('year')['popularity'].mean().reset_index()

# Visualize Popularity Trends Over Time (Line Graph)
plt.figure(figsize=(12, 6))
plt.plot(popularity_over_time['year'], popularity_over_time['popularity'], marker='o', color='b')
plt.title("Popularity Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Average Popularity")
plt.grid(True)
plt.show()

# Explore Popularity Trends by Genre
popularity_by_genre = df.groupby(['year', 'genre'])['popularity'].mean().reset_index()

# Visualize Popularity Across Genres Over Time (Line Plot)
plt.figure(figsize=(12, 8))
sns.lineplot(x='year', y='popularity', hue='genre', data=popularity_by_genre, markers=True)
plt.title("Popularity Trends by Genre Over Time")
plt.xlabel("Year")
plt.ylabel("Average Popularity")
plt.grid(True)
plt.legend(title="Genre")
plt.show()

# Genre Popularity Heatmap (Optional: Correlation between year and popularity by genre)
genre_popularity_pivot = df.pivot_table(index='year', columns='genre', values='popularity', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(genre_popularity_pivot, cmap='coolwarm', annot=True, fmt='.2f')
plt.title("Genre Popularity Heatmap")
plt.xlabel("Genre")
plt.ylabel("Year")
plt.show()

# Summary
print(f"Data types:\n{df.dtypes}")
print(f"Summary Statistics:\n{df.describe()}")

df_objective2 = df.copy()  # Create a copy of the cleaned dataframe
df_objective2.to_csv("spotify_data_objective2_cleaned.csv", index=False)