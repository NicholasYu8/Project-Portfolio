# Objective 3: Identify Characteristics of 'Hit' Songs #


from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize BigQuery client
client = bigquery.Client()

# Your SQL query to fetch the data
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

# Run the query and get the results into a pandas dataframe
query_job = client.query(query)
df = query_job.to_dataframe()

# Data Cleaning
df = df.dropna()  # Drop rows with missing values
df = df.drop_duplicates()  # Remove duplicates

# Ensure correct data types
df['popularity'] = df['popularity'].astype(float)
df['danceability'] = df['danceability'].astype(float)
df['energy'] = df['energy'].astype(float)
df['tempo'] = df['tempo'].astype(float)
df['genre'] = df['genre'].astype(str)

# Define 'hit' songs (popularity > 75)
hit_threshold = 75
df['hit'] = df['popularity'] > hit_threshold  

# Visualize the distribution of features for hit vs non-hit songs
plt.figure(figsize=(8, 6))
sns.barplot(x='hit', y='popularity', data=df, errorbar=None)  
plt.title('Average Popularity of Hit vs Non-Hit Songs')
plt.xticks([0, 1], ['Non-Hit', 'Hit'])  # Update x-ticks labels for clarity
plt.show()

# Distribution of key audio features for hit and non-hit songs
features = ['danceability', 'energy', 'loudness', 'tempo']
for feature in features:
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df[df['hit'] == True][feature], label='Hit Songs', fill=True) 
    sns.kdeplot(df[df['hit'] == False][feature], label='Non-Hit Songs', fill=True)  
    plt.title(f'Distribution of {feature} for Hit and Non-Hit Songs')
    plt.legend()
    plt.show()

# Correlation analysis: see how features correlate with popularity
correlation = df[['popularity', 'danceability', 'energy', 'loudness', 'tempo']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Features with Popularity")
plt.show()

# Average features of hit and non-hit songs (excluding 'genre' column)
hit_avg = df[df['hit'] == True].mean(numeric_only=True) 
non_hit_avg = df[df['hit'] == False].mean(numeric_only=True)

print("Average Features for Hit Songs:\n", hit_avg)
print("Average Features for Non-Hit Songs:\n", non_hit_avg)

# Genre analysis: explore which genres are more likely to produce hit songs
plt.figure(figsize=(12, 6))
sns.countplot(x='genre', hue='hit', data=df)  
plt.title('Genre Distribution for Hit and Non-Hit Songs')
plt.xticks(rotation=90)
plt.show()

df_objective3 = df.copy()  # Create a copy of the cleaned dataframe
df_objective3.to_csv("spotify_data_objective3_cleaned.csv", index=False)