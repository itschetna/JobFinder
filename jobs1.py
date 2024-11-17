import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Function to clean and extract experience range
def clean_experience(experience):
    """
    Clean and extract experience range from a string.
    """
    numbers = re.findall(r"\d+", experience)  
    return [int(numbers[0]), int(numbers[-1])] if numbers else [0, 0]

# Load the job data
data = pd.read_csv("jobs_info.csv")

# Process experience range
data["Experience Range"] = data["Job Experience"].apply(clean_experience)

# Vectorize skills and titles using TF-IDF
skills_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
title_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

tfidf_skills = skills_vectorizer.fit_transform(data["Key Skills"])
tfidf_titles = title_vectorizer.fit_transform(data["Job Title"])

# Experience similarity function
def experience_similarity(candidate_exp, job_exp_range):
    """
    Compute similarity between candidate's experience and job's experience range.
    """
    if candidate_exp < job_exp_range[0]:
        return max(0, 1 - (job_exp_range[0] - candidate_exp) / max(job_exp_range[0], 1))
    elif candidate_exp > job_exp_range[1]:
        return max(0, 1 - (candidate_exp - job_exp_range[1]) / max(candidate_exp, 1))
    else:
        return 1

# Recommendation function
def recommend_jobs(query_skills, query_title, query_experience):
    """
    Recommend jobs based on skills, title, and experience.
    """
    # Transform input query into TF-IDF vectors
    query_skills_vec = skills_vectorizer.transform([query_skills])
    query_title_vec = title_vectorizer.transform([query_title])

    # Compute cosine similarity for skills and titles
    skills_similarity = cosine_similarity(query_skills_vec, tfidf_skills).flatten()
    title_similarity = cosine_similarity(query_title_vec, tfidf_titles).flatten()

    # Normalize the similarities to a 0-1 range
    skills_similarity = (skills_similarity - skills_similarity.min()) / (skills_similarity.max() - skills_similarity.min() + 1e-5)
    title_similarity = (title_similarity - title_similarity.min()) / (title_similarity.max() - title_similarity.min() + 1e-5)

    # Combine skill and title similarities
    combined_similarity = (skills_similarity + title_similarity) / 2

    # Apply experience similarity
    experience_scores = np.array([experience_similarity(query_experience, x) for x in data["Experience Range"]])
    combined_score = combined_similarity * experience_scores

    # Select top 10 recommendations based on the combined score
    indices = np.argsort(-combined_score)[:10]
    if len(indices) == 0 or combined_score[indices[0]] == 0:  # No recommendations found
        return [{"message": "No recommendations found!"}]

    # Retrieve results and convert to dictionary
    results = data.iloc[indices]
    return results.to_dict(orient='records')

# Visualizations for EDA
def visualize_data(data):
    """
    Perform EDA and visualize the data.
    """
    # Plot distribution of experience ranges
    exp_ranges = [x[1] - x[0] for x in data["Experience Range"]]
    sns.histplot(exp_ranges, bins=10, kde=True)
    plt.title("Distribution of Experience Ranges")
    plt.xlabel("Experience Range")
    plt.ylabel("Frequency")
    plt.show()

    # Bar chart for top skills
    all_skills = ", ".join(data["Key Skills"]).split(", ")
    skill_counts = pd.Series(all_skills).value_counts().head(10)
    skill_counts.plot(kind='bar', title='Top Skills in Job Data', color='skyblue')
    plt.xlabel('Skills')
    plt.ylabel('Frequency')
    plt.show()

# Clustering-based Recommendation Model
def clustering_recommendation(query_skills, n_clusters=5):
    """
    Recommend jobs based on clustering using KMeans.
    """
    # Combine skills and titles TF-IDF vectors
    combined_features = np.hstack([tfidf_skills.toarray(), tfidf_titles.toarray()])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data["Cluster"] = kmeans.fit_predict(combined_features)

    # Assign the query to a cluster
    query_vec = np.hstack([
        skills_vectorizer.transform([query_skills]).toarray(),
        title_vectorizer.transform([""]).toarray()
    ])
    query_cluster = kmeans.predict(query_vec)[0]

    # Return jobs from the same cluster
    cluster_jobs = data[data["Cluster"] == query_cluster]
    return cluster_jobs.head(10).to_dict(orient='records')

# Example usage
if __name__ == "__main__":
    # Perform EDA
    visualize_data(data)

    # Test traditional recommendation model
    results1 = recommend_jobs('java sql linux', 'software developer', 2)
    print("Results for Software Developer with 2 years of experience:")
    print(results1)

    # Test clustering-based recommendation model
    results2 = clustering_recommendation('python sql')
    print("Cluster-based recommendations for skills: python, sql:")
    print(results2)
