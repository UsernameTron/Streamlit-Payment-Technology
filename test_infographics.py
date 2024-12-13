import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def load_html_content(file_path):
    """Extract title and text content from an HTML file."""
    try:
        logging.info(f"Processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            
            # Extract title and text content
            title = soup.title.string if soup.title else "No Title"
            paragraphs = " ".join([p.get_text() for p in soup.find_all("p")])
            return title, paragraphs
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None, None

def analyze_sentiment(text):
    """Perform sentiment analysis using VADER."""
    if not text:
        return {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
    
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores

def process_html_files(file_paths):
    """Process multiple HTML files for sentiment analysis."""
    results = []
    for file_path in file_paths:
        title, content = load_html_content(file_path)
        sentiment = analyze_sentiment(content)
        results.append({
            "File Name": os.path.basename(file_path),
            "Title": title,
            "Compound Sentiment": sentiment["compound"],
            "Positive": sentiment["pos"],
            "Neutral": sentiment["neu"],
            "Negative": sentiment["neg"],
        })
    return pd.DataFrame(results)

def add_engagement_data(sentiment_data, engagement_file):
    """Merge sentiment data with engagement metrics."""
    try:
        engagement_data = pd.read_excel(engagement_file, sheet_name="TOP POSTS")
        engagement_data = engagement_data.rename(columns={
            "Engagements": "Engagements",
            "Impressions": "Impressions",
            "Post URL": "Post URL",
        })

        # Calculate Engagement Rate
        engagement_data["Engagement Rate (%)"] = np.where(
            engagement_data["Impressions"] > 0,
            (engagement_data["Engagements"] / engagement_data["Impressions"]) * 100,
            np.nan
        )
        merged_data = pd.merge(sentiment_data, engagement_data, how="inner", left_on="Title", right_on="Post URL")
        return merged_data
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        return None

def assign_performance_categories(data):
    """Categorize posts based on sentiment and engagement."""
    q1_engagement = data["Engagement Rate (%)"].quantile(0.25)
    q3_engagement = data["Engagement Rate (%)"].quantile(0.75)

    data["Performance Category"] = np.select(
        [
            (data["Compound Sentiment"] >= 0.5) & (data["Engagement Rate (%)"] >= q3_engagement),
            (data["Compound Sentiment"] < 0) & (data["Engagement Rate (%)"] <= q1_engagement),
        ],
        [
            "High Performer",
            "Low Performer",
        ],
        default="Average Performer"
    )
    return data

def predict_engagement(data):
    """Predict engagement rate based on sentiment using linear regression."""
    try:
        X = data[["Compound Sentiment"]].values
        y = data["Engagement Rate (%)"].values

        model = LinearRegression()
        model.fit(X, y)
        data["Predicted Engagement Rate"] = model.predict(X)
        logging.info(f"Model Coefficient: {model.coef_[0]}, Intercept: {model.intercept_}")
        return data
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return data

def visualize_data(data):
    """Generate visualizations for performance categories and sentiment trends."""
    try:
        # Performance Categories
        fig = px.scatter(
            data,
            x="Impressions",
            y="Engagement Rate (%)",
            color="Performance Category",
            size="Impressions",
            hover_data=["Title", "Compound Sentiment", "Engagement Rate (%)"],
            title="Performance Categories: Engagement vs Impressions",
            template="plotly_white",
            height=700
        )
        fig.write_html("performance_categories.html")
        logging.info("Performance categories chart saved as performance_categories.html.")

        # Sentiment vs Engagement Rate
        fig = px.scatter(
            data,
            x="Compound Sentiment",
            y="Engagement Rate (%)",
            color="Performance Category",
            hover_data=["Title"],
            title="Sentiment vs Engagement Rate",
            template="plotly_white",
            height=700
        )
        fig.write_html("sentiment_vs_engagement.html")
        logging.info("Sentiment vs Engagement chart saved as sentiment_vs_engagement.html.")
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")

def main():
    # Input for HTML files and engagement data
    html_files = input("Enter the paths of HTML files (comma-separated): ").split(",")
    engagement_file = input("Enter the path of the engagement Excel file: ").strip()

    html_files = [file.strip() for file in html_files if os.path.exists(file.strip())]
    if not os.path.exists(engagement_file):
        logging.error("Engagement file not found.")
        return

    # Process HTML files for sentiment analysis
    sentiment_data = process_html_files(html_files)
    if sentiment_data.empty:
        logging.error("No sentiment data to process.")
        return

    # Merge with engagement data
    combined_data = add_engagement_data(sentiment_data, engagement_file)
    if combined_data is None:
        logging.error("Failed to merge sentiment and engagement data.")
        return

    # Assign performance categories
    combined_data = assign_performance_categories(combined_data)

    # Predict engagement
    combined_data = predict_engagement(combined_data)

    # Save processed data
    combined_data.to_csv("processed_sentiment_engagement_data.csv", index=False)
    logging.info("Processed data saved as processed_sentiment_engagement_data.csv.")

    # Visualize data
    visualize_data(combined_data)

if __name__ == "__main__":
    main()