import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import logging
import datetime
from sklearn.linear_model import LinearRegression
from io import StringIO
import nltk  # Import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(page_title="Engagement & Sentiment Analysis", layout="wide")
logging.basicConfig(level=logging.INFO)

# Download NLTK VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# ElevenLabs API setup
api_key = "sk_ad36e4fe8980cdbd847d0cc03153a999c7404c5edab223a7"  # Replace with your actual API key
tts_client = ElevenLabs(api_key=api_key)

# =========================
# Helper & Utility Functions
# =========================

def shorten_url(url, max_length=30):
    """Shorten a URL for display."""
    return url[:max_length] + "..." if isinstance(url, str) and len(url) > max_length else url

def categorize_sentiment(score):
    """Convert sentiment score into categories."""
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def align_sentiments(engagement_data, sentiments):
    """Align sentiment categories with engagement data rows."""
    num_data = len(engagement_data)
    num_sents = len(sentiments)

    if num_sents < num_data:
        sentiments += ["Neutral"] * (num_data - num_sents)
    else:
        sentiments = sentiments[:num_data]

    engagement_data["Sentiment"] = sentiments
    return engagement_data

def add_reference_lines(fig, engagement_data):
    """Add reference lines to the scatter plot."""
    avg_engagement = engagement_data["Engagement Rate (%)"].mean(skipna=True)
    if not np.isnan(avg_engagement):
        fig.add_hline(
            y=avg_engagement,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Avg ER: {avg_engagement:.2f}%",
            annotation_position="top left"
        )

def generate_recommendations(engagement_data):
    """Generate actionable recommendations based on sentiment and engagement."""
    recommendations = []
    for _, row in engagement_data.iterrows():
        if row["Engagement Rate (%)"] > 20 and row["Sentiment"] == "Positive":
            recommendations.append("Replicate successful strategy.")
        elif row["Engagement Rate (%)"] < 10 and row["Sentiment"] == "Negative":
            recommendations.append("Rethink content strategy.")
        else:
            recommendations.append("Maintain and monitor.")
    engagement_data["Recommendation"] = recommendations
    return engagement_data

def add_forecast(engagement_data):
    """Add a simple linear regression forecast of future impressions based on date."""
    if "Post publish date" in engagement_data.columns and not engagement_data["Post publish date"].isna().all():
        df = engagement_data.dropna(subset=["Impressions", "Post publish date"])
        if len(df) > 5:
            df["date_ordinal"] = df["Post publish date"].map(datetime.datetime.toordinal)
            X = df[["date_ordinal"]]
            y = df["Impressions"]
            model = LinearRegression().fit(X, y)

            future_date = df["Post publish date"].max() + pd.Timedelta(days=30)
            future_ordinal = future_date.toordinal()
            future_pred = model.predict([[future_ordinal]])[0]
            return f"Predicted Impressions in 30 days: {future_pred:.2f}"
    return "Insufficient data for forecasting."

def generate_wordcloud(engagement_data):
    """Generate a word cloud from post titles."""
    if "Post URL" in engagement_data.columns:
        text = " ".join(engagement_data["Post URL"])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wordcloud.to_array(), caption="Word Cloud of Post Titles", use_column_width=True)

@st.cache_data
def compute_sentiments_with_vader(html_texts):
    """Compute sentiment categories from extracted text using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for text in html_texts:
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        sentiment = categorize_sentiment(compound)
        sentiments.append(sentiment)
    return sentiments

def process_html_files(html_files):
    """Extract text from uploaded HTML files and compute sentiments using VADER."""
    html_texts = []
    for html_file in html_files:
        try:
            soup = BeautifulSoup(html_file.getvalue(), "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            html_texts.append(text)
        except Exception as e:
            st.error(f"Error processing HTML file {html_file.name}: {e}")
    return compute_sentiments_with_vader(html_texts)

def process_excel_file(excel_file):
    """Process the engagement Excel file, ensuring required columns exist and data is clean."""
    try:
        df = pd.read_excel(excel_file, sheet_name="TOP POSTS", header=2)
        df.columns = ["Post URL", "Post publish date", "Engagements", "Impressions"]
        df["Engagements"] = pd.to_numeric(df["Engagements"], errors="coerce").fillna(0)
        df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)
        df["Engagement Rate (%)"] = np.where(df["Impressions"] > 0, (df["Engagements"] / df["Impressions"]) * 100, np.nan)
        df.dropna(subset=["Post URL", "Post publish date"], inplace=True)
        df["Post publish date"] = pd.to_datetime(df["Post publish date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
        return None

def create_audio_summary(summary_text):
    """Generate audio summary using ElevenLabs."""
    if api_key:
        audio = tts_client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Replace with desired voice ID
            text=summary_text
        )
        with open("summary_audio.mp3", "wb") as audio_file:
            audio_file.write(audio)
        st.audio("summary_audio.mp3", format="audio/mp3", start_time=0)

# =========================
# Main App Layout
# =========================

st.title("Engagement & Sentiment Analysis")
uploaded_html_files = st.sidebar.file_uploader("Upload HTML files", type=["html"], accept_multiple_files=True)
uploaded_excel_file = st.sidebar.file_uploader("Upload Engagement Excel file (Required)", type=["xlsx"])

if st.sidebar.button("Analyze"):
    if not uploaded_excel_file:
        st.error("Please upload an Excel file.")
    else:
        engagement_data = process_excel_file(uploaded_excel_file)
        if engagement_data is not None:
            sentiments = process_html_files(uploaded_html_files) if uploaded_html_files else []
            engagement_data = align_sentiments(engagement_data, sentiments)
            engagement_data = generate_recommendations(engagement_data)
            st.write("### Processed Engagement Data")
            st.dataframe(engagement_data.style.format({"Post URL": lambda x: f'<a href="{x}">{x}</a>'}, escape=False))
            create_visuals(engagement_data)
            generate_wordcloud(engagement_data)
            forecast = add_forecast(engagement_data)
            st.write(f"### Forecast: {forecast}")
            summary = f"Average engagement rate is {engagement_data['Engagement Rate (%)'].mean():.2f}%. Key recommendations and forecast provided above."
            create_audio_summary(summary)