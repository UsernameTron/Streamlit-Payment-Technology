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
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(page_title="Engagement & Sentiment Analysis", layout="wide")
logging.basicConfig(level=logging.INFO)

# Download NLTK VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# =========================
# Helper & Utility Functions
# =========================

def shorten_url(url, max_length=30):
    """
    Shorten a URL for display. If it's too long, return a truncated version.
    Full URL can remain in hover data for context.
    """
    if isinstance(url, str) and len(url) > max_length:
        return url[:max_length] + "..."
    return url

def categorize_sentiment(compound_score):
    """
    Convert a VADER compound sentiment score into categories.
    VADER compound score range: -1 (most negative) to 1 (most positive).
    Thresholds:
      compound >= 0.05 => Positive
      compound <= -0.05 => Negative
      otherwise => Neutral
    """
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def align_sentiments(engagement_data, sentiments):
    """
    Align sentiment categories with engagement data rows.
    If fewer sentiments, fill remainder as 'Neutral'.
    If more, truncate extra sentiments.
    """
    num_data = len(engagement_data)
    num_sents = len(sentiments)
    
    if num_sents < num_data:
        sentiments += ["Neutral"] * (num_data - num_sents)
    else:
        sentiments = sentiments[:num_data]
    
    engagement_data["Sentiment"] = sentiments
    return engagement_data

def add_reference_lines(fig, engagement_data):
    """
    Add reference lines to the scatter plot, for example the average engagement rate.
    """
    avg_engagement = engagement_data["Engagement Rate (%)"].mean(skipna=True)
    if not np.isnan(avg_engagement):
        fig.add_hline(y=avg_engagement, line_dash="dot", line_color="red",
                      annotation_text=f"Avg ER: {avg_engagement:.2f}%", 
                      annotation_position="top left")

def generate_summaries(engagement_data):
    """
    Compute summary metrics: average engagement rate, count by sentiment, etc.
    Also returns the date range if available.
    """
    avg_er = engagement_data["Engagement Rate (%)"].mean()
    sentiment_counts = engagement_data["Sentiment"].value_counts(dropna=False).to_dict()
    
    if "Post publish date" in engagement_data.columns and pd.api.types.is_datetime64_any_dtype(engagement_data["Post publish date"]):
        min_date = engagement_data["Post publish date"].min()
        max_date = engagement_data["Post publish date"].max()
    else:
        min_date = max_date = None
    
    return avg_er, sentiment_counts, min_date, max_date

def filter_data(engagement_data, date_range, sentiment_filter, er_threshold):
    """
    Filter the engagement_data based on user-selected criteria:
    - date_range: a tuple of (start_date, end_date)
    - sentiment_filter: a list of sentiments to include
    - er_threshold: minimum engagement rate
    """
    filtered = engagement_data.copy()
    
    if "Post publish date" in filtered.columns and date_range[0] and date_range[1]:
        start, end = date_range
        filtered = filtered[(filtered["Post publish date"] >= start) & (filtered["Post publish date"] <= end)]
    
    if sentiment_filter:
        filtered = filtered[filtered["Sentiment"].isin(sentiment_filter)]
    
    if er_threshold is not None:
        filtered = filtered[filtered["Engagement Rate (%)"] >= er_threshold]
    
    return filtered

@st.cache_data
def compute_sentiments_with_vader(html_texts):
    """
    Compute sentiment categories from extracted text using VADER.
    This function is cached to avoid recomputing on the same data.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for text in html_texts:
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        sentiments.append(categorize_sentiment(compound))
    return sentiments

def add_forecast(engagement_data):
    """
    Add a simple linear regression forecast of future impressions based on date.
    Displays a predicted impressions value 30 days from the max date.
    """
    if "Post publish date" in engagement_data.columns and not engagement_data["Post publish date"].isna().all():
        df = engagement_data.dropna(subset=["Impressions", "Post publish date"])
        if len(df) > 5:  # Ensure we have enough data points
            df["date_ordinal"] = df["Post publish date"].map(datetime.datetime.toordinal)
            X = df[["date_ordinal"]]
            y = df["Impressions"]
            model = LinearRegression().fit(X, y)
            
            future_date = (df["Post publish date"].max() + pd.Timedelta(days=30))
            future_ordinal = future_date.to_pydatetime().toordinal()
            future_pred = model.predict([[future_ordinal]])[0]
            st.write(f"**Forecast:** Predicted Impressions in 30 days: {future_pred:.2f}")

# =========================
# Data Processing Functions
# =========================

def process_html_files(html_files):
    """
    Extract text from uploaded HTML files and compute sentiments using VADER.
    """
    html_texts = []
    for html_file in html_files:
        try:
            soup = BeautifulSoup(html_file.getvalue(), "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            html_texts.append(text)
            st.write(f"Processed file: {html_file.name}")
        except Exception as e:
            st.error(f"Error processing HTML file {html_file.name}: {e}")
    sentiments = compute_sentiments_with_vader(html_texts)
    return sentiments

def process_excel_file(excel_file):
    """
    Process the engagement Excel file, ensuring required columns exist and data is clean.
    """
    try:
        df = pd.read_excel(excel_file, sheet_name="TOP POSTS", header=2)
        
        required_cols = {"Post URL", "Post publish date", "Engagements", "Impressions"}
        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {missing}")
            return None
        
        df = df[list(required_cols)].copy()
        df["Engagements"] = pd.to_numeric(df["Engagements"], errors="coerce").fillna(0)
        df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)
        
        df["Engagement Rate (%)"] = np.where(
            df["Impressions"] > 0,
            (df["Engagements"] / df["Impressions"]) * 100,
            np.nan
        )
        
        # Drop rows without URLs or publish dates
        df.dropna(subset=["Post URL", "Post publish date"], inplace=True)
        df["Post publish date"] = pd.to_datetime(df["Post publish date"], errors="coerce")
        df.reset_index(drop=True, inplace=True)
        
        if df.empty:
            st.error("No valid post data found in the engagement file.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
        return None

# =========================
# Visualization Functions
# =========================

def create_visuals(engagement_data):
    """
    Create and display visuals for engagement data with sentiment.
    Adds unique keys to avoid duplicate element ID errors.
    Shows summaries, forecasts, and two main charts.
    """
    # Display summary metrics
    avg_er, sentiment_counts, min_date, max_date = generate_summaries(engagement_data)
    st.markdown("### Key Insights")
    st.write(f"**Average Engagement Rate:** {avg_er:.2f}%")
    st.write("**Sentiment Counts:**")
    for s, c in sentiment_counts.items():
        st.write(f" - {s}: {c}")
    if min_date and max_date:
        st.write(f"**Date Range:** {min_date.date()} to {max_date.date()}")

    # Show a simple forecast for Impressions
    add_forecast(engagement_data)
    
    # Apply URL shortening for visualization
    engagement_data["Short URL"] = engagement_data["Post URL"].apply(shorten_url)
    
    # Scatter Chart: Engagement Rate vs Impressions
    scatter_fig = px.scatter(
        engagement_data,
        x="Impressions",
        y="Engagement Rate (%)",
        size="Engagements",
        color="Sentiment",
        hover_data={
            "Post URL": True, 
            "Short URL": False, 
            "Engagement Rate (%)": ':.2f', 
            "Impressions": True
        },
        title="Engagement Rate vs Impressions",
        template="plotly_white",
        color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "blue"}
    )
    scatter_fig.update_layout(
        title={
            'text': "Engagement Rate vs Impressions<br><sup>Engagement Rate = (Engagements / Impressions) * 100</sup>",
            'x':0.5,
            'xanchor':'center'
        }
    )
    add_reference_lines(scatter_fig, engagement_data)
    st.plotly_chart(scatter_fig, use_container_width=True, key=f"scatter_{np.random.rand()}")

    # Horizontal Bar Chart: Top Posts by Engagement Rate
    sorted_data = engagement_data.sort_values(by="Engagement Rate (%)", ascending=False)
    bar_fig = px.bar(
        sorted_data,
        x="Engagement Rate (%)",
        y="Short URL",
        color="Sentiment",
        orientation="h",
        hover_data={
            "Post URL": True,
            "Short URL": False,
            "Engagement Rate (%)": ':.2f',
            "Impressions": True
        },
        title="Top Posts Ranked by Engagement Rate",
        template="plotly_white",
        color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "blue"}
    )
    bar_fig.update_yaxes(autorange="reversed")
    bar_fig.update_layout(
        title={
            'text': "Top Posts Ranked by Engagement Rate<br><sup>(based on Engagement Rate (%))</sup>",
            'x':0.5,
            'xanchor':'center'
        }
    )
    st.plotly_chart(bar_fig, use_container_width=True, key=f"bar_{np.random.rand()}")

# =========================
# Main App Layout
# =========================

st.title("Engagement & Sentiment Analysis")

st.sidebar.header("Upload Files")
uploaded_html_files = st.sidebar.file_uploader("Upload HTML files (Optional)", type=["html"], accept_multiple_files=True)
uploaded_excel_file = st.sidebar.file_uploader("Upload Engagement Excel file (Required)", type=["xlsx"])

st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", value=None)
end_date = st.sidebar.date_input("End Date", value=None)
if start_date and end_date and start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

sentiment_filter_options = ["Positive", "Neutral", "Negative"]
sentiment_selected = st.sidebar.multiselect("Filter by Sentiment", sentiment_filter_options, default=sentiment_filter_options)
er_threshold = st.sidebar.slider("Minimum Engagement Rate (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

if st.sidebar.button("Analyze"):
    if not uploaded_excel_file:
        st.error("Please upload an Excel file for engagement analysis.")
    else:
        engagement_data = process_excel_file(uploaded_excel_file)
        if engagement_data is not None and not engagement_data.empty:
            sentiments = process_html_files(uploaded_html_files) if uploaded_html_files else []
            engagement_data = align_sentiments(engagement_data, sentiments)

            date_range = (pd.to_datetime(start_date) if start_date else None, 
                          pd.to_datetime(end_date) if end_date else None)
            filtered_data = filter_data(engagement_data, date_range, sentiment_selected, er_threshold)
            
            if filtered_data.empty:
                st.warning("No data available after applying filters. Try changing the date range or sentiment/ER thresholds.")
            else:
                st.write("### Processed Engagement Data (Filtered)")
                st.dataframe(filtered_data)
                create_visuals(filtered_data)
        else:
            st.error("Engagement data could not be processed. Please check the Excel file.")