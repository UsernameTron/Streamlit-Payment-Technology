import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import os
import logging
import PyPDF2
import docx
import openai
import requests
from openai import OpenAIError, APIError, InvalidRequestError
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox, QTextEdit
from PyQt5.QtGui import QFont

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("PIL").setLevel(logging.WARNING)

# Fetch API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

def detect_file_content(file_path):
    """Detect if the file matches LinkedIn Excel structure or general data."""
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        excel_file = pd.ExcelFile(file_path)
        if set(excel_file.sheet_names) >= {"DISCOVERY", "ENGAGEMENT", "TOP POSTS", "FOLLOWERS", "DEMOGRAPHICS"}:
            return "linkedin"
        else:
            return "general"
    elif file_path.endswith('.csv'):
        return "general"
    elif file_path.endswith('.pdf'):
        return "pdf"
    elif file_path.endswith('.docx'):
        return "word"
    else:
        raise ValueError("Unsupported file type. Please upload Excel, CSV, PDF, or Word files.")

def process_file(file_path=None):
    """Read and process uploaded file."""
    try:
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                None, "Select File", "", "CSV files (*.csv);;Excel files (*.xlsx;*.xls);;PDF files (*.pdf);;Word files (*.docx)"
            )
            if not file_path:
                return

        file_type = detect_file_content(file_path)

        if file_type == "linkedin":
            process_linkedin_data(file_path)
        elif file_type == "pdf":
            text = extract_text_from_pdf(file_path)
            display_text(text)
        elif file_type == "word":
            text = extract_text_from_word(file_path)
            display_text(text)
        else:
            data = process_general_file(file_path)
            column_types = analyze_data(data)
            generate_visuals(data, column_types)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        QMessageBox.critical(None, "Error", f"An error occurred: {e}")

def process_general_file(file_path):
    """Process general CSV or Excel file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
    return text

def extract_text_from_word(file_path):
    """Extract text from a Word document."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def display_text(text):
    """Display extracted text in a message box."""
    text_display = QTextEdit()
    text_display.setPlainText(text)
    text_display.setReadOnly(True)
    text_display.setWindowTitle("Extracted Text")
    text_display.resize(800, 600)
    text_display.show()

def analyze_data(data):
    """Classify columns into numeric, datetime, and categorical types."""
    column_types = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(data[col]):
            column_types[col] = "datetime"
        else:
            column_types[col] = "categorical"
    return column_types

def generate_visuals(data, column_types):
    """Generate appropriate visuals based on column classifications."""
    for col, col_type in column_types.items():
        if col_type == "numeric":
            fig = px.histogram(data, x=col, nbins=10, title=f"Distribution of {col}")
            fig.show()
        elif col_type == "datetime":
            fig = px.line(data, x=col, y=data.columns[1], title=f"Time Series of {col}")
            fig.show()
        elif col_type == "categorical":
            fig = px.bar(data[col].value_counts().reset_index(), x='index', y=col, title=f"Count of {col}")
            fig.show()

def generate_wordcloud(text_column):
    """Generate a word cloud from text data."""
    text = " ".join(str(item) for item in text_column.dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig = px.imshow(wordcloud.to_array(), title="Word Cloud")
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.show()

def process_linkedin_data(file_path):
    """Process and visualize LinkedIn-specific data."""
    excel_file = pd.ExcelFile(file_path)

    discovery = pd.read_excel(file_path, sheet_name="DISCOVERY")
    engagement = pd.read_excel(file_path, sheet_name="ENGAGEMENT")
    top_posts = pd.read_excel(file_path, sheet_name="TOP POSTS")
    followers = pd.read_excel(file_path, sheet_name="FOLLOWERS", header=2)  # Adjust header row
    demographics = pd.read_excel(file_path, sheet_name="DEMOGRAPHICS")

    logging.info(f"Discovery columns: {discovery.columns}")
    logging.info(f"Engagement columns: {engagement.columns}")
    logging.info(f"Top Posts columns: {top_posts.columns}")
    logging.info(f"Followers columns: {followers.columns}")
    logging.info(f"Demographics columns: {demographics.columns}")

    # Rename columns if necessary
    if "Date" not in followers.columns or "New followers" not in followers.columns:
        followers.columns = ["Date", "New followers"]

    visualize_linkedin_data(discovery, engagement, top_posts, followers, demographics)

def visualize_linkedin_data(discovery, engagement, top_posts, followers, demographics):
    """Generate LinkedIn-specific visualizations."""
    # Discovery Metrics
    if "Overall Performance" in discovery.columns and "11/14/2024 - 12/11/2024" in discovery.columns:
        fig = px.bar(discovery, x="Overall Performance", y="11/14/2024 - 12/11/2024", title="Discovery Metrics")
        fig.show()
    else:
        logging.error("Discovery data missing 'Overall Performance' or '11/14/2024 - 12/11/2024' columns")
        logging.info(f"Discovery data: {discovery.head()}")

    # Engagement Metrics
    if "Date" in engagement.columns and "Impressions" in engagement.columns and "Engagements" in engagement.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=engagement["Date"], y=engagement["Impressions"], mode='lines', name='Impressions'))
        fig.add_trace(go.Scatter(x=engagement["Date"], y=engagement["Engagements"], mode='lines', name='Engagements'))
        fig.update_layout(title="Engagement Metrics", xaxis_title="Date", yaxis_title="Count")
        fig.show()
    else:
        logging.error("Engagement data missing 'Date', 'Impressions', or 'Engagements' columns")
        logging.info(f"Engagement data: {engagement.head()}")

    # Top Posts Engagement Rate
    if "Maximum of 50 posts available to include in this list" in top_posts.columns and "Unnamed: 6" in top_posts.columns:
        top_posts["Unnamed: 6"] = pd.to_numeric(top_posts["Unnamed: 6"], errors='coerce')
        top_posts["Engagement Rate"] = top_posts["Unnamed: 6"] / top_posts["Unnamed: 6"]
        fig = px.bar(top_posts, x="Engagement Rate", y="Maximum of 50 posts available to include in this list", orientation='h', title="Top Posts Engagement Rate")
        fig.show()
    else:
        logging.error("Top Posts data missing 'Maximum of 50 posts available to include in this list' or 'Unnamed: 6' columns")
        logging.info(f"Top Posts data: {top_posts.head()}")

    # Followers
    if "Date" in followers.columns and "New followers" in followers.columns:
        followers["New followers"] = pd.to_numeric(followers["New followers"], errors='coerce')
        fig = px.line(followers, x="Date", y="New followers", title="New Followers Over Time")
        fig.show()
    else:
        logging.error("Followers data missing 'Date' or 'New followers' columns")
        logging.info(f"Followers data: {followers.head()}")

    # Demographics
    if "Top Demographics" in demographics.columns and "Percentage" in demographics.columns:
        demographics["Percentage"] = pd.to_numeric(demographics["Percentage"], errors='coerce')
        fig = px.pie(demographics, values="Percentage", names="Top Demographics", title="Audience Demographics")
        fig.show()
    else:
        logging.error("Demographics data missing 'Top Demographics' or 'Percentage' columns")
        logging.info(f"Demographics data: {demographics.head()}")

def main():
    app = QApplication([])

    window = QWidget()
    window.setWindowTitle("File Processor with CSV Support")

    layout = QVBoxLayout()

    label = QLabel("Welcome to the File Processor!")
    label.setFont(QFont("Arial", 14))
    layout.addWidget(label)

    upload_button = QPushButton("Upload and Process File")
    upload_button.setFont(QFont("Arial", 12))
    upload_button.clicked.connect(lambda: process_file())
    layout.addWidget(upload_button)

    window.setLayout(layout)
    window.show()

    app.exec_()

if __name__ == "__main__":
    main()