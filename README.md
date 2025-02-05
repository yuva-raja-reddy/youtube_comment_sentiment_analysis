# YouTube Comment Sentiment Analysis 🎥

A real-time sentiment analysis tool for YouTube video comments using DistilBERT multilingual model and Gradio interface.

## Features

- Analyze sentiment of YouTube comments in real-time
- Support for multiple languages
- Interactive pie chart visualization
- Detailed sentiment scores for each comment
- Customizable number of comments to analyze (10-1000)
- User-friendly Gradio web interface

## Tech Stack

- Python 3.8+
- Transformers (DistilBERT)
- YouTube Data API v3
- Gradio
- Pandas
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yuva-raja-reddy/youtube_comment_sentiment_analysis.git
cd youtube_comment_sentiment_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up YouTube API:
   - Get YouTube Data API key from Google Cloud Console
   - Replace API_KEY in multilingual_sentiment_model.py

## Usage
1. Run the application:
```bash
python app.py
```
2. Enter a YouTube video URL
3. Select number of comments to analyze (10-1000)
4. Click Submit to see results:
   - Overall sentiment summary
   - Sentiment distribution pie chart
   - Detailed comment analysis table
  
## Model Details
Uses lxyuan/distilbert-base-multilingual-cased-sentiments-student for sentiment analysis:
   - Supports multiple languages
   - Classifies sentiments as Positive, Neutral, or Negative
   - Provides confidence scores for predictions

## Example URLs
Pre-loaded example URLs for testing:
   - Popular music videos
   - Tech reviews
   - Educational content
   - Viral videos
