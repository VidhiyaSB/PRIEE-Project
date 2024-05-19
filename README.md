# PRIEE-Project
 This project combines sentiment analysis, data visualization, and advanced NLP techniques to provide a powerful tool for businesses seeking to understand and improve their customer relationships and brand reputation.

This project is a Streamlit-based application designed to perform sentiment analysis and brand reputation evaluation using user-uploaded review data. The application leverages several powerful libraries and models to provide comprehensive insights into customer feedback. Key features include:

#File Upload and Data Processing:

Users can upload CSV or XLSX files containing review data.
The application processes the data to ensure consistency and performs sentiment analysis using the VADER sentiment analysis tool.
Reviews are categorized into positive and negative sentiments, and relevant metrics are calculated.
Interactive Visualizations:

The app generates interactive visualizations using Plotly, including sentiment distribution pie charts, trend analysis line charts, and histograms showing sentiment distribution by count and percentage.
These visualizations help users easily explore and interpret the data.


#Word Clouds:

The application creates word clouds for positive and negative reviews, highlighting common themes and frequently mentioned words.
This visual representation helps identify key areas of customer satisfaction and dissatisfaction.
Sample Reviews Display:

Users can view sample positive and negative reviews to get a sense of customer feedback.
This feature provides a quick qualitative overview of the sentiments.


#AI-Powered Brand Improvement Suggestions:

The app uses Hugging Face's text generation (distilgpt2) and text correction (distilbart-cnn-12-6) models to generate suggestions for improving a brand's reputation based on analyzed negative reviews.
This provides actionable insights for businesses to address customer concerns.

