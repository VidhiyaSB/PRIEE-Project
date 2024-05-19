import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pathlib
import textwrap
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Function to generate responses using text generation
def generate_response(prompt):
    # Load the text generation pipeline
    text_generation_pipeline = pipeline("text-generation", model="distilgpt2")

    # Generate response
    response = text_generation_pipeline(prompt, max_length=300, num_return_sequences=1, temperature=0.7)[0]['generated_text']

    # Return the generated response
    return response

# Function to correct and format prompts
def correct_prompt(prompt):
    # Load the model and tokenizer for grammatical correction
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

    # Encode the input text and generate correction
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_length=150, num_beams=2, early_stopping=True)
    
    # Decode the output text
    corrected_prompt = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return corrected_prompt

def get_suggestions_for_brand(brand, data):
    """
    Generates improvement suggestions for a brand using Hugging Face Transformers.

    Args:
        brand (str): The brand name.
        data (pd.DataFrame): The DataFrame containing review data.

    Returns:
        list: A list of improvement suggestions (strings).
    """

    suggestions = []

    # Analyze negative reviews
    negative_reviews = data[(data['brand'] == brand) & (data['sentiment'] == "Negative")]

    if len(negative_reviews) == 0:
        return ["No negative reviews found for the selected brand."]

    # Summarize negative review themes
    review_summary = " ".join(negative_reviews["body"].tolist())[:1024]  # Limit the input size

    # Correct the review summary
    corrected_summary = correct_prompt(review_summary)

    # Generate suggestions based on the corrected summary
    try:
        suggestion = generate_response(corrected_summary)
        suggestions.append(suggestion)
    except Exception as e:
        suggestions.append(f"Error generating suggestions: {str(e)}")

    return suggestions

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Sentiment Analysis using VADER",
    page_icon=":)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI Powered Brand Reputation Analysis")
st.markdown("------------------------------------------------------------------------------------")

filename = st.sidebar.file_uploader("Upload data:", type=("csv", "xlsx"))

if filename is not None:
    data = pd.read_csv(filename)
    data["body"] = data["body"].astype("str")
    data["score"] = data["body"].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)["compound"])
    data["sentiment"] = np.where(data['score'] >= .5, "Positive", "Negative")
    data = data[['brand', 'body', 'sentiment', 'score', 'date']]
    data['date'] = pd.to_datetime(data['date'])
    data['quarter'] = pd.PeriodIndex(data.date, freq='Q')

    per_dt = data.groupby(['brand', 'sentiment']).size().reset_index()
    per_dt = per_dt.sort_values(['sentiment'], ascending=False)
    per_dt1 = data.groupby(['brand']).size().reset_index()
    per_dt2 = pd.merge(per_dt, per_dt1, how='left', on='brand')
    per_dt2['Sentiment_Percentage'] = per_dt2['0_x'] / per_dt2['0_y']
    per_dt2 = per_dt2[['brand', 'sentiment', 'Sentiment_Percentage']]
    
    brand_c = data.groupby(['brand']).size().reset_index()
    st.sidebar.write("Reviews count according to the brand:")
    st.sidebar.write("Nokia   : " + str(brand_c.iloc[1, 1]))
    st.sidebar.write("HUAWEI  : " + str(brand_c.iloc[0, 1]))
    st.sidebar.write("Samsung : " + str(brand_c.iloc[2, 1]))

    st.subheader("Sentiment Distribution based on Phone Reviews")

    col3, col4 = st.columns(2)

    with col4:
        data1 = data[data['brand'] == 'Nokia']
        sentiment_count = data1.groupby(['sentiment'])['sentiment'].count()
        sentiment_count = pd.DataFrame({'Sentiments': sentiment_count.index, 'sentiment': sentiment_count.values})
        fig = px.pie(sentiment_count, values='sentiment', names='Sentiments', width=550,
                    height=400).update_layout(title_text='Nokia', title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        trend_dt = data[data['brand'] == 'Nokia']
        trend_dt['Review_Month'] = trend_dt['date'].dt.strftime('%m-%Y')
        trend_dt1 = trend_dt.groupby(['Review_Month', 'sentiment']).size().reset_index()
        trend_dt1 = trend_dt1.sort_values(['sentiment'], ascending=False)
        trend_dt1.rename(columns={0: 'Sentiment_Count'}, inplace=True)

        fig2 = px.line(trend_dt1, x="Review_Month", y="Sentiment_Count", color='sentiment', width=600,
                    height=400).update_layout(title_text='Trend analysis of Nokia', title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("------------------------------------------------------------------------------------")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(data, x="brand", y="sentiment",
                        histfunc="count", color="sentiment", facet_col="sentiment",
                        labels={"sentiment": "sentiment"}, width=550, height=400).update_layout(
            title_text='Distribution by count ', title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig1 = px.histogram(per_dt2, x="brand", y="Sentiment_Percentage", color="sentiment", facet_col="sentiment",
                        labels={"sentiment": "sentiment"},
                        width=550, height=400).update_layout(yaxis_title="Percentage",
                                                            title_text='Distribution by percentage',
                                                            title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)

    # Word Cloud for Reviews (Positive)
    st.subheader("Word Cloud for Reviews (Positive)")

    positive_reviews = data[data['sentiment'] == 'Positive']
    positive_text = " ".join(positive_reviews["body"])

    stopwords = set(STOPWORDS)
    stopwords.update(["phone", "the", "like", "is"])

    wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color="white").generate(positive_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, aspect="auto")
    ax.set_axis_off()
    st.pyplot(fig)

    st.subheader("Sample Positive Reviews")
    st.write(positive_reviews["body"].iloc[:3].tolist())  # Display first 3 positive reviews

    # Word Cloud for Reviews (Negative)
    st.subheader("Word Cloud for Reviews (Negative)")

    negative_reviews = data[data['sentiment'] == 'Negative']
    negative_text = " ".join(negative_reviews["body"])

    stopwords = set(STOPWORDS)
    stopwords.update(["phone", "the", "like", "is"])

    wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color="white").generate(negative_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, aspect="auto")
    ax.set_axis_off()
    st.pyplot(fig)

    st.subheader("Sample Negative Reviews")
    st.write(negative_reviews["body"].iloc[:3].tolist())  # Display first 3 negative reviews

    st.subheader("Brand Improvement Suggestions")
    brand_to_analyze = st.sidebar.selectbox("Select a brand for suggestions:", options=data['brand'].unique())
    if st.sidebar.button("Generate Suggestions"):
        suggestions = get_suggestions_for_brand(brand_to_analyze, data)
        st.write(suggestions)

else:
    st.write("Please upload a data file to proceed.")

# Streamlit app for general question answering
def main():
    st.title("General Question Answering")

    # Text input for prompt
    prompt = st.text_area("Ask your question:")

    # Button to generate response
    if st.button("Get Answer"):
        if prompt.strip() != "":
            # Correct and format the prompt
            corrected_prompt = correct_prompt(prompt)

            # Generate response
            response = generate_response(corrected_prompt)
            st.subheader("Answer:")
            st.write(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
