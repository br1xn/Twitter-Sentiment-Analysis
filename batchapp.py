import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# pre-trained model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') 

# sentiment analyze
def analyze_sentiment(text):
    tokens = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        result = model(tokens)
    logits = result.logits
    probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()
    sentiment_score = int(torch.argmax(logits)) + 1
    sentiment_map = {
        1: "Very Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Very Positive"
    }
    sentiment_label = sentiment_map.get(sentiment_score, "Unknown")
    return sentiment_label, probabilities

# Sentiment color insert
def get_sentiment_color(sentiment):
    color_map = {
        'Very Negative': 'red',
        'Negative': 'orange',
        'Neutral': 'yellow',
        'Positive': 'lightgreen',
        'Very Positive': 'green'
    }
    return color_map.get(sentiment, 'white')

# Streamlit app
def main():
    st.title("⚖️ Batch Sentiment Analysis")

    # File Upload
    uploaded_file = st.file_uploader("🗂️ Upload a CSV file", type=["csv", "xlsx"])

    if uploaded_file is not None:

        # File format
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if not df.empty:
            st.write(f"Uploaded File Preview ({len(df)} rows):")
            st.write(df.head())

            if 'Sentence' in df.columns:
                if st.button("⚙️ Start Analyzing"):
                    st.write("Analyzing sentiment for each row...")

                    # progress bar
                    progress_bar = st.progress(0)
                    total_rows = len(df)

                    # Initialize new column for sentiment results
                    df['Sentiment'] = ""

                    for index, row in df.iterrows():
                        sentiment, probabilities = analyze_sentiment(row['Sentence'])
                        df.at[index, 'Sentiment'] = sentiment
                        # Update progress bar
                        progress_bar.progress((index + 1) / total_rows)

                    st.success("✅ Sentiment analysis completed.")

                    df['Sentiment Color'] = df['Sentiment'].apply(get_sentiment_color)

                    # Display the DataFrame with colored backgrounds
                    st.write("📊 Sentiment Analysis Results:")

                    selected_columns = df.iloc[:, :2].copy()
                    selected_columns['Sentiment'] = df['Sentiment']
                    

                    styled_df = selected_columns.style.apply(lambda x: [f'background-color: {color}' for color in df['Sentiment Color']], subset=['Sentiment'])
                    st.dataframe(styled_df)


                    # Distribution of polarity
                    sentiment_counts = df['Sentiment'].value_counts()
                    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))

                    # Bar chart
                    sentiment_counts.plot(kind='bar', color=['green', 'lightgreen', 'yellow', 'orange', 'red'], ax=ax1)
                    ax1.set_title('Sentiment Distribution', color='white')
                    ax1.set_xlabel('Sentiment Analysis', color='white')
                    ax1.set_ylabel('Sentiment Count', color='white')
                    ax1.tick_params(axis='x', rotation=45) 

                    # Adding count numbers on top of each bar
                    for index, value in enumerate(sentiment_counts):
                        ax1.text(index, value, str(value), color='white', ha='center', va='bottom')  # ha: horizontal alignment, va: vertical alignment

                    # Set the color of the x-ticks (sentiment categories) to white
                    ax1.set_xticklabels(sentiment_counts.index, color='white')
                    
                    # Pie Chart
                    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=['green', 'lightgreen', 'yellow', 'orange', 'red'])
                    ax2.set_ylabel('', color='white')  # Remove the y-label for pie chart
                    ax2.set_title('Sentiment Distribution', color='white')
                    for i, label in enumerate(ax2.texts):
                        label.set_color('white') 
                        
                    # Make background transparent
                    fig.patch.set_alpha(0.0)  # Make the figure background transparent
                    ax1.set_facecolor('none')  # Make the bar chart background transparent
                    ax2.set_facecolor('none')  # Make the pie chart background transparent

                    # Adjust layout for better spacing
                    plt.tight_layout()

                    st.pyplot(fig)

                    # Downloading new file
                    output_csv = df[['Sentence', 'Sentiment']].to_csv(index=False)

                    # Download link
                    st.download_button(label="📥 Download CSV with Sentiments", data=output_csv, file_name="sentiment_results.csv", mime="text/csv")
                else:
                    st.warning("Click the button to start analyzing.")
            else:
                st.error("Uploaded file does not contain a 'Sentence' column.")
        else:
            st.error("Uploaded file is empty.")

if __name__ == "__main__":
    main()
