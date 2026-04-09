# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import emoji
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

st.title("📱 WhatsApp Chat Analyzer")

# ---------------------------
# 1️⃣ Upload chat
# ---------------------------
uploaded_file = st.file_uploader("Upload your WhatsApp chat (.txt)", type=["txt"])
if uploaded_file:
    chat = uploaded_file.read().decode("utf-8")

    st.subheader("Preview")
    st.text_area("Chat Preview (first 1000 chars)", chat[:1000])

    # ---------------------------
    # 2️⃣ Preprocess
    # ---------------------------
    def preprocess(chat_text):
        # Regex pattern for date, time, author, message
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?[APMapm]{2})\s-\s'
        parts = re.split(pattern, chat_text)[1:]  # split by date pattern

        data = []
        for i in range(0, len(parts), 3):
            date, time, message = parts[i], parts[i+1], parts[i+2]
            data.append([f"{date} {time}", message])

        df = pd.DataFrame(data, columns=['datetime', 'message'])
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)
        return df

    df = preprocess(chat)
    st.success("✅ Preprocessing done")

    # ---------------------------
    # 3️⃣ Active users
    # ---------------------------
    df['author'] = df['message'].apply(lambda x: x.split(':')[0] if ':' in x else "Unknown")
    active_users = df['author'].value_counts().head(10)
    st.subheader("Most Active Users")
    st.bar_chart(active_users)

    # ---------------------------
    # 4️⃣ Sentiment analysis
    # ---------------------------
    st.subheader("Sentiment Analysis")
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['message'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    fig, ax = plt.subplots()
    sns.histplot(df['sentiment'], bins=20, kde=True, color='green', ax=ax)
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

    # ---------------------------
    # 5️⃣ Word Cloud
    # ---------------------------
    st.subheader("Word Cloud")
    all_text = " ".join(df['message'].dropna().tolist()).strip()
    if len(all_text) == 0:
        st.warning("⚠ No text available to generate Word Cloud.")
    else:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig, ax = plt.subplots(figsize=(15,7))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # ---------------------------
    # 6️⃣ Emoji Analysis
    # ---------------------------
    st.subheader("Emoji Analysis")
    def extract_emojis(s):
        return [c for c in s if c in emoji.EMOJI_DATA]

    df['emojis'] = df['message'].apply(extract_emojis)
    all_emojis = sum(df['emojis'], [])
    if len(all_emojis) == 0:
        st.warning("⚠ No emojis found in chat.")
    else:
        emoji_counts = Counter(all_emojis).most_common(10)
        st.write("Most common emojis:", emoji_counts)
        emoji_df = pd.DataFrame(emoji_counts, columns=['Emoji', 'Count'])
        st.bar_chart(emoji_df.set_index('Emoji'))

    # ---------------------------
    # 7️⃣ Topic Modeling (LDA)
    # ---------------------------
    st.subheader("Topic Modeling (LDA)")
    text_messages = df[
        (~df['message'].str.contains('<Media omitted>')) &
        (df['message'].str.strip() != '')
    ]['message']

    if text_messages.shape[0] < 3:
        st.warning("⚠ Not enough text messages for LDA topic modeling.")
    else:
        try:
            cv = CountVectorizer(max_df=0.95, min_df=1, stop_words=None)
            X = cv.fit_transform(text_messages)
            n_topics = min(3, text_messages.shape[0])
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(X)
            topics = []
            for i, topic in enumerate(lda.components_):
                top_words = [cv.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
                topics.append(f"Topic {i+1}: {top_words}")
            st.write(topics)
        except ValueError as e:
            st.warning(f"⚠ Topic modeling skipped: {e}")

    # ---------------------------
    # 8️⃣ Clustering (KMeans)
    # ---------------------------
    st.subheader("Message Clustering (KMeans)")
    try:
        tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        text_messages_list = text_messages.tolist()
        if len(text_messages_list) < 1:
            st.warning("⚠ Not enough text for clustering.")
        else:
            X = tfidf.fit_transform(text_messages_list)
            kmeans = KMeans(n_clusters=min(3, len(text_messages_list)), random_state=42)
            clusters = kmeans.fit_predict(X)
            df_cluster = pd.DataFrame({'message': text_messages_list, 'cluster': clusters})
            st.write(df_cluster.head(10))
    except ValueError as e:
        st.warning(f"⚠ Clustering skipped: {e}")
