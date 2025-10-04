import streamlit as st
import sqlite3
import pandas as pd
import feedparser
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# RSS источники
RSS_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "http://rss.cnn.com/rss/money_news_international.rss",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EDJI,%5EGSPC,%5EIXIC&region=US&lang=en-US"
]

# База
def setup_database():
    conn = sqlite3.connect("news.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            link TEXT UNIQUE,
            source TEXT,
            published TEXT
        )
    """)
    conn.commit()
    conn.close()

# Сбор новостей
def fetch_news():
    articles = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        source = url.split("/")[2]
        for entry in feed.entries:
            articles.append({
                "title": entry.title[:200],
                "link": entry.link,
                "source": source,
                "published": getattr(entry, "published", datetime.now().strftime("%Y-%m-%d %H:%M"))
            })
    return pd.DataFrame(articles)

# Сохранение
def save_to_db(df):
    conn = sqlite3.connect("news.db")
    df.to_sql("news", conn, if_exists="append", index=False)
    conn.close()

# Дедупликация
def deduplicate(df):
    if df.empty:
        return df
    df["clean_title"] = df["title"].str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)
    df = df.drop_duplicates(subset=["clean_title"], keep="first")
    return df

# Hotness (очень простой)
def calculate_hotness(df):
    if df.empty:
        return df
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["clean_title"])
    similarity = cosine_similarity(X)
    df["hotness"] = similarity.mean(axis=1)  # простая метрика
    return df.sort_values("hotness", ascending=False).head(10)

# ---------------- STREAMLIT ----------------
st.title("📰 Финансовые новости")

setup_database()

if st.button("🚀 Обновить новости"):
    df = fetch_news()
    save_to_db(df)
    st.success(f"Собрано {len(df)} новостей")

    st.subheader("📌 Последние новости")
    st.dataframe(df[["title","source","published"]])

    st.subheader("🔧 Дедупликация")
    df_dedup = deduplicate(df)
    st.dataframe(df_dedup[["title","source"]])

    st.subheader("🔥 Hotness (топ-10)")
    hot = calculate_hotness(df_dedup)
    st.dataframe(hot[["title","hotness"]])
