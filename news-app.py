# news_pipeline.py
import sqlite3
import requests
import time
import re
import pandas as pd
from datetime import datetime
from tabulate import tabulate

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

# ======================================================
# 1. СБОР ФИНАНСОВЫХ НОВОСТЕЙ
# ======================================================

RSS_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",   # CNBC Business
    "http://rss.cnn.com/rss/money_news_international.rss",      # CNN
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EDJI,%5EGSPC,%5EIXIC&region=US&lang=en-US",  # Yahoo
]

def setup_database():
    conn = sqlite3.connect('news.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS news')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT UNIQUE,
            source TEXT,
            published_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_http_headers():
    return {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/rss+xml,application/xml,text/xml',
    }

def fetch_rss_news():
    all_articles = []
    for rss_url in RSS_FEEDS:
        try:
            headers = get_http_headers()
            response = requests.get(rss_url, headers=headers, timeout=15)
            if response.status_code != 200:
                continue
            content = response.text
            items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
            for item in items:
                title_match = re.search(r'<title>(.*?)</title>', item, re.DOTALL)
                link_match = re.search(r'<link>(.*?)</link>', item, re.DOTALL)
                date_match = re.search(r'<pubDate>(.*?)</pubDate>', item, re.DOTALL)
                title = title_match.group(1).strip() if title_match else None
                url = link_match.group(1).strip() if link_match else None
                pub_date = date_match.group(1).strip() if date_match else None
                if title and url:
                    source = "CNBC" if "cnbc" in rss_url else \
                             "CNN" if "cnn" in rss_url else \
                             "Yahoo" if "yahoo" in rss_url else "Other"
                    all_articles.append({
                        'title': title[:150],
                        'url': url,
                        'source': source,
                        'published_at': pub_date or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            time.sleep(1)
        except:
            continue
    return all_articles

def save_articles(articles):
    conn = sqlite3.connect('news.db')
    cursor = conn.cursor()
    saved = 0
    for article in articles:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO news (title, url, source, published_at)
                VALUES (?, ?, ?, ?)
            ''', (article['title'], article['url'], article['source'], article['published_at']))
            if cursor.rowcount > 0:
                saved += 1
        except:
            continue
    conn.commit()
    conn.close()
    return saved

# ======================================================
# 2. ДЕДУПЛИКАЦИЯ / КЛАСТЕРИЗАЦИЯ
# ======================================================

def preprocess_and_deduplicate():
    conn = sqlite3.connect('news.db')
    df = pd.read_sql_query("SELECT * FROM news", conn)
    if df.empty:
        print("Нет данных")
        return

    df['clean_title'] = df['title'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.lower()

    df = df.drop_duplicates(subset=['clean_title'], keep='first')

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    title_vectors = vectorizer.fit_transform(df['clean_title'].fillna(''))
    cosine_dist = 1 - cosine_similarity(title_vectors)
dbscan = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
    clusters = dbscan.fit_predict(cosine_dist)
    df['cluster_id'] = clusters

    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS deduplicated_news')
    cursor.execute('''
        CREATE TABLE deduplicated_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id INTEGER,
            title TEXT,
            url TEXT,
            source TEXT,
            published_at TEXT,
            clean_title TEXT,
            cluster_id INTEGER
        )
    ''')
    for _, row in df.iterrows():
        cursor.execute('''
            INSERT INTO deduplicated_news
            (original_id, title, url, source, published_at, clean_title, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (row['id'], row['title'], row['url'], row['source'], row['published_at'], row['clean_title'], row['cluster_id']))
    conn.commit()
    conn.close()

# ======================================================
# 3. HOTNESS SCORING
# ======================================================

def calculate_hotness_scores():
    conn = sqlite3.connect('news.db')
    df = pd.read_sql_query("SELECT * FROM deduplicated_news", conn)
    if df.empty:
        print("Нет данных для hotness")
        return

    source_weights = {'Reuters': 1.0, 'CNBC': 0.9, 'Bloomberg': 0.9, 'Financial Times': 0.9, 'CNN': 0.8, 'Yahoo': 0.7, 'Other': 0.5}
    surprise_keywords = ['surge','plunge','crisis','deal','crash','rally','record','high','low']

    results = []
    for cluster_id in df['cluster_id'].unique():
        cluster_articles = df[df['cluster_id'] == cluster_id]
        if cluster_articles.empty:
            continue
        hotness = min(len(cluster_articles)/10, 1.0) * 0.4
        hotness += max(source_weights.get(s,0.5) for s in cluster_articles['source']) * 0.3
        hotness += any(any(k in str(t).lower() for k in surprise_keywords) for t in cluster_articles['title']) * 0.2
        hotness += min(cluster_articles['source'].nunique()/5, 1.0) * 0.1
        results.append((cluster_id, hotness, cluster_articles.iloc[0]['title']))

    hot_df = pd.DataFrame(results, columns=['cluster_id','hotness','title']).sort_values('hotness', ascending=False).head(10)
    print(tabulate(hot_df, headers='keys', tablefmt='grid', showindex=True))
    conn.close()

# ======================================================
# MAIN PIPELINE
# ======================================================

if name == "main":
    print("🔄 Сбор новостей...")
    setup_database()
    articles = fetch_rss_news()
    saved = save_articles(articles)
    print(f"💾 Сохранено {saved} статей")

    print("\n🔧 Дедупликация и кластеризация...")
    preprocess_and_deduplicate()

    print("\n🔥 Hotness ranking...")
    calculate_hotness_scores()
