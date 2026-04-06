stopwords = [
    "і","та","на","що","до","з","за","у","в","не","це","як","для","по", "про","від","але","або","чи","також","його","її","вони","ми",
    "нa", "щo", "тa", "нe", "зa", 'цe', "пpo", "дo", "щօ", 'але', 'по', "він","які","року","під","час","із","після","буде","щоб","aлe",
    "йoгo","тaкoж","пo","вжe","чepeз","вoни","щe","щодо", "можна","був","пам", "про", "дօ", 'чac', 'бyдe', 'тaм', 'тaк', 'дyжe', 'зapaз',
    'хто', 'что','это','єр','the', 'ви', 'бо', 'тут', 'тоді', 'ти', 'нато', 'єс', 'путін'
]

tfidf = TfidfVectorizer(
    max_df=0.05,
    min_df=3,
    stop_words=stopwords
)

count = CountVectorizer(
    max_df=0.05,
    min_df=3,
    stop_words=stopwords
)

X_tfidf = tfidf.fit_transform(texts)
X_count = count.fit_transform(texts)

print("TF-IDF shape:", X_tfidf.shape)
print("Count shape:", X_count.shape)


def get_top_words(model, feature_names, n_top=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top - 1:-1]]
        topics.append(top_features)
    return topics


def print_topics(topics):
    for i, words in enumerate(topics):
        print(f"Topic {i+1}: {', '.join(words)}")


lsa_3 = TruncatedSVD(n_components=3, random_state=42)
lsa_5 = TruncatedSVD(n_components=5, random_state=42)

lsa3_topics = lsa_3.fit_transform(X_tfidf)
lsa5_topics = lsa_5.fit_transform(X_tfidf)

lsa3_words = get_top_words(lsa_3, tfidf.get_feature_names_out())
lsa5_words = get_top_words(lsa_5, tfidf.get_feature_names_out())

print("LSA k=3")
print_topics(lsa3_words)

print("\nLSA k=5")
print_topics(lsa5_words)


lda_3 = LatentDirichletAllocation(n_components=3, random_state=42)
lda_5 = LatentDirichletAllocation(n_components=5, random_state=42)

lda3_topics = lda_3.fit_transform(X_count)
lda5_topics = lda_5.fit_transform(X_count)

lda3_words = get_top_words(lda_3, count.get_feature_names_out())
lda5_words = get_top_words(lda_5, count.get_feature_names_out())

print("\nLDA k=3")
print_topics(lda3_words)

print("LDA k=5")
print_topics(lda5_words)


def get_top_docs(topic_matrix, texts, top_n=3):
    top_docs = {}

    for topic_idx in range(topic_matrix.shape[1]):
        topic_scores = topic_matrix[:, topic_idx]
        top_indices = topic_scores.argsort()[-top_n:][::-1]

        top_docs[topic_idx] = [texts[i] for i in top_indices]

    return top_docs


lsa5_docs = get_top_docs(lsa5_topics, texts)
lda5_docs = get_top_docs(lda5_topics, texts)


for topic, docs in lsa5_docs.items():
    print(f"\nLSA Topic {topic+1}:")
    for d in docs:
        print("-", d[:200])

for topic, docs in lda5_docs.items():
    print(f"\nLDA Topic {topic+1}:")
    for d in docs:
        print("-", d[:200])