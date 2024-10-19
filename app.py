from flask import Flask, render_template, request
import os
import re
from collections import defaultdict

app = Flask(__name__)

# Preprocessing function
def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())

# Load documents
def load_documents(folder_path):
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                docs[filename] = preprocess(file.read())
    return docs

# Load queries
def load_queries(query_file_path):
    with open(query_file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Compute term frequencies and document frequencies
def compute_statistics(docs):
    doc_count = len(docs)
    term_doc_freq = defaultdict(int)
    term_freq = defaultdict(lambda: defaultdict(int))

    for doc_id, words in docs.items():
        word_set = set(words)
        for word in words:
            term_freq[doc_id][word] += 1
        for word in word_set:
            term_doc_freq[word] += 1

    return term_freq, term_doc_freq, doc_count

# Compute relevance probabilities using BIM
def compute_relevance_prob(query, term_freq, term_doc_freq, doc_count):
    scores = {}
    for doc_id in term_freq:
        score = 1.0
        for term in query:
            tf = term_freq[doc_id].get(term, 0)
            df = term_doc_freq.get(term, 0)
            p_term_given_relevant = (tf + 1) / (sum(term_freq[doc_id].values()) + len(term_doc_freq))
            p_term_given_not_relevant = (df + 1) / (doc_count - df + len(term_doc_freq))
            score *= (p_term_given_relevant / p_term_given_not_relevant)
        scores[doc_id] = score
    return scores

# Route for homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        query = request.form['query']
        folder_path = 'D:/Ubuntu/final tech 400 data folder-20241019T134906Z-001'
        documents = load_documents(folder_path)

        term_freq, term_doc_freq, doc_count = compute_statistics(documents)

        query_terms = preprocess(query)
        scores = compute_relevance_prob(query_terms, term_freq, term_doc_freq, doc_count)
        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = [(doc_id, score) for doc_id, score in ranked_docs if score > 0]  # Filter out zero scores

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
