from flask import Flask, render_template, request, jsonify, session
import os
import re
import requests
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FLASK_SECRET = os.getenv("FLASK_SECRET")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", 8))

# CONSTANTS
WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
NEWS_API_URL = "https://newsapi.org/v2/everything"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# THRESHOLDS
WIKI_SIM_THRESHOLD = 0.25
NEWS_SIM_THRESHOLD = 0.15

# INITIALIZATION
app = Flask(__name__)
app.secret_key = FLASK_SECRET
genai.configure(api_key=GEMINI_API_KEY)

# CLEAN TEXT
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s\.,'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# NEWS VALIDATION
def is_valid_news(text):
    text = text.lower().strip()

    if len(text) < 20:
        return False, "Too short. Please enter a full sentence."

    casual = ["hi", "hello", "hey", "ok", "bro", "hlo", "what's up"]
    if text in casual:
        return False, "Casual greeting detected."

    if re.search(r"[😀-🙏🔥❤️-🧿]", text):
        return False, "Contains emojis — invalid news."

   # verbs = ["is", "was", "were", "reported", "said", "confirmed",
    #         "died", "killed", "arrested", "launched", "announced"]

   # if not any(v in text for v in verbs):
    #    return False, "No real news-like action detected."

    return True, "Valid news input."
    
# TF-IDF similarity check
def similarity(a, b):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([a, b])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return float(sim)

# WIKIPEDIA
def wiki_verify(query):
    try:
        r = requests.get(WIKI_SEARCH_URL, params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3
        }, timeout=8)

        data = r.json()
        results = data.get("query", {}).get("search", [])

        if not results:
            return False, None, 0.0

        best_score = 0
        best_title = None

        for item in results:
            title = item["title"]
            summary_resp = requests.get(
                WIKI_SUMMARY_URL.format(title.replace(" ", "_")), timeout=5
            )
            if summary_resp.status_code != 200:
                continue

            summary = summary_resp.json().get("extract", "")
            if not summary:
                continue

            sim = similarity(query, summary)
            if sim > best_score:
                best_score = sim
                best_title = title

        return best_score > WIKI_SIM_THRESHOLD, best_title, best_score

    except Exception:
        return False, None, 0.0
# NEWS API
def news_verify(text):
    try:
        r = requests.get(NEWS_API_URL, params={
            "q": text,
            "language": "en",
            "pageSize": 5,
            "apiKey": NEWS_API_KEY
        }, timeout=8)

        data = r.json()
        articles = data.get("articles", [])

        if not articles:
            return False, None, 0.0

        best_score = 0
        best_article = None

        for a in articles:
            content = f"{a['title']} {a.get('description','')}"
            sim = similarity(text, content)
            if sim > best_score:
                best_score = sim
                best_article = a

        return best_score > NEWS_SIM_THRESHOLD, best_article, best_score

    except Exception:
        return False, None, 0.0

# GEMINI CONTEXTUAL REASONING
def gemini_context_analysis(text):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
You are a professional, real-time fact-checking Al. Your priority is to determine the absolute CURRENT veracity of the statement.
Search for and prioritize the **most recent** and **authoritative primary sources** (like official spokespersons, family, or global news agencies) over prior rumors or hoaxes.
Given the statement:
\"{text}\"
Determine its factual accuracy based on global context, knowledge, and logic.
Respond strictly in one of the following formats:

REAL - <short reason>
FAKE - <short reason>
UNCERTAIN - <short reason>
"""
        response = model.generate_content(
            prompt,
            request_options={"timeout": GEMINI_TIMEOUT}
        )
        return response.text.strip()
    except Exception as e:
        print("Gemini API error:", e)
        return "UNCERTAIN - Gemini reasoning failed"

# GOOGLE FACT CHECK
def factcheck_verify(text):
    try:
        params = {"query": text, "key": GOOGLE_FACTCHECK_API_KEY}
        r = requests.get(FACTCHECK_URL, params=params, timeout=8)
        data = r.json()

        claims = data.get("claims", [])
        if not claims:
            return False, None, None

        claim = claims[0]
        rating = claim.get("claimReview", [{}])[0].get("textualRating", "")
        publisher = claim.get("claimReview", [{}])[0].get("publisher", {}).get("name", "")

        return True, rating, publisher

    except Exception:
        return False, None, None

# ROUTES
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = clean_text(data.get("text", "").strip())

    # Validation
    ok, reason = is_valid_news(user_text)
    if not ok:
        return jsonify({
            "prediction": "INVALID INPUT",
            "confidence": 0,
            "api_match": reason
        })

    print(f"\n🧠 Analyzing: {user_text}\n{'-'*60}")

    # Initial Defaults
    prediction, confidence, api_match = "Uncertain", 60.0, "No strong match found"

    # 1. Wikipedia Verification
    w_ok, w_title, w_sim = wiki_verify(user_text)
    if w_ok:
        prediction = "Real"
        confidence = w_sim * 100
        api_match = f"Wikipedia: {w_title}"

    # 2. NewsAPI Verification
    elif True:
        n_ok, article, n_sim = news_verify(user_text)
        if n_ok:
            prediction = "Real"
            confidence = n_sim * 100
            api_match = f"{article['source']['name']} - {article['title']}"

        # 3. Gemini Context Analysis
        else:
            gem = gemini_context_analysis(user_text)
            api_match = gem

            if gem.startswith("REAL"):
                prediction = "Real"
                confidence = 85
            elif gem.startswith("FAKE"):
                prediction = "Fake"
                confidence = 85
            else:
                prediction = "Uncertain"
                confidence = 60

      # 4. Google Fact Check
    fc_ok, rating, publisher = factcheck_verify(user_text)
    if fc_ok and rating:  # Only override if rating exists
        if "true" in rating.lower():
            prediction = "Real"
            confidence = 95.0
            api_match = f"Google Fact Check ({publisher}: {rating})"

        elif "false" in rating.lower():
            prediction = "Fake"
            confidence = 95.0
            api_match = f"Google Fact Check ({publisher}: {rating})"

        else:
            prediction = "Uncertain"
            confidence = 70.0
            api_match = f"Google Fact Check ({publisher}: {rating})"

    # Save to History
    history = session.get("history", [])
    history.insert(0, {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "text": user_text,
        "prediction": prediction,
        "confidence": confidence,
        "match": api_match
    })
    session["history"] = history[:30]
    session.modified = True

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "api_match": api_match
    })

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("history", None)
    return ("", 204)

if __name__ == "__main__":
    app.run()
