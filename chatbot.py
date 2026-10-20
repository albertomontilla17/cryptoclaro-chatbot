"""
CryptoClaro Bot - a rule-free, machine-learning intent chatbot for Crypto 101.

How it works:
  1. NLTK cleans the text: lowercase -> tokenize -> remove stop words -> stem.
  2. scikit-learn turns the cleaned text into numbers (TfidfVectorizer).
  3. A Naive Bayes classifier (MultinomialNB) predicts the user's intent.
  4. If the model isn't confident enough, the bot says it doesn't know.
  5. Coin detection (keyword entity extraction) spots Bitcoin, Ethereum,
     Tether, BNB or Solana in the message.
  6. NLTK's VADER scores the mood of the message (positive / neutral / negative).
  7. The intent picks the answer, the coin adds specifics, the mood adjusts the tone.

No AI service or internet API is used to answer. Everything runs locally.

Run in the terminal:   python chatbot.py
Run the web interface: python app.py
"""

import json
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download required NLTK resources (only the first time)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

BOT_NAME = "CryptoClaro Bot"

# Minimum probability the model needs before we trust its prediction.
CONFIDENCE_THRESHOLD = 0.15

DATA_DIR = Path(__file__).resolve().parent


def _load_json(name):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# 1. Training data: [example phrase, intent]
training_data = _load_json("training_data.json")

# 2. Coins the bot knows (used by coin detection in step 6)
# Each coin has the names people use for it. We look for them in the
# original words. ("sol" is left out on purpose: it's also Peru's currency.)
COINS = _load_json("coins.json")


# Every single-word coin name. The intent model ignores these, because
# coin detection handles them (otherwise "bitcoin" would drag every
# question toward one intent).
COIN_WORDS = {a for coin in COINS.values() for a in coin["aliases"] if " " not in a}


# 3. Text preprocessing (NLTK)
stemmer = PorterStemmer()

# Keep "you" so phrases like "what can you do" aren't erased completely.
# Add "vs" so "bitcoin vs ethereum" isn't confused with "hot wallet vs cold wallet".
STOP_WORDS = (set(stopwords.words("english")) - {"you"}) | {"vs", "versus"}


def preprocess_text(text):
    """Tokenize, drop punctuation, stop words and coin names, then stem each word."""
    tokens = word_tokenize(text.lower())
    words = [t for t in tokens
             if t.isalnum() and t not in STOP_WORDS and t not in COIN_WORDS]
    return " ".join(stemmer.stem(w) for w in words)


# 4. Train the model.
# Training phrases go through the SAME cleaning as user input,
# so the model learns exactly the words it will see later.
X_train = [preprocess_text(text) for text, _ in training_data]
y_train = [intent for _, intent in training_data]

# TF-IDF turns text into numbers and gives less weight to words that appear
# in many intents (like "crypto"), so the meaningful words decide the answer.
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train_vectorized, y_train)


# 5. Sentiment analysis (NLTK VADER)
# VADER is a ready-made sentiment model built for short, informal text.
# It reads the ORIGINAL message, because it uses words like "not",
# punctuation ("!!!") and CAPITALS to judge how strong the feeling is.
sentiment_analyzer = SentimentIntensityAnalyzer()

# Words VADER doesn't know that carry strong feeling in a crypto chat
sentiment_analyzer.lexicon.update({
    "scammed": -3.0, "hacked": -3.0, "stolen": -3.0,
    "fraud": -3.0, "rugged": -2.5, "losing": -2.0, "lost": -2.0,
    # In crypto questions these words are topics, not feelings
    # ("Is USDT safe?" isn't a happy message), so they count as neutral.
    "scam": 0.0, "scams": 0.0, "safe": 0.0, "safely": 0.0, "save": 0.0, "secure": 0.0, "protect": 0.0,
    "interest": 0.0, "worth": 0.0, "want": 0.0,
})


def analyze_sentiment(text):
    """Return the mood label and VADER's compound score (-1 to +1)."""
    score = sentiment_analyzer.polarity_scores(text)["compound"]
    if score >= 0.3:
        label = "positive"
    elif score <= -0.3:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": score}


# 6. Coin detection (keyword-based entity extraction)
def detect_coins(text):
    """Return the coins mentioned in the message, in the order they appear."""
    lowered = text.lower()
    words = word_tokenize(lowered)
    found = []
    for word in words:
        for key, coin in COINS.items():
            if key not in found and word in coin["aliases"]:
                found.append(key)
    # Two-word names like "binance coin"
    for key, coin in COINS.items():
        if key not in found and any(" " in a and a in lowered for a in coin["aliases"]):
            found.append(key)
    return found


def analyze(text):
    """Return the full prediction details (used by the web interface)."""
    sentiment = analyze_sentiment(text)
    coins = detect_coins(text)
    clean_text = preprocess_text(text)
    input_vector = vectorizer.transform([clean_text])

    # Words the model has never seen before are ignored by the vectorizer
    known_words = [w for w in clean_text.split() if w in vectorizer.vocabulary_]

    # Nothing recognizable -> unknown
    if input_vector.nnz == 0:
        return {"intent": "unknown", "confidence": 0.0, "top": [], "sentiment": sentiment,
                "coins": coins, "clean_text": clean_text, "known_words": known_words}

    probabilities = classifier.predict_proba(input_vector)[0]
    ranked = probabilities.argsort()[::-1]
    top = [(classifier.classes_[i], float(probabilities[i])) for i in ranked[:3]]
    intent, confidence = top[0]

    # Not confident enough -> unknown
    if confidence < CONFIDENCE_THRESHOLD:
        intent = "unknown"

    return {"intent": intent, "confidence": confidence, "top": top, "sentiment": sentiment,
            "coins": coins, "clean_text": clean_text, "known_words": known_words}


def identify_intent_ml(text):
    """Predicts the user's intent with the trained Naive Bayes model."""
    return analyze(text)["intent"]


DISCLAIMER = " (Disclaimer: This chatbot is educational only. This is not financial advice.)"

responses = _load_json("responses.json")
responses["greeting"] = responses["greeting"].format(bot_name=BOT_NAME)

# Intents where the bot explains a crypto topic
TOPIC_INTENTS = {"crypto_basics", "crypto_value", "crypto_wallets", "crypto_safety",
                 "investing_trading", "web3_nfts", "everyday_use", "inflation_savings", "defi"}

for intent in TOPIC_INTENTS:
    responses[intent] += DISCLAIMER


def generate_response(intent, sentiment_label="neutral", coins=()):
    """Builds the answer: the intent picks it, coins add specifics, the mood sets the tone."""
    coins = list(coins)

    if coins and intent in ("crypto_basics", "unknown"):
        # A question about the coin itself -> its profile (or a comparison of several)
        if len(coins) == 1:
            answer = COINS[coins[0]]["profile"] + DISCLAIMER
        else:
            answer = "Here's a quick comparison: " + " ".join(COINS[c]["note"] for c in coins) + DISCLAIMER
        is_topic = True
    else:
        answer = responses.get(intent, responses["unknown"])
        is_topic = intent in TOPIC_INTENTS
        # A topic question that mentions a coin -> add a line about that coin
        # (for safety questions, that coin's main risk)
        if coins and is_topic:
            if intent == "crypto_safety":
                notes = " ".join(f"Specific to {COINS[c]['name']}: "
                                 + COINS[c]["profile"].split("Main risk: ")[1] for c in coins)
            else:
                notes = " ".join(COINS[c]["note"] for c in coins)
            if answer.endswith(DISCLAIMER):
                answer = answer[: -len(DISCLAIMER)] + " " + notes + DISCLAIMER
            else:
                answer += " " + notes

    if sentiment_label == "negative":
        if intent == "bye":
            return "Sorry I couldn't be more helpful. Goodbye, and stay safe out there!"
        if not is_topic and intent in ("greeting", "unknown", "help"):
            return ("Sorry this has been frustrating. I only know crypto basics, so try "
                    "asking about wallets, scams, stablecoins, or type 'help' to see every topic.")
        if is_topic:
            answer = "I'm sorry you're dealing with that. " + answer
    elif sentiment_label == "positive" and is_topic:
        answer = "Love the enthusiasm! " + answer

    return answer


# 7. Terminal chatbot loop (only runs with: python chatbot.py)
if __name__ == "__main__":
    print(f"Welcome to {BOT_NAME}! (Type 'bye' to exit)")
    while True:
        user_input = input("You: ")
        details = analyze(user_input)
        intent = details["intent"]
        print("Bot:", generate_response(intent, details["sentiment"]["label"], details["coins"]))
        if intent == "bye":
            break
