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

# 1. Training data: (example phrase, intent)
training_data = [
    # Greetings
    ("hello", "greeting"), ("hi there", "greeting"), ("hey", "greeting"),
    ("good morning", "greeting"), ("good afternoon", "greeting"), ("good evening", "greeting"),
    ("yo", "greeting"), ("greetings", "greeting"), ("is anyone there", "greeting"),
    ("hello bot", "greeting"), ("howdy", "greeting"),

    # Help / Capabilities
    ("help me", "help"), ("what can you do", "help"), ("i need assistance", "help"),
    ("instructions", "help"), ("how do i use this", "help"), ("commands", "help"),
    ("what topics do you know", "help"), ("list categories", "help"), ("menu", "help"),
    ("what can i ask", "help"), ("show me the topics", "help"), ("options", "help"),

    # Thanks
    ("thanks", "thanks"), ("thank you", "thanks"), ("thank you so much", "thanks"),
    ("that helped a lot", "thanks"), ("great answer", "thanks"), ("awesome thanks", "thanks"),
    ("appreciate it", "thanks"), ("cool", "thanks"), ("thanks for the info", "thanks"),

    # Bye / Exit
    ("bye", "bye"), ("exit", "bye"), ("quit", "bye"),
    ("goodbye", "bye"), ("see you later", "bye"), ("farewell", "bye"),
    ("i am done", "bye"), ("stop", "bye"), ("close", "bye"), ("hasta la vista baby", "bye"),

    # Crypto 101: What is crypto / blockchain
    # (Coin names like "bitcoin" are handled by coin detection, not by this model.)
    ("what is cryptocurrency", "crypto_basics"), ("what is crypto", "crypto_basics"),
    ("what is a blockchain", "crypto_basics"), ("how does blockchain work", "crypto_basics"),
    ("tell me about crypto", "crypto_basics"), ("who created it", "crypto_basics"),
    ("crypto for beginners", "crypto_basics"), ("what is mining", "crypto_basics"),
    ("crypto", "crypto_basics"), ("explain crypto to me", "crypto_basics"),
    ("what is a coin", "crypto_basics"), ("compare coins", "crypto_basics"),

    # Crypto 101: Value
    ("why does crypto have value", "crypto_value"), ("what gives crypto value", "crypto_value"),
    ("is cryptocurrency real money", "crypto_value"), ("explain crypto value", "crypto_value"),
    ("why is bitcoin worth anything", "crypto_value"), ("crypto vs fiat currency", "crypto_value"),
    ("why is bitcoin so expensive", "crypto_value"), ("what determines the price of crypto", "crypto_value"),
    ("why does the price go up and down", "crypto_value"), ("why is crypto valuable", "crypto_value"),
    ("is bitcoin worth it", "crypto_value"),

    # Crypto 101: Wallets
    ("how do crypto wallets work", "crypto_wallets"), ("what is a private key", "crypto_wallets"),
    ("public vs private keys", "crypto_wallets"), ("set up a crypto wallet", "crypto_wallets"),
    ("hot wallet vs cold wallet", "crypto_wallets"), ("how to store cryptocurrency", "crypto_wallets"),
    ("what is a seed phrase", "crypto_wallets"), ("where do i keep my bitcoin", "crypto_wallets"),
    ("what is a hardware wallet", "crypto_wallets"), ("what is a wallet address", "crypto_wallets"),

    # Crypto 101: Safety & Scams
    ("how to avoid crypto scams", "crypto_safety"), ("crypto phishing protection", "crypto_safety"),
    ("is crypto safe", "crypto_safety"), ("how to secure my crypto", "crypto_safety"),
    ("spotting fake crypto sites", "crypto_safety"), ("common cryptocurrency scams", "crypto_safety"),
    ("someone asked for my seed phrase", "crypto_safety"), ("is this a scam", "crypto_safety"),
    ("rug pull", "crypto_safety"), ("protect my account with 2fa", "crypto_safety"),
    ("i got hacked", "crypto_safety"), ("someone promised to double my money", "crypto_safety"),
    ("a stranger on telegram offered guaranteed profits", "crypto_safety"), ("fraud", "crypto_safety"),

    # Intermediate: Investing & Trading
    ("how to buy crypto", "investing_trading"), ("trading cryptocurrency", "investing_trading"),
    ("what is hodl", "investing_trading"), ("crypto exchanges", "investing_trading"),
    ("how to swap tokens", "investing_trading"), ("portfolio diversification crypto", "investing_trading"),
    ("where can i buy bitcoin", "investing_trading"), ("should i invest in crypto", "investing_trading"),
    ("what is dollar cost averaging", "investing_trading"), ("how to sell crypto", "investing_trading"),
    ("best app to buy crypto", "investing_trading"),
    ("i lost money on crypto", "investing_trading"), ("the price crashed", "investing_trading"),

    # Intermediate: Web3 & Digital Ownership
    ("what is web3", "web3_nfts"), ("explain nfts", "web3_nfts"),
    ("digital ownership on blockchain", "web3_nfts"), ("what are dapps", "web3_nfts"),
    ("how to interact with decentralized apps", "web3_nfts"), ("metaverse and crypto", "web3_nfts"),
    ("what is an nft", "web3_nfts"), ("nft art", "web3_nfts"), ("what is a smart contract", "web3_nfts"),

    # Intermediate: Practical Everyday Use
    ("pay for things with crypto", "everyday_use"), ("crypto payments", "everyday_use"),
    ("international money transfer crypto", "everyday_use"), ("how do stablecoins work", "everyday_use"),
    ("usdc and usdt explained", "everyday_use"), ("crypto remittances", "everyday_use"),
    ("send money to my family abroad", "everyday_use"), ("what is a stablecoin", "everyday_use"),
    ("can i buy coffee with bitcoin", "everyday_use"), ("digital dollars", "everyday_use"),

    # Intermediate: Inflation & Savings (the CryptoClaro mission)
    ("protect my savings from inflation", "inflation_savings"), ("crypto and inflation", "inflation_savings"),
    ("my currency is losing value", "inflation_savings"), ("save in dollars", "inflation_savings"),
    ("is bitcoin a hedge against inflation", "inflation_savings"), ("saving money with crypto", "inflation_savings"),
    ("devaluation of my currency", "inflation_savings"), ("keep my savings safe from inflation", "inflation_savings"),

    # Intermediate: DeFi (Decentralized Finance)
    ("what is defi", "defi"), ("how does staking work", "defi"),
    ("earn interest on cryptocurrency", "defi"), ("decentralized lending", "defi"),
    ("yield farming basics", "defi"), ("liquidity pools explained", "defi"),
    ("what is staking", "defi"), ("decentralized finance", "defi"), ("borrow crypto", "defi"),
]

# 2. Coins the bot knows (used by coin detection in step 6)
# Each coin has the names people use for it. We look for them in the
# original words. ("sol" is left out on purpose: it's also Peru's currency.)
COINS = {
    "bitcoin": {
        "name": "Bitcoin (BTC)",
        "aliases": ["bitcoin", "bitcoins", "btc"],
        "profile": "Bitcoin (BTC) is the first cryptocurrency, launched in 2009 by an anonymous "
                   "creator called Satoshi Nakamoto. Only 21 million will ever exist, which is why "
                   "people call it 'digital gold' and use it as a long-term store of value. New coins "
                   "are created by 'mining', where computers compete to secure the network. "
                   "Main risk: its price can rise or fall 20% or more in weeks.",
        "note": "Bitcoin is the original cryptocurrency, with a fixed supply of 21 million coins.",
    },
    "ethereum": {
        "name": "Ethereum (ETH)",
        "aliases": ["ethereum", "eth", "ether"],
        "profile": "Ethereum (ETH) launched in 2015, created by Vitalik Buterin and others. It's more "
                   "than money: it runs 'smart contracts', programs that power most of DeFi, NFTs and "
                   "many stablecoins. Since 2022 it uses 'proof of stake' instead of mining, which uses "
                   "far less energy. Main risk: fees can get expensive when the network is busy, and "
                   "the price is volatile.",
        "note": "Ethereum is the main platform for smart contracts, DeFi and NFTs.",
    },
    "tether": {
        "name": "Tether (USDT)",
        "aliases": ["tether", "usdt"],
        "profile": "Tether (USDT) is the largest stablecoin, created in 2014. Each USDT is meant to "
                   "be worth exactly 1 US dollar, backed by reserves the company Tether holds. In Latin "
                   "America it's widely used to save in 'digital dollars' and for peer-to-peer payments. "
                   "Main risk: it depends on one company: Tether can freeze funds, and its reserves "
                   "have been questioned in the past.",
        "note": "Tether (USDT) is a stablecoin designed to always be worth 1 US dollar.",
    },
    "bnb": {
        "name": "BNB",
        "aliases": ["bnb", "binance coin"],
        "profile": "BNB was created in 2017 by Binance, the world's largest crypto exchange. It started "
                   "as a way to get discounts on trading fees and now also powers BNB Chain, a blockchain "
                   "with low fees. Main risk: its value is closely tied to one company, Binance, and "
                   "its legal and regulatory problems.",
        "note": "BNB is the coin of the Binance exchange and its BNB Chain network.",
    },
    "solana": {
        "name": "Solana (SOL)",
        "aliases": ["solana"],
        "profile": "Solana (SOL) launched in 2020, founded by Anatoly Yakovenko. It's built for speed: "
                   "thousands of transactions per second with fees of a fraction of a cent, which makes "
                   "it popular for apps, payments, NFTs and meme coins. Main risk: the network has had "
                   "several outages in the past, and the price is very volatile.",
        "note": "Solana is a fast blockchain known for very low transaction fees.",
    },
}


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

responses = {
    # Conversation
    "greeting": f"Hello! Welcome to {BOT_NAME}. I can explain the basics of cryptocurrency "
                "and some intermediate topics. What would you like to explore?",

    "help": "You can ask me about: what crypto is, why it has value, wallets, safety and scams, "
            "investing, Web3 and NFTs, everyday use and stablecoins, inflation and savings, or DeFi. "
            "I can also explain five major coins: Bitcoin, Ethereum, Tether (USDT), BNB and Solana.",

    "thanks": "You're welcome! Ask me anything else about crypto, or type 'bye' to leave.",

    "bye": "Goodbye! Stay safe out there in the crypto world!",

    # Beginner path
    "crypto_basics": "Cryptocurrency is digital money that runs on a blockchain: a shared public record "
                     "of transactions kept by thousands of computers instead of one bank. Bitcoin (2009) "
                     "was the first; Ethereum added programmable 'smart contracts'." + DISCLAIMER,

    "crypto_value": "Crypto doesn't rely on a central bank. Its value comes from supply and demand, "
                    "limited supply (only 21 million Bitcoin will ever exist), its usefulness, and the "
                    "trust of the people who use the network. That's also why prices can swing a lot." + DISCLAIMER,

    "crypto_wallets": "A crypto wallet stores your keys, not your coins. Your public address is like an "
                      "email address people send funds to; your private key (or seed phrase) is the secret "
                      "that approves transactions. Hot wallets are online apps, cold wallets are offline "
                      "devices. Never share your private key or seed phrase!" + DISCLAIMER,

    "crypto_safety": "To avoid scams: never share your seed phrase (no real company will ask for it), "
                     "ignore 'guaranteed profit' offers, double-check website URLs, and turn on 2FA. "
                     "Blockchain transactions are permanent and can't be reversed." + DISCLAIMER,

    # Intermediate path
    "investing_trading": "People buy crypto on exchanges (like Coinbase or Binance) or peer-to-peer. Common "
                         "strategies are holding long term ('HODL') or buying a fixed amount regularly "
                         "(dollar-cost averaging). Prices are volatile, so only use money you can afford "
                         "to lose." + DISCLAIMER,

    "web3_nfts": "Web3 is the idea of an internet where users own their data and digital assets. "
                 "NFTs (Non-Fungible Tokens) are unique tokens that prove ownership of a specific digital "
                 "item, and dApps are apps that run on smart contracts instead of a company's server." + DISCLAIMER,

    "everyday_use": "Crypto can be used for payments and for sending money across borders quickly. "
                    "Stablecoins like USDC and USDT are pegged to the US dollar, so their price stays "
                    "around $1. They're only as safe as the company holding the reserves behind them." + DISCLAIMER,

    "inflation_savings": "In countries with high inflation, some people save in stablecoins (digital dollars) "
                         "so their money doesn't lose value as fast as the local currency. Bitcoin is sometimes "
                         "called 'digital gold', but its price is very volatile in the short term." + DISCLAIMER,

    "defi": "Decentralized Finance (DeFi) replaces banks with smart contracts. Users can lend, borrow, "
            "or earn interest directly, and staking means locking coins to help secure a network in "
            "exchange for rewards. Higher yields usually mean higher risk." + DISCLAIMER,

    # Fallback
    "unknown": "I'm not sure about that topic. Type 'help' to see the crypto topics I can explain.",
}


# Intents where the bot explains a crypto topic
TOPIC_INTENTS = {"crypto_basics", "crypto_value", "crypto_wallets", "crypto_safety",
                 "investing_trading", "web3_nfts", "everyday_use", "inflation_savings", "defi"}


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
