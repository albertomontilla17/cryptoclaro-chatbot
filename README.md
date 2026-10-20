# CryptoClaro Bot (class version)

A Crypto 101 chatbot written in Python. It uses classic NLP and machine learning,
with no AI service or API, so every answer comes from code you can read.

## Files

| File | What it does |
|------|--------------|
| `chatbot.py` | The chatbot: training data, NLTK text cleaning, Naive Bayes model, answers. Runs on its own in the terminal. |
| `app.py` | A small Flask server that connects the web page to `chatbot.py`. |
| `templates/index.html` | The chat interface, with an "Under the hood" panel showing how the model read each message. |
| `requirements.txt` | Python libraries to install. |

## Run it

```bash
pip install -r requirements.txt
```

Web interface (recommended for the demo):

```bash
python app.py
```

Then open http://127.0.0.1:5000

Terminal only:

```bash
python chatbot.py
```

The first run downloads small NLTK data files, so it needs internet once.

## How it works

1. **Clean the text (NLTK):** lowercase, split into words, remove stop words ("the", "is", ...), and stem ("wallets" becomes "wallet").
2. **Turn words into numbers (TF-IDF):** words that appear in many topics, like "crypto", count less.
3. **Classify (Multinomial Naive Bayes):** the model predicts which of 13 intents the message belongs to.
4. **Confidence check:** if the top guess is below 15%, or no known words are found, the bot says it doesn't know.
5. **Sentiment (NLTK VADER):** scores the original message from -1 (negative) to +1 (positive). VADER is a ready-made model, so it needs no training data. A few crypto words like "scammed" and "hacked" were added to its word list.
6. **Answer:** each intent has one fixed, educational answer. The mood adjusts the tone: a sympathetic opening for negative messages and an upbeat one for positive messages.

Intent classification answers *what* the user is asking about. Sentiment analysis answers *how they feel* about it.

Topics: greeting, help, thanks, bye, crypto basics, value, wallets, safety and scams,
investing, Web3 and NFTs, everyday use and stablecoins, inflation and savings, DeFi.
