"""
Web interface for the CryptoClaro Bot.

Run:  python app.py
Then open http://127.0.0.1:5000 in your browser.

The browser page sends each message to /api/chat. This file passes it
to chatbot.py, which does all the NLP and machine-learning work.
"""

from flask import Flask, jsonify, render_template, request

import chatbot

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", bot_name=chatbot.BOT_NAME)


@app.route("/api/chat", methods=["POST"])
def chat():
    message = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    details = chatbot.analyze(message)
    sentiment = details["sentiment"]
    return jsonify({
        "reply": chatbot.generate_response(details["intent"], sentiment["label"], details["coins"]),
        "intent": details["intent"],
        "coins": [chatbot.COINS[c]["name"] for c in details["coins"]],
        "sentiment": sentiment["label"],
        "sentiment_score": round(sentiment["score"], 3),
        "confidence": round(details["confidence"], 3),
        "top": [{"intent": i, "probability": round(p, 3)} for i, p in details["top"]],
        "clean_text": details["clean_text"],
        "known_words": details["known_words"],
        "threshold": chatbot.CONFIDENCE_THRESHOLD,
    })


if __name__ == "__main__":
    app.run(debug=True)
