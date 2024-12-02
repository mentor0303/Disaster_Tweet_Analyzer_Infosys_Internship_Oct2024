import nltk
nltk.download('vader_lexicon')
from flask import Flask, request, render_template, jsonify
import pickle
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load models and libraries
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

# Load the Logistic Regression model, vectorizer, and scaler
lr_model = pickle.load(open('lr_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
st = pickle.load(open('scaler.pkl', 'rb'))

# Function to predict disaster-related information from a tweet
def predict_disaster_tweet(tweet_text):
    # Preprocess tweet
    tweet_vectorized = vectorizer.transform([tweet_text])
    tweet_scaled = st.transform(tweet_vectorized.toarray())

    # Make prediction
    prediction = lr_model.predict(tweet_scaled)[0]
    is_disaster = "Disaster" if prediction == 1 else "Non-Disaster"

    # Extract location
    doc = nlp(tweet_text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    location = locations[0] if locations else "Unknown"

    # Sentiment analysis
    sentiment_scores = sia.polarity_scores(tweet_text)
    sentiment = 'Positive' if sentiment_scores['compound'] >= 0.05 else 'Negative' if sentiment_scores['compound'] <= -0.05 else 'Neutral'

    # Assign a dummy category for demonstration (replace with actual logic if available)
    category = "Hurricane" if is_disaster == "Disaster" else "None"

    return is_disaster, location, category, sentiment

# Route to render the main page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Prediction API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    tweet_text = data.get("tweet", "")

    # Predict disaster-related information
    is_disaster, location, category, sentiment = predict_disaster_tweet(tweet_text)

    # Create response data
    response_data = {
        "tweet_text": tweet_text,
        "is_disaster": is_disaster,
        "location": location,
        "category": category,
        "sentiment": sentiment
    }

    return jsonify(response_data)

# Additional routes
@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/submitted_feedback', methods=['POST'])
def feedback_submitted():
    return render_template('feedback_submitted.html')

@app.route('/motivation')
def motivation():
    return render_template('motivation.html')

@app.route('/model-insight')
def model_insight():
    return render_template('model-insight.html')

@app.route('/about-us')
def about_us():
    return render_template('about-us.html')
@app.route('/team')
def team():
    return render_template('team.html')

if __name__ == "__main__":
    app.run(debug=True)
