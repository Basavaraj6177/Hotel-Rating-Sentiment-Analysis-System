# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# import joblib
# import os

# app = Flask(__name__)

# # Global model variable
# model = None
# EXPECTED_FEATURES = 4  # Must match your form inputs

# def train_model():
#     global model
    
#     # Load data with 4 features
#     df = pd.read_csv('train.csv')
    
#     # Verify features
#     required_features = ['cleanliness', 'service', 'comfort', 'amenities']
#     X = df[required_features]
#     y = df['rating']
    
#     print(f"Training with features: {X.columns.tolist()}")
    
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X, y)
    
#     joblib.dump(model, 'rating_model.pkl')
#     print(f"Model trained with {model.n_features_in_} features")

# def load_model():
#     global model
#     try:
#         if os.path.exists('rating_model.pkl'):
#             model_data = joblib.load('rating_model.pkl')
#             if model_data['n_features'] != EXPECTED_FEATURES:
#                 raise ValueError("Feature mismatch in saved model")
                
#             model = model_data['model']
#             print(f"Model loaded with {model_data['n_features']} features")
#         else:
#             raise FileNotFoundError
#     except Exception as e:
#         print(f"Model loading failed: {str(e)} - Training new model")
#         train_model()

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     total_rating = None
#     error_message = None
#     rating_class = ""
    
#     if request.method == 'POST':
#         try:
#             # Get and validate 4 features from form
#             features = [
#                 int(request.form.get('cleanliness', 5)),
#                 int(request.form.get('service', 5)),
#                 int(request.form.get('comfort', 5)),
#                 int(request.form.get('amenities', 5))
#             ]
            
#             # Validate feature count
#             if len(features) != EXPECTED_FEATURES:
#                 raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {len(features)}")
            
#             # Clamp values between 1-10
#             features = [max(1, min(10, f)) for f in features]
            
#             # Predict
#             input_features = [features]
#             predicted_rating = model.predict(input_features)[0]
#             total_rating = round(max(1, min(10, predicted_rating)), 1)

#             if model.n_features_in_ != 4:
#                 error_message = "Model configuration mismatch!"
#                 return render_template(..., error_message=error_message)
            
#             # Determine rating class
#             if total_rating >= 8:
#                 rating_class = "good"
#             elif total_rating >= 5:
#                 rating_class = "average"
#             else:
#                 rating_class = "poor"
            
#         except Exception as e:
#             error_message = f"Error: {str(e)}"
#             rating_class = "error"
    
#     return render_template('index.html',
#                          total_rating=total_rating,
#                          error_message=error_message,
#                          rating_class=rating_class)

# if __name__ == '__main__':
#     load_model()
#     app.run(debug=True)
import traceback
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Global variables
hybrid_model = None
sentiment_model = None

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

def train_models():
    global hybrid_model, sentiment_model
    
    # Load and preprocess data
    df = pd.read_csv('train.csv')
    df = df.dropna(subset=['rating'])
    
    # Clean text
    df['cleaned_review'] = df['review_text'].apply(clean_text)
    
    # Train sentiment analysis model
    sentiment_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression())
    ])
    sentiment_model.fit(df['cleaned_review'], (df['rating'] >= 5).astype(int))
    
    # Train hybrid rating model
    numeric_features = ['cleanliness', 'service', 'comfort', 'amenities']
    text_features = 'cleaned_review'
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('text', TfidfVectorizer(max_features=1000), text_features)
        ])
    
    hybrid_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    hybrid_model.fit(df, df['rating'])
    
    # Save models
    joblib.dump(hybrid_model, 'hybrid_model.pkl')
    joblib.dump(sentiment_model, 'sentiment_model.pkl')

def load_models():
    global hybrid_model, sentiment_model
    if os.path.exists('hybrid_model.pkl') and os.path.exists('sentiment_model.pkl'):
        hybrid_model = joblib.load('hybrid_model.pkl')
        sentiment_model = joblib.load('sentiment_model.pkl')
        print("Models loaded successfully!")
        print(f"Hybrid model expects {hybrid_model.named_steps['preprocessor'].transformers_[0][2]} numerical features")
    else:
        print("Training new models...")
        train_models()

@app.route('/', methods=['GET', 'POST'])
def home():
    results = {
        'numerical_prediction': None,
        'sentiment_score': None,
        'final_score': None,
        'error': None
    }
    
    if request.method == 'POST':
        try:
            print("\n=== RAW FORM DATA ===")
            print("Form Data Received:", request.form)
            
            # Process numerical features
            numerical_features = [
                max(1, min(10, int(request.form.get(feature, 5))))
                for feature in ['cleanliness', 'service', 'comfort', 'amenities']
            ]
            print("\n=== PROCESSED NUMERICAL FEATURES ===")
            print("Cleanliness:", numerical_features[0])
            print("Service:", numerical_features[1])
            print("Comfort:", numerical_features[2])
            print("Amenities:", numerical_features[3])
            
            # Process text review
            review = clean_text(request.form.get('review', ''))
            print("\n=== PROCESSED REVIEW ===")
            print("Original Review:", request.form.get('review', ''))
            print("Cleaned Review:", review)
            
            # Create input DataFrame
            input_data = pd.DataFrame([{
                'cleanliness': numerical_features[0],
                'service': numerical_features[1],
                'comfort': numerical_features[2],
                'amenities': numerical_features[3],
                'review_text': review,
                'cleaned_review': review
            }])
            print("\n=== INPUT DATA FOR PREDICTION ===")
            print(input_data)
            
            # Make predictions
            numerical_pred = hybrid_model.predict(input_data)
            print("\n=== RAW MODEL PREDICTION ===")
            print("Model Output:", numerical_pred)
            print("Prediction Type:", type(numerical_pred))
            
            sentiment_prob = sentiment_model.predict_proba([review])
            print("\n=== SENTIMENT ANALYSIS ===")
            print("Sentiment Probabilities:", sentiment_prob)
            
            # Calculate final score
            final_score = 0.7 * numerical_pred[0] + 0.3 * sentiment_prob[0][1] * 10
            print("\n=== FINAL CALCULATIONS ===")
            print("Numerical Contribution:", 0.7 * numerical_pred[0])
            print("Sentiment Contribution:", 0.3 * sentiment_prob[0][1] * 10)
            print("Final Score:", final_score)
            
            # Update results
            results.update({
                'numerical_prediction': round(numerical_pred[0], 1),
                'sentiment_score': round(sentiment_prob[0][1] * 100, 1),
                'final_score': round(final_score, 1)
            })
            print("\n=== FINAL RESULTS ===")
            print(results)
            
        except Exception as e:
            print("\n!!! ERROR TRACEBACK !!!")
            traceback.print_exc()
            results['error'] = f"Prediction error: {str(e)}"
    
    return render_template('index.html', results=results)

if __name__ == '__main__':
    load_models()
    app.run(debug=True)