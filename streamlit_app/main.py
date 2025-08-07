# IMDB Movie Review Sentiment Analysis - Enhanced Streamlit App
# Created for Simple RNN sentiment classification

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import os
from datetime import datetime
import base64

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }
    .positive-sentiment {
        background: linear-gradient(45deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .negative-sentiment {
        background: linear-gradient(45deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #4ECDC4;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_and_data():
    """Load model and IMDB word mappings with caching for performance"""
    try:
        # Try different model paths
        model_paths = ['../models/simple_rnn_imdb.h5', 'models/simple_rnn_imdb.h5', 'simple_rnn_imdb.h5']
        model = None
        
        for path in model_paths:
            if os.path.exists(path):
                model = load_model(path)
                break
        
        if model is None:
            st.error("❌ Model file not found! Please ensure the model is trained and saved.")
            return None, None, None
        
        # Load word mappings
        word_index = imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        return model, word_index, reverse_word_index
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def clean_text(text):
    """Enhanced text cleaning for better predictions"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

def preprocess_text(text, word_index, max_len=500):
    """Convert text to model input format with detailed analysis"""
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    
    # Convert words to indices
    encoded_review = []
    unknown_words = []
    
    for word in words:
        if word in word_index:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2)  # Unknown word token
            unknown_words.append(word)
    
    # Pad sequence
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    
    return padded_review, unknown_words, len(words), cleaned_text

def predict_sentiment_enhanced(text, model, word_index):
    """Enhanced prediction with detailed analysis"""
    if not text.strip():
        return None
    
    # Preprocess text
    preprocessed_input, unknown_words, word_count, cleaned_text = preprocess_text(text, word_index)
    
    # Make prediction
    prediction_prob = model.predict(preprocessed_input, verbose=0)[0][0]
    
    # Determine sentiment and confidence
    sentiment = 'Positive' if prediction_prob > 0.5 else 'Negative'
    confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
    
    # Calculate metrics
    unknown_ratio = len(unknown_words) / word_count if word_count > 0 else 0
    
    return {
        'sentiment': sentiment,
        'probability': float(prediction_prob),
        'confidence': float(confidence),
        'word_count': word_count,
        'unknown_words': unknown_words,
        'unknown_ratio': unknown_ratio,
        'cleaned_text': cleaned_text
    }

def get_confidence_level(confidence):
    """Categorize confidence levels"""
    if confidence >= 0.9:
        return "Very High", "🟢"
    elif confidence >= 0.8:
        return "High", "🟡"
    elif confidence >= 0.7:
        return "Moderate", "🟠"
    elif confidence >= 0.6:
        return "Low", "🔴"
    else:
        return "Very Low", "⚫"

def create_confidence_gauge(confidence):
    """Create a confidence gauge using Plotly"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300)
    return fig

def create_probability_chart(probability):
    """Create probability visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Probability Distribution', 'Sentiment Score'),
        specs=[[{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Probability bar chart
    fig.add_trace(
        go.Bar(x=['Negative', 'Positive'], 
               y=[1-probability, probability],
               marker_color=['red', 'green']),
        row=1, col=1
    )
    
    # Indicator
    fig.add_trace(
        go.Indicator(
            mode = "number+gauge",
            value = probability,
            title = {'text': "Positive Probability"},
            gauge = {'axis': {'range': [0, 1]},
                    'bar': {'color': "green" if probability > 0.5 else "red"}},
            domain = {'x': [0, 1], 'y': [0, 1]}
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400)
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Load model and data
    model, word_index, reverse_word_index = load_model_and_data()
    
    if model is None:
        st.stop()
    
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    st.markdown('<h1 class="main-header">🎬 IMDB Movie Review Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Simple RNN Deep Learning Model</p>', unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    st.sidebar.markdown("## 🎯 Model Information")
    st.sidebar.info(f"""
    **Model Architecture:** Simple RNN
    **Training Data:** IMDB 50k Reviews
    **Vocabulary Size:** {len(word_index):,} words
    **Parameters:** {model.count_params():,}
    **Input Length:** 500 words max
    """)
    
    st.sidebar.markdown("## 📊 How it Works")
    st.sidebar.markdown("""
    1. **Text Preprocessing**: Cleans and tokenizes your review
    2. **Word Encoding**: Converts words to numerical indices
    3. **Sequence Padding**: Adjusts length to 500 words
    4. **RNN Processing**: Analyzes sequential patterns
    5. **Sentiment Prediction**: Outputs probability score
    """)
    
    st.sidebar.markdown("## 🎮 Sample Reviews")
    sample_reviews = {
        "Positive Example": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        "Negative Example": "Terrible movie with poor acting and a confusing storyline. Complete waste of time.",
        "Mixed Example": "The movie had great visuals but the story was somewhat predictable and slow."
    }
    
    selected_sample = st.sidebar.selectbox("Choose a sample review:", list(sample_reviews.keys()))
    if st.sidebar.button("Use Sample Review"):
        st.session_state.sample_text = sample_reviews[selected_sample]
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📝 Enter Your Movie Review")
        
        # Text input with sample text handling
        default_text = st.session_state.get('sample_text', '')
        user_input = st.text_area(
            "Write your movie review here:",
            value=default_text,
            height=200,
            placeholder="Example: This movie was amazing! The cinematography was breathtaking and the story was compelling..."
        )
        
        # Clear the sample text after use
        if 'sample_text' in st.session_state:
            del st.session_state.sample_text
        
        # Analysis options
        st.markdown("### ⚙️ Analysis Options")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            show_details = st.checkbox("Show detailed analysis", value=True)
        with col_opt2:
            show_preprocessing = st.checkbox("Show preprocessing steps", value=False)
        with col_opt3:
            show_visualizations = st.checkbox("Show visualizations", value=True)
    
    with col2:
        st.markdown("## 📈 Quick Stats")
        if user_input:
            word_count = len(user_input.split())
            char_count = len(user_input)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Text Metrics</h3>
                <p><strong>Words:</strong> {word_count}</p>
                <p><strong>Characters:</strong> {char_count}</p>
                <p><strong>Sentences:</strong> {user_input.count('.') + user_input.count('!') + user_input.count('?')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # PREDICTION SECTION
    # ========================================================================
    
    if st.button("🎯 Analyze Sentiment", key="analyze_btn"):
        if user_input.strip():
            with st.spinner("🔍 Analyzing sentiment..."):
                # Simulate processing time for better UX
                time.sleep(1)
                
                # Make prediction
                result = predict_sentiment_enhanced(user_input, model, word_index)
                
                if result:
                    # ============================================================
                    # MAIN RESULTS DISPLAY
                    # ============================================================
                    st.markdown("## 🎯 Sentiment Analysis Results")
                    
                    # Create result cards
                    col_result1, col_result2, col_result3 = st.columns(3)
                    
                    with col_result1:
                        sentiment_class = "positive-sentiment" if result['sentiment'] == 'Positive' else "negative-sentiment"
                        sentiment_emoji = "😊" if result['sentiment'] == 'Positive' else "😞"
                        
                        st.markdown(f"""
                        <div class="{sentiment_class}">
                            <h2>{sentiment_emoji} {result['sentiment']}</h2>
                            <p>Predicted Sentiment</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_result2:
                        confidence_level, confidence_emoji = get_confidence_level(result['confidence'])
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h2>{confidence_emoji} {result['confidence']:.1%}</h2>
                            <p>Confidence Level: {confidence_level}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_result3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h2>📊 {result['probability']:.3f}</h2>
                            <p>Raw Probability Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ============================================================
                    # DETAILED ANALYSIS
                    # ============================================================
                    if show_details:
                        st.markdown("## 📊 Detailed Analysis")
                        
                        # Text analysis metrics
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.markdown("### 📝 Text Analysis")
                            st.write(f"**Word Count:** {result['word_count']}")
                            st.write(f"**Unknown Words:** {len(result['unknown_words'])} ({result['unknown_ratio']:.1%})")
                            
                            if result['unknown_words'] and len(result['unknown_words']) <= 10:
                                st.write(f"**Unknown Words List:** {', '.join(result['unknown_words'])}")
                            elif len(result['unknown_words']) > 10:
                                st.write(f"**Unknown Words (first 10):** {', '.join(result['unknown_words'][:10])}...")
                        
                        with col_detail2:
                            st.markdown("### 🎯 Prediction Breakdown")
                            if result['sentiment'] == 'Positive':
                                st.write(f"**Positive Evidence:** {result['probability']:.1%}")
                                st.write(f"**Negative Evidence:** {1-result['probability']:.1%}")
                            else:
                                st.write(f"**Negative Evidence:** {1-result['probability']:.1%}")
                                st.write(f"**Positive Evidence:** {result['probability']:.1%}")
                            
                            st.write(f"**Decision Threshold:** 50%")
                            st.write(f"**Confidence Margin:** {abs(result['probability'] - 0.5):.1%}")
                    
                    # ============================================================
                    # PREPROCESSING DETAILS
                    # ============================================================
                    if show_preprocessing:
                        st.markdown("## 🔧 Preprocessing Details")
                        
                        col_prep1, col_prep2 = st.columns(2)
                        
                        with col_prep1:
                            st.markdown("### 📝 Original Text")
                            st.text_area("Original Review", user_input, height=100, disabled=True)
                        
                        with col_prep2:
                            st.markdown("### 🧹 Cleaned Text")
                            st.text_area("Processed Review", result['cleaned_text'], height=100, disabled=True)
                    
                    # ============================================================
                    # VISUALIZATIONS
                    # ============================================================
                    if show_visualizations:
                        st.markdown("## 📊 Visual Analysis")
                        
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            st.markdown("### 🎯 Confidence Gauge")
                            confidence_fig = create_confidence_gauge(result['confidence'])
                            st.plotly_chart(confidence_fig, use_container_width=True)
                        
                        with col_viz2:
                            st.markdown("### 📊 Probability Distribution")
                            prob_fig = create_probability_chart(result['probability'])
                            st.plotly_chart(prob_fig, use_container_width=True)
                    
                    # ============================================================
                    # ADDITIONAL INSIGHTS
                    # ============================================================
                    st.markdown("## 💡 Insights & Recommendations")
                    
                    insights = []
                    
                    # Confidence-based insights
                    if result['confidence'] >= 0.9:
                        insights.append("✅ Very high confidence prediction - the model is very certain about this classification.")
                    elif result['confidence'] >= 0.7:
                        insights.append("✅ Good confidence prediction - the model shows reasonable certainty.")
                    else:
                        insights.append("⚠️ Lower confidence prediction - the review might contain mixed signals or ambiguous language.")
                    
                    # Unknown words insights
                    if result['unknown_ratio'] > 0.2:
                        insights.append("📝 High percentage of unknown words detected - consider using more common vocabulary for better accuracy.")
                    elif result['unknown_ratio'] > 0.1:
                        insights.append("📝 Some unknown words detected - this might slightly affect prediction accuracy.")
                    else:
                        insights.append("✅ Most words recognized by the model - good vocabulary match with training data.")
                    
                    # Length insights
                    if result['word_count'] < 10:
                        insights.append("📏 Short review detected - longer reviews often provide more reliable predictions.")
                    elif result['word_count'] > 400:
                        insights.append("📏 Long review detected - the model uses the first 500 words for analysis.")
                    
                    # Display insights
                    for insight in insights:
                        st.markdown(f"- {insight}")
                    
                else:
                    st.error("❌ Error processing your review. Please try again.")
        else:
            st.warning("⚠️ Please enter a movie review to analyze.")
    
    # ========================================================================
    # BATCH ANALYSIS SECTION
    # ========================================================================
    st.markdown("---")
    st.markdown("## 📊 Batch Analysis")
    st.markdown("Analyze multiple reviews at once for comparison")
    
    with st.expander("🔍 Batch Review Analysis"):
        batch_reviews = st.text_area(
            "Enter multiple reviews (one per line):",
            height=150,
            placeholder="Review 1: This movie was amazing!\nReview 2: Terrible film, waste of time.\nReview 3: Mixed feelings about this one..."
        )
        
        if st.button("🔄 Analyze Batch"):
            if batch_reviews.strip():
                reviews = [review.strip() for review in batch_reviews.split('\n') if review.strip()]
                
                if reviews:
                    batch_results = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, review in enumerate(reviews):
                        status_text.text(f'Processing review {i+1}/{len(reviews)}...')
                        result = predict_sentiment_enhanced(review, model, word_index)
                        if result:
                            batch_results.append({
                                'Review': review[:50] + '...' if len(review) > 50 else review,
                                'Sentiment': result['sentiment'],
                                'Probability': result['probability'],
                                'Confidence': result['confidence'],
                                'Word Count': result['word_count']
                            })
                        progress_bar.progress((i + 1) / len(reviews))
                    
                    status_text.text('Analysis complete!')
                    
                    # Display results
                    if batch_results:
                        st.markdown("### 📋 Batch Results")
                        
                        # Create DataFrame
                        df = pd.DataFrame(batch_results)
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary statistics
                        col_summary1, col_summary2, col_summary3 = st.columns(3)
                        
                        with col_summary1:
                            positive_count = sum(1 for r in batch_results if r['Sentiment'] == 'Positive')
                            st.metric("Positive Reviews", positive_count, f"{positive_count/len(batch_results):.1%}")
                        
                        with col_summary2:
                            avg_confidence = np.mean([r['Confidence'] for r in batch_results])
                            st.metric("Average Confidence", f"{avg_confidence:.1%}")
                        
                        with col_summary3:
                            avg_words = np.mean([r['Word Count'] for r in batch_results])
                            st.metric("Average Word Count", f"{avg_words:.0f}")
                        
                        # Visualization
                        fig = px.scatter(df, x='Word Count', y='Confidence', 
                                       color='Sentiment', title='Confidence vs Word Count by Sentiment')
                        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🎬 IMDB Sentiment Analyzer | Built with Streamlit & TensorFlow</p>
        <p>Model trained on 50,000 IMDB movie reviews using Simple RNN architecture</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
