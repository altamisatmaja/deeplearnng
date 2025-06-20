import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from main import SentimentAnalyzer
import os

@st.cache_resource
def load_analyzer():
    analyzer = SentimentAnalyzer()
    analyzer.load_model("./converted_model_safe")
    return analyzer

analyzer = load_analyzer()

st.title("Sentiment Analysis Chat")
st.write("Analyze sentiment of text or upload a CSV file for batch processing")

with st.sidebar:
    st.header("Model Information")
    st.write("This app uses a fine-tuned sentiment analysis model")
    st.write("Labels: Negative (0), Neutral (1), Positive (2)")


tab1, tab2 = st.tabs(["Interactive Chat", "Batch Processing"])

with tab1:
    st.header("Chat with the Sentiment Analyzer")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Enter text to analyze sentiment"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        cleaned_text = analyzer.preprocessor.clean_text(prompt, for_bert=True)
        pred = analyzer.predict(cleaned_text)
        
        
        response = f"""
        **Sentiment:** {pred['label']}  
        **Confidence:** {pred['confidence']:.2%}  
        **Cleaned Text:** {cleaned_text}
        """
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.header("Batch Processing")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        
        if "Sentence" not in df.columns or "Sentiment" not in df.columns:
            st.error("CSV must contain 'Sentence' and 'Sentiment' columns")
        else:
            
            label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
            df['label'] = df['Sentiment'].map(label_mapping)
            df['cleaned'] = df['Sentence'].apply(lambda x: analyzer.preprocessor.clean_text(x, for_bert=True))
            
            y_true = df['label'].tolist()
            y_pred = []
            results = []
            
            with st.spinner("Analyzing sentiments..."):
                for idx, text in enumerate(df['cleaned']):
                    pred = analyzer.predict(text)
                    predicted_label_id = list(analyzer.label_map.keys())[list(analyzer.label_map.values()).index(pred["label"])]
                    y_pred.append(predicted_label_id)
                    
                    results.append({
                        "original": df['Sentence'][idx],
                        "cleaned": text,
                        "true_label": df['Sentiment'][idx],
                        "predicted_label": pred["label"],
                        "confidence": pred["confidence"]
                    })
            
            
            st.subheader("Performance Metrics")
            st.write("Accuracy:", accuracy_score(y_true, y_pred))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_true, y_pred))
            st.write("Classification Report:")
            st.write(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"], zero_division=0))
            
            
            st.subheader("Analysis Results")
            result_df = pd.DataFrame(results)
            st.dataframe(result_df)
            
            
            st.download_button(
                label="Download Results",
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )