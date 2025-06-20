import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from  main import SentimentAnalyzer

os.makedirs("test", exist_ok=True)

df = pd.read_csv("data/sentiment_data_100.csv")

label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
df['label'] = df['Sentiment'].map(label_mapping)

analyzer = SentimentAnalyzer()
analyzer.load_model("./converted_model_safe")

df['cleaned'] = df['Sentence'].apply(lambda x: analyzer.preprocessor.clean_text(x, for_bert=True))

y_true = df['label'].tolist()
y_pred = []
results = []

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

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"],  zero_division=0))

result_df = pd.DataFrame(results)
result_df.to_csv("test/hasil_test.csv", index=False)
