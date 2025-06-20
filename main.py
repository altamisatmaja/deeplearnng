import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NO_DTENSOR"] = "1"

import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import unicodedata
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logger/main_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IndonesianTextPreprocessor:
    def __init__(self, model_name: str = 'indobenchmark/indobert-base-p2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = stopword_factory.create_stop_word_remover()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.slang_dict = self.load_slang_dict('./data/combined_slang_words.json')

    def load_slang_dict(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading slang dict: {e}")
            return {}

    def clean_text(self, text: str, for_bert: bool = False):
        if not text:
            return ""
        
        text = re.sub(r'http\S+|www\S+|@\S+', '', text)
        
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower()
        
        text = re.sub(r'[^\w\s]', ' ', text)
        
        words = text.split()
        words = [self.slang_dict.get(word, word) for word in words]
        text = ' '.join(words)
        
        if not for_bert:
            text = self.stopword_remover.remove(text)
            words = text.split()
            words = [self.stemmer.stem(word) for word in words]
            text = ' '.join(words)
        
        return text.strip()

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SentimentAnalyzer:
    def __init__(self, model_name: str = 'indobenchmark/indobert-base-p2'):
        self.device = 'cpu'  
        self.preprocessor = IndonesianTextPreprocessor(model_name)
        self.model = None
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        
    def load_model(self, path: str):
        try:
            
            self.model = BertForSequenceClassification.from_pretrained(
                path,
                use_safetensors=True,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model berhasil dimuat dari {path}")
        except Exception as e:
            logger.error(f"Gagal memuat model: {e}")
            raise


    def predict(self, text: str) -> Dict:
        try:
            inputs = self.preprocessor.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs).item()
            
            return {
                "label": self.label_map[pred_idx],
                "confidence": float(probs[0][pred_idx]),
                "probabilities": {
                    label: float(prob) 
                    for label, prob in zip(self.label_map.values(), probs[0])
                }
            }
        except Exception as e:
            logger.error(f"Prediksi gagal: {e}")
            traceback.print_exc()
            raise

def main():
    try:
        logger.info("Memulai sistem analisis sentimen")
        
        analyzer = SentimentAnalyzer()
        analyzer.load_model("./converted_model_safe")
        
        slang_texts = [
            "gk ngerti sih gw ama pelayanannya",
            "aneh dah, ini mah bukan toko online yg bener",
            "udh dibilang jangan dibeli, tp dibeli jg",
            "btw dia tuh adminnya? bkn customer service ya?",
        ]

        short_texts = [
            "tdk ada yg bantu waktu saya butuh",
            "dr td saya nunggu kabar loh",
            "udh cek email blm?",
            "plis dong bales, gw udah capek nunggu",
        ]
        
        emotion_texts = [
            "sumpah ini produk terbaik yg pernah gue beli",
            "parah banget kirimannya telat dan rusak",     
            "oke sih, tapi biasa aja",                     
        ]

        mix_texts = [
            "gue udah checkout, tapi nggak dapet notifikasi payment",
            "delivery-nya cepat sih, tapi packing kurang aman",
            "sistem refund-nya ribet bgt",
        ]

        qa_texts = [
            "berapa lama pengiriman ke Bandung?",
            "bisa retur kalau ukurannya salah?",
            "siapa yang bisa saya hubungi kalau ada masalah?",
        ]
        
        test_texts = slang_texts + short_texts + emotion_texts + mix_texts + qa_texts

        
        for text in test_texts:
            result = analyzer.predict(text)
            logger.info(f"Hasil prediksi untuk '{text}': {result}")
            
    except Exception as e:
        logger.error(f"Sistem gagal: {e}")
        traceback.print_exc()
    finally:
        logger.info("Sistem dimatikan")
        
if __name__ == "__main__":
    main()