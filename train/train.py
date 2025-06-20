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
        logging.FileHandler(f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
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
        self.slang_dict = self.load_slang_dict('combined_slang_words.json')

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
        
    def load_data(self, filepath: str):
        try:
            df = pd.read_csv(filepath)
            texts = df['Sentence'].tolist()
            labels = [{'Negative': 0, 'Neutral': 1, 'Positive': 2}[label] for label in df['Sentiment']]
            logger.info(f"Loaded {len(df)} samples")
            return texts, labels
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def initialize_model(self):
        try:
            config = AutoConfig.from_pretrained(
                self.preprocessor.tokenizer.name_or_path,
                num_labels=len(self.label_map))
            
            self.model = BertForSequenceClassification.from_pretrained(
                self.preprocessor.tokenizer.name_or_path,
                config=config,
                use_safetensors=True
            )
            self.model.to(self.preprocessor.device)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def train(self, texts: List[str], labels: List[int], epochs: int = 3, batch_size: int = 16):
        
        self.initialize_model()
        
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels)
        
        
        train_encodings = self.preprocessor.tokenizer(
            train_texts, truncation=True, padding=True, max_length=128)
        val_encodings = self.preprocessor.tokenizer(
            val_texts, truncation=True, padding=True, max_length=128)
        
        
        train_dataset = SentimentDataset(train_encodings, train_labels)
        val_dataset = SentimentDataset(val_encodings, val_labels)
        
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_dir='./logs',
            load_best_model_at_end=False,
            save_safetensors=True,  
            save_on_each_node=True,  
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False
        )
        
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda p: {
                'accuracy': accuracy_score(
                    p.label_ids, 
                    np.argmax(p.predictions, axis=1))
            }
        )
        
        try:
            logger.info("Starting training...")
            trainer.train()
            
            self.save_model("./saved_model")
            
            eval_result = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_result}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            traceback.print_exc()
            raise

    def save_model(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': self.model.config
            },
            os.path.join(path, 'pytorch_model.bin')
        )
        
        self.model.config.save_pretrained(path)
        self.preprocessor.tokenizer.save_pretrained(path)
        
        logger.info(f"Model safely saved to {path}")


    def load_model(self, path: str):
        try:
            self.model = BertForSequenceClassification.from_pretrained(path, use_safetensors=True)
            self.model.to(self.preprocessor.device)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, text: str) -> Dict:
        try:
            
            original_device = next(self.model.parameters()).device
            if str(original_device) == 'mps':
                self.model.to('cpu')
                
            inputs = self.preprocessor.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to('cpu')  
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs).item()
            
            
            if str(original_device) == 'mps':
                self.model.to(original_device)
                
            return {
                "label": self.label_map[pred_idx],
                "confidence": float(probs[0][pred_idx]),
                "probabilities": {
                    label: float(prob) 
                    for label, prob in zip(self.label_map.values(), probs[0])
                }
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
            if str(original_device) == 'mps':
                self.model.to(original_device)
            raise

def main():
    try:
        logger.info("Starting sentiment analysis system")
        
        analyzer = SentimentAnalyzer()
        texts, labels = analyzer.load_data("sentiment_data.csv")
        
        logger.info("Training model...")
        analyzer.train(texts, labels, epochs=3, batch_size=16)
        
        
        test_text = "Produk ini sangat bagus dan berkualitas tinggi"
        result = analyzer.predict(test_text)
        logger.info(f"Prediction for '{test_text}': {result}")
        
    except Exception as e:
        logger.error(f"System failed: {e}")
        traceback.print_exc()
    finally:
        logger.info("System shutdown completed")

if __name__ == "__main__":
    main()