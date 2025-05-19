import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoConfig
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import PreTrainedTokenizerFast


class VefkDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Dataset oluşturuluyor - Örnek sayısı: {len(texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        print(f"Tokenize ediliyor - Örnek {idx}: {text[:50]}...")
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        print(f"Tokenize edildi - Shape: {encoding['input_ids'].shape}")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class VefkTransformer(nn.Module):
    def __init__(self, num_classes, pretrained_model_name="dbmdz/bert-base-turkish-cased"):
        super(VefkTransformer, self).__init__()
        print(f"Model yükleniyor: {pretrained_model_name}")
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.is_decoder = True
        config.output_hidden_states = True
        self.bert = AutoModelForCausalLM.from_pretrained(pretrained_model_name, config=config)
        print("Model yüklendi")
        print(f"Hidden size: {self.bert.config.hidden_size}")
        
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = self.bert.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        print(f"Sınıf sayısı: {num_classes}")

    def forward(self, input_ids, attention_mask, labels=None):
        print(f"Input shape: {input_ids.shape}")
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Son katmanın hidden state'ini al
        sequence_output = outputs.hidden_states[-1]
        print(f"Sequence output shape: {sequence_output.shape}")
        
        # CLS token'ının çıktısını al (ilk token)
        pooled_output = sequence_output[:, 0, :]
        print(f"Pooled output shape: {pooled_output.shape}")
        
        # Dense katmanından geçir
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        print(f"After dense shape: {pooled_output.shape}")
        
        # Sınıflandırma
        logits = self.classifier(pooled_output)
        print(f"Logits shape: {logits.shape}")
        
        # Eğitim sırasında loss hesapla
        if labels is not None:
            loss = self.criterion(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}

    def train_model(self, texts, labels, epochs=3, batch_size=2):
        print("\n=== Model Eğitimi Başlıyor ===")
        print(f"Toplam örnek sayısı: {len(texts)}")
        print(f"Eğitim parametreleri:")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        
        # Veriyi eğitim ve test setlerine ayır
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        print(f"\nVeri seti boyutları:")
        print(f"- Eğitim seti: {len(train_texts)} örnek")
        print(f"- Doğrulama seti: {len(val_texts)} örnek")

        print("\nDataset'ler oluşturuluyor...")
        train_dataset = VefkDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = VefkDataset(val_texts, val_labels, self.tokenizer)
        print("Dataset'ler oluşturuldu")

        print("\nModel parametreleri ayarlanıyor...")
        training_args = TrainingArguments(
            output_dir="./vefk_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=1,
            no_cuda=True,
            save_strategy="no",
            report_to="none",
            learning_rate=1e-4
        )

        print("\nTrainer ayarlanıyor...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()}
        )

        print("\nEğitim başlıyor...")
        try:
            trainer.train()
            print("Model eğitimi tamamlandı!")
            
            # Modeli değerlendirme
            eval_results = trainer.evaluate()
            print(f"\nDeğerlendirme sonuçları:")
            print(f"Doğruluk: {eval_results['eval_accuracy']:.4f}")
            
        except Exception as e:
            print(f"\nEğitim sırasında hata oluştu:")
            print(f"Hata mesajı: {str(e)}")
            print("\nHata detayları:")
            import traceback
            traceback.print_exc()

class VefkNLP:
    def __init__(self):
        print("BERT modeli yükleniyor...")
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        print("BERT modeli yüklendi!")
        self.model = None
        self.intent_classes = []
        self.load_nltk_resources()
        
    def load_nltk_resources(self):
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
        except Exception as e:
            print(f"NLTK kaynakları yüklenirken hata: {e}")

    def preprocess_text(self, text):
        # Metni küçük harfe çevir
        text = text.lower()
        
        # Kelimelere ayır
        tokens = word_tokenize(text)
        
        # Stop words'leri kaldır
        stop_words = set(stopwords.words('turkish'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Tekrar birleştir
        return ' '.join(tokens)

    def prepare_training_data(self, conversations):
        texts = []
        labels = []
        
        for conv in conversations:
            texts.append(self.preprocess_text(conv['input']))
            labels.append(conv['intent'])
            
        self.intent_classes = list(set(labels))
        label_map = {label: idx for idx, label in enumerate(self.intent_classes)}
        numeric_labels = [label_map[label] for label in labels]
        
        return texts, numeric_labels

    def train_model(self, texts, labels, epochs=3, batch_size=2):
        print("\n=== Model Eğitimi Başlıyor ===")
        print(f"Toplam örnek sayısı: {len(texts)}")
        print(f"Eğitim parametreleri:")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        
        # Veriyi eğitim ve test setlerine ayır
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        print(f"\nVeri seti boyutları:")
        print(f"- Eğitim seti: {len(train_texts)} örnek")
        print(f"- Doğrulama seti: {len(val_texts)} örnek")

        print("\nDataset'ler oluşturuluyor...")
        train_dataset = VefkDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = VefkDataset(val_texts, val_labels, self.tokenizer)
        print("Dataset'ler oluşturuldu")

        print("\nModel parametreleri ayarlanıyor...")
        training_args = TrainingArguments(
            output_dir="./vefk_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=1,
            no_cuda=True,
            save_strategy="no",
            report_to="none",
            learning_rate=1e-4
        )

        print("\nModel oluşturuluyor...")
        self.model = VefkTransformer(num_classes=len(self.intent_classes))
        
        print("\nModel mimarisi:")
        print(self.model)

        print("\nTrainer ayarlanıyor...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        print("\nEğitim başlıyor...")
        try:
            trainer.train()
            print("Model eğitimi tamamlandı!")
        except Exception as e:
            print(f"\nEğitim sırasında hata oluştu:")
            print(f"Hata mesajı: {str(e)}")
            print("\nHata detayları:")
            import traceback
            traceback.print_exc()

    def predict_intent(self, text):
        if self.model is None:
            return "Model henüz eğitilmedi"

        # Metni ön işle
        processed_text = self.preprocess_text(text)
        
        # Tokenize et
        encoding = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tahmin yap
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(encoding['input_ids'], encoding['attention_mask'])
            predicted_intent = torch.argmax(outputs['logits'], dim=1).item()

        return self.intent_classes[predicted_intent]

class VefkEsmaUygulamasi:
    def __init__(self, root):
        print("Uygulama başlatılıyor...")
        self.root = root
        self.root.title("Vefk ve Esma-ül Hüsna Chatbot")
        self.root.geometry("800x600")
        
        print("Arayüz oluşturuluyor...")
        # Ana çerçeve
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Chatbot alanı
        self.chat_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=20)
        self.chat_area.pack(fill=tk.BOTH, expand=True, pady=5)
        self.chat_area.config(state=tk.DISABLED)
        
        # Kullanıcı giriş alanı
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.process_input)
        
        send_button = ttk.Button(input_frame, text="Gönder", command=self.process_input)
        send_button.pack(side=tk.RIGHT, padx=5)
        
        print("NLP modeli başlatılıyor...")
        self.nlp = VefkNLP()
        
        print("Esma verileri yükleniyor...")
        self.esma_data = self.esma_verileri_yukle()  # ESMA_DATA yerine esma.json'dan yükle
        
        # Gezegen-gün eşleşmeleri
        self.gezegen_gun_eslesmeleri = {
            "Şems": "Güneş - Pazar",
            "Kamer": "Ay - Pazartesi",
            "Merih": "Mars - Salı",
            "Utarid": "Merkür - Çarşamba",
            "Müşteri": "Jüpiter - Perşembe",
            "Zühre": "Venüs - Cuma",
            "Zuhal": "Satürn - Cumartesi"
        }
        
        # Türkçe isimden Arapça isime eşleştirme
        self.turkce_arapca_eslesme = {
            "güneş": "Şems",
            "ay": "Kamer",
            "mars": "Merih", 
            "merkür": "Utarid",
            "jüpiter": "Müşteri",
            "venüs": "Zühre",
            "satürn": "Zuhal"
        }
        
        self.son_vefk_esmalari = []
        
        print("Hoş geldiniz mesajı gösteriliyor...")
        self.add_bot_message("Esma-ül Hüsna ve Vefk Chatbot'una hoş geldiniz! Size nasıl yardımcı olabilirim?")
        
        print("Model eğitimi başlatılıyor...")
        self.load_and_train_model()

    def normalize_text(self, text):
        # Türkçe karakterleri normalize et
        replacements = {
            'â': 'a', 'î': 'i', 'û': 'u',
            'Â': 'A', 'Î': 'I', 'Û': 'U',
            'ı': 'i', 'İ': 'I', 'ğ': 'g',
            'Ğ': 'G', 'ü': 'u', 'Ü': 'U',
            'ş': 's', 'Ş': 'S', 'ö': 'o',
            'Ö': 'O', 'ç': 'c', 'Ç': 'C',
            'é': 'e', 'É': 'E'
        }
        normalized = text.lower()
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized

    def find_esma(self, esma_adi):
        normalized_input = self.normalize_text(esma_adi)
        
        # Ya ile başlayan ifadeler için özel kontrol
        if normalized_input.lower().startswith("ya-") or normalized_input.lower().startswith("ya "):
            # "ya-" veya "ya " ön ekini kaldır
            if normalized_input.lower().startswith("ya-"):
                normalized_without_ya = normalized_input[3:]
            else:
                normalized_without_ya = normalized_input[3:]
                
            # Kalan isme göre arama yap
            for gezegen, esmalar in self.esma_data.items():
                for esma in esmalar:
                    esma_normalized = self.normalize_text(esma["isim"])
                    # "el-", "er-" gibi ön ekleri kaldır
                    esma_without_prefix = esma_normalized
                    for prefix in ["el-", "er-", "es-", "eş-", "en-", "ed-", "ez-", "et-", "zü'l-"]:
                        if esma_without_prefix.startswith(prefix):
                            esma_without_prefix = esma_without_prefix[len(prefix):]
                            break
                    
                    if normalized_without_ya == esma_without_prefix:
                        return esma, gezegen
        
        # Türkçe gezegen ismiyle arama yapılıyorsa Arapça karşılığını bulalım
        for turkce, arapca in self.turkce_arapca_eslesme.items():
            if turkce in normalized_input.lower():
                for esma in self.esma_data[arapca]:
                    esma_normalized = self.normalize_text(esma["isim"])
                    if esma_normalized in normalized_input or normalized_input in esma_normalized:
                        return esma, arapca
        
        # Normal arama
        for gezegen, esmalar in self.esma_data.items():
            for esma in esmalar:
                if self.normalize_text(esma["isim"]) == normalized_input:
                    return esma, gezegen
        return None, None

    def add_bot_message(self, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, "🤖 Bot: " + message + "\n\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)
    
    def add_user_message(self, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, "👤 Siz: " + message + "\n\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)
    
    def esma_verileri_yukle(self):
        try:
            # JSON dosyasını oku
            with open('esma.json', 'r', encoding='utf-8') as file:
                esma_data = json.load(file)
        
            # Gezegen sayıları
            self.gezegen_sayilari = {
                "Şems": 1,
            "Kamer": 2,
            "Merih": 3,
            "Utarid": 4,
            "Müşteri": 5,
            "Zühre": 6,
            "Zuhal": 7
            }
        
            return esma_data
        
        except Exception as e:
            print(f"Esma verileri yüklenirken hata oluştu: {str(e)}")
            return {}
    
    def uclu_vefk_hesapla(self, user_input):
        # Başlangıçta son vefk esmalarını temizle
        self.son_vefk_esmalari = []
        
        # Sabit element desenleri
        element_patterns = {
            "toprak": [4, 9, 2, 3, 5, 7, 8, 1, 6],
            "ateş": [8, 3, 4, 1, 5, 9, 6, 7, 2],
            "hava": [6, 1, 8, 7, 5, 3, 2, 9, 4],
            "su": [2, 7, 6, 9, 5, 1, 4, 3, 8]
        }
        
        # Derece-eksiltme-artırma değerleri
        derece_cikarma = {
            30: {"cikarma": 12, "artirma": 1},
            60: {"cikarma": 24, "artirma": 2},
            90: {"cikarma": 36, "artirma": 3},
            120: {"cikarma": 48, "artirma": 4},
            150: {"cikarma": 60, "artirma": 5},
            180: {"cikarma": 72, "artirma": 6}
        }
        
        # Elementi bulun - doğrudan user_input içinde tam element ismini arayarak
        element = None
        if "toprak" in user_input.lower():
            element = "toprak"
        elif "ateş" in user_input.lower() or "ates" in user_input.lower():
            element = "ateş"
        elif "hava" in user_input.lower():
            element = "hava"
        elif "su" in user_input.lower():
            element = "su"
        
        if not element:
            element = "toprak"  # Varsayılan element
        
        # Dereceyi bulun
        derece = None
        for d in derece_cikarma.keys():
            if str(d) in user_input:
                derece = d
                break
        
        if not derece:
            derece = 30  # Varsayılan derece
        
        # Esmaları bul ve ebced değerlerini topla
        toplam_ebced = 0
        bulunan_esmalar = []
        
        # Benzersiz esmaları takip etmek için küme oluştur
        eklenen_esmalar = set()
        
        # Allah ve Zülcelal özel durumlarını kontrol et
        if "allah" in user_input.lower() or "zülcelal" in user_input.lower():
            for gezegen, esmalar in self.esma_data.items():
                for esma in esmalar:
                    if "allah" in esma["isim"].lower() and "allah" in user_input.lower() and esma["isim"] not in eklenen_esmalar:
                        toplam_ebced += esma["ebced"]
                        bulunan_esmalar.append((esma["isim"], esma["ebced"], gezegen))
                        eklenen_esmalar.add(esma["isim"])
                    elif "zülcelal" in esma["isim"].lower() and "zülcelal" in user_input.lower() and esma["isim"] not in eklenen_esmalar:
                        toplam_ebced += esma["ebced"]
                        bulunan_esmalar.append((esma["isim"], esma["ebced"], gezegen))
                        eklenen_esmalar.add(esma["isim"])
        
        # Ya ile başlayan ifadeleri kontrol et
        ya_matches = re.finditer(r'ya[\s-]([a-zçğıöşüâîû]+)', user_input, re.IGNORECASE)
        for ya_match in ya_matches:
            ya_esma = f"ya-{ya_match.group(1)}"
            esma_obj, gezegen = self.find_esma(ya_esma)
            if esma_obj and esma_obj["isim"] not in eklenen_esmalar:
                toplam_ebced += esma_obj["ebced"]
                bulunan_esmalar.append((esma_obj["isim"], esma_obj["ebced"], gezegen))
                eklenen_esmalar.add(esma_obj["isim"])
        
        # Diğer esmaları kontrol et
        for prefix in ["el-", "er-", "es-", "eş-", "en-", "ed-", "ez-", "et-"]:
            matches = re.finditer(f"{prefix}\\w+", user_input, re.IGNORECASE)
            for match in matches:
                esma_adi = match.group(0)
                esma_obj, gezegen = self.find_esma(esma_adi)
                if esma_obj and esma_obj["isim"] not in eklenen_esmalar:
                    toplam_ebced += esma_obj["ebced"]
                    bulunan_esmalar.append((esma_obj["isim"], esma_obj["ebced"], gezegen))
                    eklenen_esmalar.add(esma_obj["isim"])
        
        if not bulunan_esmalar:
            return "❌ Hiçbir esma bulunamadı. Lütfen geçerli esma isimleri girin."
        
        # Vefk hesaplama
        cikarma = derece_cikarma[derece]["cikarma"]
        artirma = derece_cikarma[derece]["artirma"]
        orijinal_ebced = toplam_ebced
        
        # Eğer çıkarma değeri ebced değerinden büyükse 361 ekle
        if cikarma > toplam_ebced:
            toplam_ebced += 361
        
        hesaplanan_deger = toplam_ebced - cikarma
        bolum = hesaplanan_deger // 3
        kalan = hesaplanan_deger % 3
        
        # Vefk matrisini oluştur
        vefk = [0] * 9
        for i in range(9):
            vefk[i] = bolum + (element_patterns[element][i] - 1) * artirma
        
        # Kalan değeri 7. haneye ekle
        if kalan > 0:
            vefk[6] += kalan
        
        # Vefk tablosunu oluştur
        vefk_table = f"🔮 {len(bulunan_esmalar)}li 3x3 {element.capitalize()} Vefki ({derece} derece)\n"
        vefk_table += "=" * 40 + "\n\n"
        
        # Kullanılan esmaların bilgilerini ekle
        vefk_table += "📝 Kullanılan Esmalar:\n"
        for i, (isim, ebced, gezegen) in enumerate(bulunan_esmalar, 1):
            vefk_table += f"{i}. {isim} (Ebced: {ebced}, Gezegen: {gezegen})\n"
        vefk_table += "\n"
        
        # Vefk tablosunu ekle
        vefk_table += "🎯 Vefk Tablosu:\n"
        for i in range(0, 9, 3):
            vefk_table += f"{vefk[i]}\t{vefk[i+1]}\t{vefk[i+2]}\n"
        vefk_table += "\n"
        
        # Hesaplama detaylarını ekle
        vefk_table += "📊 Hesaplama Detayları:\n"
        vefk_table += f"➊ Toplam Ebced Değeri: {orijinal_ebced}\n"
        if toplam_ebced > orijinal_ebced:
            vefk_table += f"➋ 361 Eklendi (Yeni Değer: {toplam_ebced})\n"
        vefk_table += f"➌ Çıkarma Değeri: {cikarma}\n"
        vefk_table += f"➍ Hesaplanan Değer: {hesaplanan_deger}\n"
        vefk_table += f"➎ Bölüm: {bolum}, Kalan: {kalan}\n"
        vefk_table += f"➏ Artırma Değeri: {artirma}\n"
        vefk_table += f"➐ Element: {element.capitalize()}"
        
        # Son vefk esmalarını güncelle
        self.son_vefk_esmalari = bulunan_esmalar
        
        return vefk_table
    
    def dortlu_vefk_hesapla(self, user_input):
        # Başlangıçta son vefk esmalarını temizle
        self.son_vefk_esmalari = []
        
        # Sabit vefk deseni (4x4 için)
        vefk_siralama = [
            8, 11, 14, 1,
            13, 2, 7, 12,
            3, 16, 9, 6,
            10, 5, 4, 15
        ]
        
        # Derece-eksiltme-artırma değerleri
        derece_cikarma = {
            30: {"cikarma": 30, "artirma": 1},
            60: {"cikarma": 60, "artirma": 2},
            90: {"cikarma": 90, "artirma": 3},
            120: {"cikarma": 120, "artirma": 4},
            150: {"cikarma": 150, "artirma": 5},
            180: {"cikarma": 180, "artirma": 6}
        }
        
        # Elementi bulun - doğrudan user_input içinde tam element ismini arayarak
        element = None
        if "toprak" in user_input.lower():
            element = "toprak"
        elif "ateş" in user_input.lower() or "ates" in user_input.lower():
            element = "ateş"
        elif "hava" in user_input.lower():
            element = "hava"
        elif "su" in user_input.lower():
            element = "su"
        
        if not element:
            element = "toprak"  # Varsayılan element
        
        # Dereceyi bulun
        derece = None
        for d in derece_cikarma.keys():
            if str(d) in user_input:
                derece = d
                break
        
        if not derece:
            derece = 30  # Varsayılan derece
        
        # Esmaları bul ve ebced değerlerini topla
        toplam_ebced = 0
        bulunan_esmalar = []
        
        # Benzersiz esmaları takip etmek için küme oluştur
        eklenen_esmalar = set()
        
        # Allah ve Zülcelal özel durumlarını kontrol et
        if "allah" in user_input.lower() or "zülcelal" in user_input.lower():
            for gezegen, esmalar in self.esma_data.items():
                for esma in esmalar:
                    if "allah" in esma["isim"].lower() and "allah" in user_input.lower() and esma["isim"] not in eklenen_esmalar:
                        toplam_ebced += esma["ebced"]
                        bulunan_esmalar.append((esma["isim"], esma["ebced"], gezegen))
                        eklenen_esmalar.add(esma["isim"])
                    elif "zülcelal" in esma["isim"].lower() and "zülcelal" in user_input.lower() and esma["isim"] not in eklenen_esmalar:
                        toplam_ebced += esma["ebced"]
                        bulunan_esmalar.append((esma["isim"], esma["ebced"], gezegen))
                        eklenen_esmalar.add(esma["isim"])
        
        # Ya ile başlayan ifadeleri kontrol et
        ya_matches = re.finditer(r'ya[\s-]([a-zçğıöşüâîû]+)', user_input, re.IGNORECASE)
        for ya_match in ya_matches:
            ya_esma = f"ya-{ya_match.group(1)}"
            esma_obj, gezegen = self.find_esma(ya_esma)
            if esma_obj and esma_obj["isim"] not in eklenen_esmalar:
                toplam_ebced += esma_obj["ebced"]
                bulunan_esmalar.append((esma_obj["isim"], esma_obj["ebced"], gezegen))
                eklenen_esmalar.add(esma_obj["isim"])
        
        # Diğer esmaları kontrol et
        for prefix in ["el-", "er-", "es-", "eş-", "en-", "ed-", "ez-", "et-"]:
            matches = re.finditer(f"{prefix}\\w+", user_input, re.IGNORECASE)
            for match in matches:
                esma_adi = match.group(0)
                esma_obj, gezegen = self.find_esma(esma_adi)
                if esma_obj and esma_obj["isim"] not in eklenen_esmalar:
                    toplam_ebced += esma_obj["ebced"]
                    bulunan_esmalar.append((esma_obj["isim"], esma_obj["ebced"], gezegen))
                    eklenen_esmalar.add(esma_obj["isim"])
        
        if not bulunan_esmalar:
            return "❌ Hiçbir esma bulunamadı. Lütfen geçerli esma isimleri girin."
        
        # Vefk hesaplama işlemi
        cikarma = derece_cikarma[derece]["cikarma"]
        artirma = derece_cikarma[derece]["artirma"]
        orijinal_ebced = toplam_ebced
        
        # Eğer çıkarma değeri ebced değerinden büyükse veya eşitse, 361 ekle
        if cikarma >= toplam_ebced:
            toplam_ebced += 361
        
        hesaplanan_deger = toplam_ebced - cikarma
        bolum = hesaplanan_deger // 4  # 4'e bölüm
        kalan = hesaplanan_deger % 4
        
        # Vefk matrisini oluştur
        vefk = [0] * 16  # 4x4 = 16 eleman
        for i in range(16):
            vefk[i] = bolum + (vefk_siralama[i] - 1) * artirma
        
        # Kalan değeri ekle
        if kalan == 1:
            vefk[12] += 1  # 13. haneye
        elif kalan == 2:
            vefk[8] += 1  # 9. haneye
        elif kalan == 3:
            vefk[4] += 1  # 5. haneye
        
        # Vefk tablosunu oluştur
        vefk_table = f"🔮 {len(bulunan_esmalar)}li 4x4 {element.capitalize()} Vefki ({derece} derece)\n"
        vefk_table += "=" * 40 + "\n\n"
        
        # Kullanılan esmaların bilgilerini ekle
        vefk_table += "📝 Kullanılan Esmalar:\n"
        for i, (isim, ebced, gezegen) in enumerate(bulunan_esmalar, 1):
            vefk_table += f"{i}. {isim} (Ebced: {ebced}, Gezegen: {gezegen})\n"
        vefk_table += "\n"
        
        # Vefk tablosunu ekle
        vefk_table += "🎯 Vefk Tablosu:\n"
        for i in range(0, 16, 4):
            vefk_table += f"{vefk[i]}\t{vefk[i+1]}\t{vefk[i+2]}\t{vefk[i+3]}\n"
        vefk_table += "\n"
        
        # Hesaplama detaylarını ekle
        vefk_table += "📊 Hesaplama Detayları:\n"
        vefk_table += f"➊ Toplam Ebced Değeri: {orijinal_ebced}\n"
        if toplam_ebced > orijinal_ebced:
            vefk_table += f"➋ 361 Eklendi (Yeni Değer: {toplam_ebced})\n"
        vefk_table += f"➌ Çıkarma Değeri: {cikarma}\n"
        vefk_table += f"➍ Hesaplanan Değer: {hesaplanan_deger}\n"
        vefk_table += f"➎ Bölüm: {bolum}, Kalan: {kalan}\n"
        vefk_table += f"➏ Artırma Değeri: {artirma}\n"
        vefk_table += f"➐ Element: {element.capitalize()}"
        
        # Son vefk esmalarını güncelle
        self.son_vefk_esmalari = bulunan_esmalar
        
        return vefk_table
    
    def besli_vefk_hesapla(self, user_input):
        # Başlangıçta son vefk esmalarını temizle
        self.son_vefk_esmalari = []
        
        # Sabit element desenleri
        element_patterns = {
            "toprak": [
                18, 10, 22, 14, 1,
                12, 4, 16, 8, 25,
                6, 23, 15, 2, 19,
                5, 17, 9, 21, 13,
                24, 11, 3, 20, 7
            ],
            "ateş": [
                1, 25, 19, 13, 7,
                14, 8, 2, 21, 20,
                22, 16, 15, 9, 3,
                10, 4, 23, 17, 11,
                18, 12, 6, 5, 24
            ],
            "hava": [
                7, 20, 3, 11, 24,
                13, 21, 9, 17, 5,
                19, 2, 25, 23, 6,
                25, 8, 16, 4, 12,
                1, 14, 22, 10, 18
            ],
            "su": [
                2, 7, 6, 9, 5, 
                8, 3, 2, 7, 9,
                4, 5, 1, 6, 8,
                6, 7, 8, 1, 2,
                5, 3, 4, 8, 6
            ]
        }
        
        # Derece-eksiltme-artırma değerleri
        derece_cikarma = {
            30: {"cikarma": 60, "artirma": 1},
            60: {"cikarma": 120, "artirma": 2},
            90: {"cikarma": 180, "artirma": 3},
            120: {"cikarma": 240, "artirma": 4},
            150: {"cikarma": 300, "artirma": 5},
            180: {"cikarma": 360, "artirma": 6}
        }
        
        # Elementi bulun - doğrudan user_input içinde tam element ismini arayarak
        element = None
        if "toprak" in user_input.lower():
            element = "toprak"
        elif "ateş" in user_input.lower() or "ates" in user_input.lower():
            element = "ateş"
        elif "hava" in user_input.lower():
            element = "hava"
        elif "su" in user_input.lower():
            element = "su"
        
        if not element:
            element = "toprak"  # Varsayılan element
        
        # Dereceyi bulun
        derece = None
        for d in derece_cikarma.keys():
            if str(d) in user_input:
                derece = d
                break
        
        if not derece:
            derece = 30  # Varsayılan derece
        
        # Esmaları bul ve ebced değerlerini topla
        toplam_ebced = 0
        bulunan_esmalar = []
        
        # Benzersiz esmaları takip etmek için küme oluştur
        eklenen_esmalar = set()
        
        # Allah ve Zülcelal özel durumlarını kontrol et
        if "allah" in user_input.lower() or "zülcelal" in user_input.lower():
            for gezegen, esmalar in self.esma_data.items():
                for esma in esmalar:
                    if "allah" in esma["isim"].lower() and "allah" in user_input.lower() and esma["isim"] not in eklenen_esmalar:
                        toplam_ebced += esma["ebced"]
                        bulunan_esmalar.append((esma["isim"], esma["ebced"], gezegen))
                        eklenen_esmalar.add(esma["isim"])
                    elif "zülcelal" in esma["isim"].lower() and "zülcelal" in user_input.lower() and esma["isim"] not in eklenen_esmalar:
                        toplam_ebced += esma["ebced"]
                        bulunan_esmalar.append((esma["isim"], esma["ebced"], gezegen))
                        eklenen_esmalar.add(esma["isim"])
        
        # Ya ile başlayan ifadeleri kontrol et
        ya_matches = re.finditer(r'ya[\s-]([a-zçğıöşüâîû]+)', user_input, re.IGNORECASE)
        for ya_match in ya_matches:
            ya_esma = f"ya-{ya_match.group(1)}"
            esma_obj, gezegen = self.find_esma(ya_esma)
            if esma_obj and esma_obj["isim"] not in eklenen_esmalar:
                toplam_ebced += esma_obj["ebced"]
                bulunan_esmalar.append((esma_obj["isim"], esma_obj["ebced"], gezegen))
                eklenen_esmalar.add(esma_obj["isim"])
        
        # Diğer esmaları kontrol et
        for prefix in ["el-", "er-", "es-", "eş-", "en-", "ed-", "ez-", "et-"]:
            matches = re.finditer(f"{prefix}\\w+", user_input, re.IGNORECASE)
            for match in matches:
                esma_adi = match.group(0)
                esma_obj, gezegen = self.find_esma(esma_adi)
                if esma_obj and esma_obj["isim"] not in eklenen_esmalar:
                    toplam_ebced += esma_obj["ebced"]
                    bulunan_esmalar.append((esma_obj["isim"], esma_obj["ebced"], gezegen))
                    eklenen_esmalar.add(esma_obj["isim"])
        
        if not bulunan_esmalar:
            return "❌ Hiçbir esma bulunamadı. Lütfen geçerli esma isimleri girin."
        
        # Vefk hesaplama
        cikarma = derece_cikarma[derece]["cikarma"]
        artirma = derece_cikarma[derece]["artirma"]
        orijinal_ebced = toplam_ebced
        
        # Eğer çıkarma değeri ebced değerinden büyükse veya eşitse 361 ekle
        if cikarma >= toplam_ebced:
            toplam_ebced += 361
        
        hesaplanan_deger = toplam_ebced - cikarma
        bolum = hesaplanan_deger // 5
        kalan = hesaplanan_deger % 5
        
        # Eğer bolum 0 ise veya negatifse, ebced değerine 361 ekle ve tekrar hesapla
        if bolum <= 0:
            toplam_ebced += 361
            hesaplanan_deger = toplam_ebced - cikarma
            bolum = hesaplanan_deger // 5
            kalan = hesaplanan_deger % 5
        
        # Element desenini al
        element_desen = element_patterns[element]
        
        # Vefk matrisini oluştur
        vefk = [0] * 25
        for i in range(25):
            vefk[i] = bolum + (element_desen[i] - 1) * artirma
        
        # Kalan değerleri ekle
        if kalan == 1:
            vefk[20] += 1  # 21. haneye
        elif kalan == 2:
            vefk[15] += 1  # 16. haneye
        elif kalan == 3:
            vefk[10] += 1  # 11. haneye
        elif kalan == 4:
            vefk[5] += 1   # 6. haneye
        
        # Vefk tablosunu oluştur
        vefk_table = f"🔮 {len(bulunan_esmalar)}li 5x5 {element.capitalize()} Vefki ({derece} derece)\n"
        vefk_table += "=" * 40 + "\n\n"
        
        # Kullanılan esmaların bilgilerini ekle
        vefk_table += "📝 Kullanılan Esmalar:\n"
        for i, (isim, ebced, gezegen) in enumerate(bulunan_esmalar, 1):
            vefk_table += f"{i}. {isim} (Ebced: {ebced}, Gezegen: {gezegen})\n"
        vefk_table += "\n"
        
        # Vefk tablosunu ekle
        vefk_table += "🎯 Vefk Tablosu:\n"
        for i in range(0, 25, 5):
            vefk_table += f"{vefk[i]}\t{vefk[i+1]}\t{vefk[i+2]}\t{vefk[i+3]}\t{vefk[i+4]}\n"
        vefk_table += "\n"
        
        # Hesaplama detaylarını ekle
        vefk_table += "📊 Hesaplama Detayları:\n"
        vefk_table += f"➊ Toplam Ebced Değeri: {orijinal_ebced}\n"
        if toplam_ebced > orijinal_ebced:
            vefk_table += f"➋ 361 Eklendi (Yeni Değer: {toplam_ebced})\n"
        vefk_table += f"➌ Çıkarma Değeri: {cikarma}\n"
        vefk_table += f"➍ Hesaplanan Değer: {hesaplanan_deger}\n"
        vefk_table += f"➎ Bölüm: {bolum}, Kalan: {kalan}\n"
        vefk_table += f"➏ Artırma Değeri: {artirma}\n"
        vefk_table += f"➐ Element: {element.capitalize()}"
        
        # Son vefk esmalarını güncelle
        self.son_vefk_esmalari = bulunan_esmalar
        
        return vefk_table
    
    def amac_icin_esma_oner(self, tema):
        uygun_esmalar = []
        tema_esma_map = {
            "koruma": ["el-Hafiz", "el-Muheymin", "el-Kavi", "el-Aziz", "el-Mumin", "el-Mevla", "ed-Dafi", "el-Muhit", "el-Mani", "el-Vekil", "el-Metin", "el-Kaahir", "el-Muheymin", "el-Karib"],
            "bereket": ["er-Rezzak", "el-Vehhab", "el-Basit", "el-Kerim", "el-Fettah", "el-Latif", "el-Kafi", "el-Mugni", "el-Gani", "el-Berr", "el-Ekrem", "el-Macid"],
            "şifa": ["eş-Şafi", "el-Latif", "er-Rauf", "er-Rahim", "er-Rahman", "ed-Dafi", "el-Muhyi", "el-Hayy", "el-Kafi", "el-Müstean", "el-Nafi", "el-Mevla"],
            "rızık": ["er-Rezzak", "el-Vehhab", "el-Basit", "el-Kerim", "el-Gani", "el-Fettah", "el-Kafi", "el-Mugni", "el-Varis", "el-Berr", "el-Ekrem", "el-Macid", "el-Mecid"],
            "güç": ["el-Kavi", "el-Aziz", "el-Cebbar", "el-Kahhar", "el-Muizz", "el-Kaahir", "el-Ala", "el-Metin", "el-Kadir", "el-Muktedir", "el-Gafir", "el-Hallak"],
            "ilim": ["el-Alim", "el-Hakim", "el-Habir", "el-Basir", "el-Vasi", "el-Alem", "el-Muhit", "el-Muhsi", "er-Reşid", "el-Hakem", "el-Fatır", "el-Falık"],
            "sevgi": ["el-Vedud", "er-Rauf", "el-Latif", "er-Rahim", "er-Rahman", "el-Karib", "el-Berr", "el-Gafur", "el-Afüvv", "eş-Şakir"],
            "zenginlik": ["el-Gani", "er-Rezzak", "el-Vehhab", "el-Kerim", "el-Basit", "el-Kafi", "el-Mugni", "el-Varis", "el-Berr", "el-Ekrem", "el-Macid", "el-Mecid"],
            "kahır": ["el-Kahhar", "el-Cebbar", "el-Muntakim", "el-Aziz", "el-Muzill", "el-Kaahir", "el-Kadir", "el-Muktedir", "ed-Darr", "el-Ala", "el-Gafir"],
            "başarı": ["el-Fettah", "el-Muizz", "el-Kavi", "el-Aziz", "el-Vasi", "en-Nasir", "el-Müstean", "el-Kadir", "el-Muktedir", "el-Ala", "el-Mevla"],
            "yaratma": ["el-Halik", "el-Bari", "el-Hallak", "el-Fatır", "el-Falık", "el-Mübdi", "el-Bais", "el-Muhyi", "el-Alem", "el-İlah"],
            "yardım": ["en-Nasir", "el-Müstean", "el-Mevla", "el-Kafi", "er-Rab", "el-Vekil", "el-Müstean", "el-Kadir", "el-Karib", "ed-Dafi"],
            "af": ["el-Gafir", "el-Gaffar", "er-Rahim", "el-Halim", "el-Latif", "el-Afüvv", "el-Tevvab", "eş-Şakir"],
            "dilek": ["el-Mucib", "el-Vehhab", "el-Kerim", "el-Gafur", "el-Afüvv", "el-Tevvab", "el-Müstean", "el-Kadir", "el-Muktedir", "el-Vekil", "el-Macid", "el-Mecid"],
            "hacet": ["el-Mucib", "el-Vehhab", "el-Kerim", "el-Gafur", "el-Afüvv", "el-Tevvab", "el-Müstean", "el-Kadir", "el-Muktedir", "el-Vekil", "el-Karib", "el-Mevla"]
        }
        
        if tema.lower() in tema_esma_map:
            for esma_adi in tema_esma_map[tema.lower()]:
                for gezegen, esmalar in self.esma_data.items():
                    for esma in esmalar:
                        if self.normalize_text(esma["isim"]) == self.normalize_text(esma_adi):
                            uygun_esmalar.append((esma["isim"], esma["tema"], esma["ebced"], gezegen))
        
        if not uygun_esmalar:
            return f"❌ {tema} için uygun esma bulunamadı."
        
        yanit = f"🔍 {tema.capitalize()} için önerilen esmalar:\n\n"
        for isim, tema, ebced, gezegen in uygun_esmalar:
            yanit += f"• {isim}\n"
            yanit += f"  Tema: {tema}\n"
            yanit += f"  Ebced: {ebced}\n"
            yanit += f"  Gezegen: {gezegen} ({self.gezegen_gun_eslesmeleri[gezegen]} günü)\n\n"
        
        if tema.lower() in ["dilek", "hacet"]:
            yanit += "💫 Özel Not: Bu esmalar dilek ve hacetlerin kabulü için özellikle tavsiye edilir. " + \
                    "Dilekleriniz için bu esmaları zikrederken samimi bir kalp ve güçlü bir inançla yaklaşmanız önemlidir."
        
        return yanit
    
    def esma_bilgisi_ver(self, esma_adi):
        for gezegen, esmalar in self.esma_data.items():
            for esma in esmalar:
                if esma['isim'].lower() == esma_adi.lower():
                    bilgi = f"Esma Bilgisi:\n\n"
                    bilgi += f"İsim: {esma['isim']}\n"
                    bilgi += f"Ebced Değeri: {esma['ebced']}\n"
                    bilgi += f"Tema: {esma['tema']}\n"
                    bilgi += f"Harf Sayısı: {esma['harf_sayisi']}\n"
                    bilgi += f"Gezegen Grubu: {gezegen} ({self.gezegen_gun_eslesmeleri[gezegen]} günü)\n"
                    return bilgi
        
        return f"{esma_adi} isimli esma bulunamadı."

    def coklu_davet_olustur(self, esma_isimleri):
        if len(esma_isimleri) > 7:
            return "❌ En fazla 7 esma seçebilirsiniz."

        cevap = "🌟 Çoklu Esma Daveti 🌟\n"
        cevap += "=" * 40 + "\n\n"
        
        bulunan_esmalar = []
        
        # Esmaları bul
        for esma_adi in esma_isimleri:
            esma, gezegen = self.find_esma(esma_adi)
            if esma:
                bulunan_esmalar.append((esma, gezegen))
        
        if not bulunan_esmalar:
            return "❌ Hiçbir esma bulunamadı. Lütfen geçerli esma isimleri girin."
        
        cevap += f"📝 Toplam {len(bulunan_esmalar)} esma için hesaplama yapılıyor...\n\n"
        
        # Toplam hesaplamalar için değişkenler
        toplam_hesaplama1 = 0
        toplam_hesaplama2 = 0
        toplam_hesaplama3 = 0
        toplam_hesaplama4 = 0
        toplam_hesaplama5 = 0
        
        # Her esma için ayrı hesaplama
        for i, (esma, gezegen) in enumerate(bulunan_esmalar, 1):
            ebced = esma['ebced']
            harf_sayisi = esma['harf_sayisi']
            gezegen_sayisi = self.gezegen_sayilari[gezegen]
            
            cevap += f"🔹 {i}. ESMA: {esma['isim']}\n"
            cevap += f"   Gezegen: {gezegen} ({self.gezegen_gun_eslesmeleri[gezegen]} günü) (Sayı: {gezegen_sayisi})\n"
            cevap += f"   Ebced: {ebced}, Harf Sayısı: {harf_sayisi}\n\n"
            
            # 5 farklı hesaplama
            hesaplama1 = ebced  # Kendi ebced değeri
            hesaplama2 = ebced * gezegen_sayisi  # Ebced × Gezegen sayısı
            hesaplama3 = ebced * harf_sayisi  # Ebced × Harf sayısı
            hesaplama4 = ebced * ebced  # Ebced × Ebced
            hesaplama5 = hesaplama1 + hesaplama2 + hesaplama3 + hesaplama4  # Toplam
            
            toplam_hesaplama1 += hesaplama1
            toplam_hesaplama2 += hesaplama2
            toplam_hesaplama3 += hesaplama3
            toplam_hesaplama4 += hesaplama4
            toplam_hesaplama5 += hesaplama5
            
            cevap += "   📊 Hesaplamalar:\n"
            cevap += f"   1. Ebced Değeri: {hesaplama1}\n"
            cevap += f"   2. Ebced × Gezegen Sayısı: {hesaplama2}\n"
            cevap += f"   3. Ebced × Harf Sayısı: {hesaplama3}\n"
            cevap += f"   4. Ebced × Ebced: {hesaplama4}\n"
            cevap += f"   5. Toplam: {hesaplama5}\n\n"
            
            cevap += "   " + "-" * 40 + "\n\n"
        
        # Toplam değerleri göster
        cevap += "📈 TOPLAM DEĞERLER:\n"
        cevap += f"1. Toplam Ebced: {toplam_hesaplama1}\n"
        cevap += f"2. Toplam (Ebced × Gezegen): {toplam_hesaplama2}\n"
        cevap += f"3. Toplam (Ebced × Harf): {toplam_hesaplama3}\n"
        cevap += f"4. Toplam (Ebced × Ebced): {toplam_hesaplama4}\n"
        cevap += f"5. Genel Toplam: {toplam_hesaplama5}\n\n"
        
        # Esma kombinasyonunu oluştur
        esma_kombinasyonu = ""
        if len(bulunan_esmalar) == 1:
            # Tek esma varsa, 'el-', 'er-' vb. ekleri kaldır
            esma_ismi = bulunan_esmalar[0][0]['isim']
            esma_ismi = esma_ismi.replace('el-', '').replace('er-', '').replace('es-', '').replace('eş-', '').replace('en-', '').replace('ed-', '').replace('ez-', '').replace('et-', '').replace('zü\'l-', '')
            esma_kombinasyonu = f"Ya {esma_ismi}"
        else:
            # Birden fazla esma varsa
            esma_isimleri = []
            for i, (esma, _) in enumerate(bulunan_esmalar):
                esma_ismi = esma['isim']
                # 'el-', 'er-' vb. ekleri kaldır
                esma_ismi = esma_ismi.replace('el-', '').replace('er-', '').replace('es-', '').replace('eş-', '').replace('en-', '').replace('ed-', '').replace('ez-', '').replace('et-', '').replace('zü\'l-', '')
                # Son esma hariç hepsine 'ul' ekle
                if i < len(bulunan_esmalar) - 1:
                    esma_isimleri.append(f"{esma_ismi}ul")
                else:
                    esma_isimleri.append(esma_ismi)
            
            esma_kombinasyonu = "Ya " + " ".join(esma_isimleri)
        
        # Toplam okuma sayısı hesaplama
        okuma_sayisi = toplam_hesaplama1 * len(bulunan_esmalar)
        
        cevap += "\n📌 DAVET TALİMATLARI:\n"
        cevap += f"1. Okuma Sayısı: {okuma_sayisi} (Toplam Ebced × Esma Sayısı)\n"
        cevap += f"2. Okuma: '{esma_kombinasyonu}' şeklinde okunmalı\n"
        cevap += "3. Her 100 de kasem(isimler x2 defa)okunur\n"
        cevap += "4. Zikir sırasında kıbleye dönük oturulmalı\n"
        cevap += "5. Zikre başlamadan önce abdest alınmalı\n"
        
        return cevap

    def davet_olustur_tema(self, tema=None):
        if not self.son_vefk_esmalari:
            return "❌ Önce bir vefk hesaplaması yapmalısınız."
        
        davet = "🌟 Vefk Daveti\n"
        davet += "=" * 40 + "\n\n"
        
        # Vefk türünü belirle
        vefk_turu = None
        if len(self.son_vefk_esmalari) == 1:
            vefk_turu = "3lü"
        elif len(self.son_vefk_esmalari) == 2:
            vefk_turu = "4lü"
        else:
            vefk_turu = "5li"
        
        # Kullanılan esmaların bilgilerini ekle
        davet += "📝 Kullanılan Esmalar:\n"
        for i, (isim, ebced, gezegen) in enumerate(self.son_vefk_esmalari, 1):
            davet += f"{i}. {isim} (Ebced: {ebced}, Gezegen: {gezegen} - {self.gezegen_gun_eslesmeleri[gezegen]} günü)\n"
        davet += "\n"
        
        # Vefk türüne göre özel pozisyonlar ve hesaplamalar
        if vefk_turu == "3lü":
            davet += "🎯 3'LÜ VEFK ÖZEL POZİSYONLARI:\n"
            for i, (isim, ebced, gezegen) in enumerate(self.son_vefk_esmalari, 1):
                gezegen_sayisi = self.gezegen_sayilari[gezegen]
                hesaplama1 = ebced * 3  # Vefk katsayısı
                hesaplama2 = ebced * gezegen_sayisi
                hesaplama3 = ebced * 9  # 3x3 matris
                
                davet += f"\n{isim} için:\n"
                davet += f"1. Doğu Pozisyonu: {hesaplama1} kere\n"
                davet += f"2. Güney Pozisyonu: {hesaplama2} kere\n"
                davet += f"3. Batı Pozisyonu: {hesaplama3} kere\n"
        
        elif vefk_turu == "4lü":
            davet += "🎯 4'LÜ VEFK ÖZEL POZİSYONLARI:\n"
            for i, (isim, ebced, gezegen) in enumerate(self.son_vefk_esmalari, 1):
                gezegen_sayisi = self.gezegen_sayilari[gezegen]
                
                # 4'lü vefk için özel hesaplamalar
                hesaplama1 = ebced * 4  # Vefk katsayısı
                hesaplama2 = ebced * gezegen_sayisi  # Gezegen etkisi
                hesaplama3 = ebced * 16  # 4x4 matris
                hesaplama4 = ebced * gezegen_sayisi * 4  # Özel kombinasyon
                hesaplama5 = hesaplama1 + hesaplama2 + hesaplama3 + hesaplama4  # Toplam etki
                
                davet += f"\n{isim} için:\n"
                davet += f"1. Kuzeydoğu Pozisyonu: {hesaplama1} kere → {self.ebced_to_isim(hesaplama1)}\n"
                davet += f"2. Güneydoğu Pozisyonu: {hesaplama2} kere → {self.ebced_to_isim(hesaplama2)}\n"
                davet += f"3. Güneybatı Pozisyonu: {hesaplama3} kere → {self.ebced_to_isim(hesaplama3)}\n"
                davet += f"4. Kuzeybatı Pozisyonu: {hesaplama4} kere → {self.ebced_to_isim(hesaplama4)}\n"
                davet += f"5. Merkez Pozisyonu: {hesaplama5} kere → {self.ebced_to_isim(hesaplama5)}\n"
                
                # Vefk köşe değerleri
                vefk_matrix, element = self.dort_vefk_olustur(f"{isim} için 4lü vefk", ebced)
                davet += f"\n📊 {element.capitalize()} Elementi Vefk Değerleri:\n"
                for row in vefk_matrix:
                    davet += " ".join(f"{num:4}" for num in row) + "\n"
                davet += "\n"
                
                # Her köşe için isim oluştur
                koseler = [
                    vefk_matrix[0][0],  # Kuzeydoğu
                    vefk_matrix[0][3],  # Güneydoğu
                    vefk_matrix[3][3],  # Güneybatı
                    vefk_matrix[3][0]   # Kuzeybatı
                ]
                
                davet += "🔮 Köşe İsimleri:\n"
                yonler = ["Kuzeydoğu", "Güneydoğu", "Güneybatı", "Kuzeybatı"]
                for j, (kose, yon) in enumerate(zip(koseler, yonler), 1):
                    davet += f"{j}. {yon}: {self.ebced_to_isim(kose)}\n"
                davet += "\n" + "-" * 40 + "\n\n"
        
        else:  # 5li vefk
            davet += "🎯 5'Lİ VEFK ÖZEL POZİSYONLARI:\n"
            for i, (isim, ebced, gezegen) in enumerate(self.son_vefk_esmalari, 1):
                gezegen_sayisi = self.gezegen_sayilari[gezegen]
                hesaplama1 = ebced * 5  # Vefk katsayısı
                hesaplama2 = ebced * gezegen_sayisi
                hesaplama3 = ebced * 25  # 5x5 matris
                hesaplama4 = ebced * 3  # Tek katsayı
                hesaplama5 = ebced * gezegen_sayisi * 5  # Özel kombinasyon
                
                davet += f"\n{isim} için:\n"
                davet += f"1. Merkez Pozisyonu: {hesaplama1} kere\n"
                davet += f"2. Doğu Pozisyonu: {hesaplama2} kere\n"
                davet += f"3. Güney Pozisyonu: {hesaplama3} kere\n"
                davet += f"4. Batı Pozisyonu: {hesaplama4} kere\n"
                davet += f"5. Kuzey Pozisyonu: {hesaplama5} kere\n"
        
        # Davet detaylarını ekle
        davet += "\n🕯️ DAVET DETAYLARI:\n"
        davet += "➊ Davet Günleri: "
        
        # Gezegen günlerini ekle
        gezegen_gunleri = {
            "Şems": "Pazar",
            "Kamer": "Pazartesi",
            "Mirrih": "Salı",
            "Utarid": "Çarşamba",
            "Müşteri": "Perşembe",
            "Zühre": "Cuma",
            "Zuhal": "Cumartesi"
        }
        
        gunler = []
        for _, _, gezegen in self.son_vefk_esmalari:
            if gezegen in gezegen_gunleri:
                gunler.append(gezegen_gunleri[gezegen])
        
        davet += ", ".join(sorted(set(gunler))) + "\n\n"
        
       
        
        # Genel talimatlar
        davet += "\n🌙 GENEL TALİMATLAR:\n"
        
        # Esma kombinasyonunu oluştur
        esma_kombinasyonu = ""
        if len(self.son_vefk_esmalari) == 1:
            # Tek esma varsa, 'el-', 'er-' vb. ekleri kaldır
            esma_ismi = self.son_vefk_esmalari[0][0]
            esma_ismi = esma_ismi.replace('el-', '').replace('er-', '').replace('es-', '').replace('eş-', '').replace('en-', '').replace('ed-', '').replace('ez-', '').replace('et-', '').replace('zü\'l-', '')
            esma_kombinasyonu = f"Ya {esma_ismi}"
        else:
            # Birden fazla esma varsa
            esma_isimleri = []
            for i, (isim, _, _) in enumerate(self.son_vefk_esmalari):
                # 'el-', 'er-' vb. ekleri kaldır
                esma_ismi = isim.replace('el-', '').replace('er-', '').replace('es-', '').replace('eş-', '').replace('en-', '').replace('ed-', '').replace('ez-', '').replace('et-', '').replace('zü\'l-', '')
                # Son esma hariç hepsine 'ul' ekle
                if i < len(self.son_vefk_esmalari) - 1:
                    esma_isimleri.append(f"{esma_ismi}ul")
                else:
                    esma_isimleri.append(esma_ismi)
            
            esma_kombinasyonu = "Ya " + " ".join(esma_isimleri)
        
        # Toplam ebced değeri ve okuma sayısı hesaplama
        toplam_ebced = sum(ebced for _, ebced, _ in self.son_vefk_esmalari)
        okuma_sayisi = toplam_ebced * len(self.son_vefk_esmalari)
        
        davet += f"• Okuma Sayısı: {okuma_sayisi} (Toplam Ebced × Esma Sayısı)\n"
        davet += f"• Okuma: '{esma_kombinasyonu}' şeklinde okunmalı\n"
        davet += "• Her 100 zikirde bir ara verilmeli\n"
        davet += "• Davet öncesi abdest alınmalı\n"
        davet += "• Temiz ve güzel kokulu kıyafetler giyilmeli\n"
        davet += "• Kıbleye yönelerek oturulmalı\n"
        
        return davet

    def load_and_train_model(self):
        try:
            # Örnek eğitim verileri
            conversations = [
                # Vefk hesaplama örnekleri
                {"input": "el-hafız için koruma vefki", "intent": "vefk_hesapla"},
                {"input": "el-rahman için 4lü vefk", "intent": "vefk_hesapla"},
                {"input": "el-melik için 5li vefk", "intent": "vefk_hesapla"},
                {"input": "el-aziz için 3lü vefk", "intent": "vefk_hesapla"},
                {"input": "el-vedud için 4lü vefk", "intent": "vefk_hesapla"},
                {"input": "el-basit için 5li vefk", "intent": "vefk_hesapla"},
                {"input": "koruma için 4lü vefk", "intent": "vefk_hesapla"},
                {"input": "bereket için 3lü vefk", "intent": "vefk_hesapla"},
                {"input": "rızık için 5li vefk", "intent": "vefk_hesapla"},
                
                # Esma önerme örnekleri
                {"input": "rızık için esma öner", "intent": "esma_oner"},
                {"input": "bereket için hangi esma", "intent": "esma_oner"},
                {"input": "koruma için esma", "intent": "esma_oner"},
                {"input": "şifa için esma", "intent": "esma_oner"},
                {"input": "zenginlik için esma", "intent": "esma_oner"},
                {"input": "başarı için esma", "intent": "esma_oner"},
                
                # Esma bilgi örnekleri
                {"input": "er-rahman esması nedir", "intent": "esma_bilgi"},
                {"input": "el-hafız ne demek", "intent": "esma_bilgi"},
                {"input": "el-vedud esması", "intent": "esma_bilgi"},
                {"input": "el-aziz anlamı", "intent": "esma_bilgi"},
                {"input": "el-basir özellikleri", "intent": "esma_bilgi"},
                
                # Davet hesaplama örnekleri
                {"input": "vefkin davetini ver", "intent": "davet_hesapla"},
                {"input": "davet oluştur", "intent": "davet_hesapla"},
                {"input": "davet hazırla", "intent": "davet_hesapla"},
                {"input": "daveti nasıl yapılır", "intent": "davet_hesapla"},
                {"input": "zikir sayısını hesapla", "intent": "davet_hesapla"}
            ]
            
            # Verileri hazırla ve modeli eğit
            texts, labels = self.nlp.prepare_training_data(conversations)
            
            # Eğitim parametrelerini güncelle
            training_args = TrainingArguments(
                output_dir="./vefk_model",
                num_train_epochs=5,  # Epoch sayısını artır
                per_device_train_batch_size=4,  # Batch size'ı artır
                per_device_eval_batch_size=4,
                warmup_steps=100,  # Warmup adımlarını artır
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=1,
                no_cuda=True,
                save_strategy="no",
                report_to="none",
                learning_rate=1e-4  # Öğrenme oranını düşür
            )
            
            self.nlp.train_model(texts, labels, epochs=5, batch_size=4)
            
            self.add_bot_message("Model başarıyla eğitildi! Artık sorularınızı yanıtlamaya hazırım.")
            
        except Exception as e:
            self.add_bot_message(f"Model eğitimi sırasında bir hata oluştu: {str(e)}")
            self.add_bot_message("Hata detayı: Lütfen tüm gerekli kütüphanelerin yüklü olduğundan emin olun.")

    def retrain_model(self):
        self.load_and_train_model()

    def process_input(self, event=None):
        user_text = self.user_input.get()
        if not user_text.strip():
            return
            
        self.add_user_message(user_text)
        self.user_input.delete(0, tk.END)
        
        # Esma listesini gösterme kontrolü
        if "esma listesi" in user_text.lower() or "esmaları listele" in user_text.lower() or "esma-ül hüsna listesi" in user_text.lower() or ("esmalar" in user_text.lower() and "listele" in user_text.lower()):
            response = self.esmalari_listele()
            self.add_bot_message(response)
            return
        
        # Davet oluşturma kontrolü
        if "davet" in user_text.lower():
            esma_isimleri = []
            if "," in user_text:
                # Virgülle ayrılmış esmalar
                for esma in user_text.split(","):
                    # Malikul Mülk için özel kontrol
                    if re.search(r'malik(ul)?[\s-]?m[uü]lk', esma.strip(), re.IGNORECASE):
                        esma_isimleri.append("Malikul-Mülk")
                    # Ya- ile başlayan esmalar için kontrol
                    elif re.search(r'ya[\s-]([a-zçğıöşüâîû]+)', esma.strip(), re.IGNORECASE):
                        match = re.search(r'ya[\s-]([a-zçğıöşüâîû]+)', esma.strip(), re.IGNORECASE)
                        if match:
                            esma_isimleri.append(f"ya-{match.group(1)}")
                    else:
                        match = re.search(r'(el-|er-|es-|eş-|en-|ed-|ez-|et-|allah|zü\'l-)\w+', esma.strip(), re.IGNORECASE)
                        if match:
                            esma_isimleri.append(match.group(0))
            else:
                # Malikul Mülk için özel kontrol
                if re.search(r'malik(ul)?[\s-]?m[uü]lk', user_text, re.IGNORECASE):
                    esma_isimleri.append("Malikul-Mülk")
                # Ya- ile başlayan esmalar için kontrol
                else:
                    # Ya- ile başlayan esmalar için kontrol
                    ya_matches = re.finditer(r'ya[\s-]([a-zçğıöşüâîû]+)', user_text, re.IGNORECASE)
                    for ya_match in ya_matches:
                        esma_isimleri.append(f"ya-{ya_match.group(1)}")
                    
                    # Standart esmalar için kontrol
                    if not esma_isimleri:  # Eğer "ya-" ile başlayan esma bulunamadıysa
                        matches = re.finditer(r'(el-|er-|es-|eş-|en-|ed-|ez-|et-|allah|zü\'l-)\w+', user_text, re.IGNORECASE)
                        esma_isimleri.extend(match.group(0) for match in matches)
                
            response = self.davet_olustur(esma_isimleri if esma_isimleri else None)
            self.add_bot_message(response)
            return
            
        # Ebced sorgulaması - ebced değerlerini doğrudan yakalama
        ebced_match1 = re.search(r'(\d+)\s*ebced', user_text.lower())
        ebced_match2 = re.search(r'ebced\s*(\d+)', user_text.lower())
        if ebced_match1 or ebced_match2:
            try:
                ebced_degeri = int(ebced_match1.group(1) if ebced_match1 else ebced_match2.group(1))
                response = self.ebced_to_esma(ebced_degeri)
                self.add_bot_message(response)
                return
            except Exception as e:
                self.add_bot_message(f"❌ Ebced değeri işlenirken bir hata oluştu: {str(e)}")
            return

        # NLP modeli ile kullanıcı niyetini belirle
        intent = self.nlp.predict_intent(user_text)
        
        # Niyete göre uygun yanıtı oluştur
        response = self.generate_response_by_intent(intent, user_text)
        self.add_bot_message(response)

    def generate_response_by_intent(self, intent, user_input):
        # Ebced sorgulaması - regex desenlerini güçlendirelim
        ebced_match1 = re.search(r'(\d+)\s*(?:ebced|ebçed|ebcet)', user_input.lower())
        ebced_match2 = re.search(r'(?:ebced|ebçed|ebcet)\s*(\d+)', user_input.lower())
        
        if ebced_match1 or ebced_match2:
            try:
                ebced_degeri = int(ebced_match1.group(1) if ebced_match1 else ebced_match2.group(1))
                return self.ebced_to_esma(ebced_degeri)
            except Exception as e:
                # Hata durumunda bilgilendirici mesaj
                print(f"Ebced değeri çözümlenirken hata: {str(e)}")
                return "❌ Ebced değeri işlenirken bir hata oluştu. Lütfen '100 ebced' veya 'ebced 100' formatında yazınız."
            
        # Vefk hesaplama - Vefk tipini doğru belirlemek için daha kesin kontroller ekleyelim
        input_lower = user_input.lower()
        
        # "Xli vefk" veya "X lü vefk" şeklindeki ifadeleri daha kesin yakala
        vefk_4lu_match = re.search(r'(4|dört|dörtlü|dortlu|4lü|4lu)\s*(?:li|lü|lu)?\s*vefk', input_lower)
        vefk_5li_match = re.search(r'(5|beş|beşli|besli|5li|5li)\s*(?:li|lü|lu)?\s*vefk', input_lower)
        vefk_3lu_match = re.search(r'(3|üç|üçlü|uclu|3lü|3lu)\s*(?:li|lü|lu)?\s*vefk', input_lower)
        
        if vefk_4lu_match:
            # 4lü vefk için kesin eşleşme var
            return self.dortlu_vefk_hesapla(user_input)
        elif vefk_5li_match:
            # 5li vefk için kesin eşleşme var
            return self.besli_vefk_hesapla(user_input)
        elif vefk_3lu_match:
            # 3lü vefk için kesin eşleşme var
            return self.uclu_vefk_hesapla(user_input)
        elif "vefk" in input_lower:
            # Vefk kelimesi geçiyor ama tipi belirtilmemiş, açıkça sor
            return "🔮 Hangi tip vefk hesaplamak istiyorsunuz? '3lü vefk', '4lü vefk' veya '5li vefk' şeklinde belirtiniz."
             
        # Yardım komutu
        if user_input.lower() == "yardım":
            return "🤖 Yardım Menüsü:\n\n" + \
                   "1. Vefk Hesaplama:\n" + \
                   "   - '3lü vefk' veya '3 lü vefk'\n" + \
                   "   - 'Seçili esmalar(, ile ayrılır birden fazla esmada) 3lü vefk 30,60,90,120,150 ve 180 derece toprak,ateş,su ve hava elementi '\n" + \
                   "   - '4lü vefk' veya '4 lü vefk'\n" + \
                   "   - 'Seçili esmalar(, ile ayrılır birden fazla esmada) 4lü vefk 30,60,90,120,150 ve 180 derece '\n" + \
                   "   - '5li vefk' veya '5 li vefk'\n\n" + \
                   "   - 'Seçili esmalar(, ile ayrılır birden fazla esmada) 5li vefk 30,60,90,120,150 ve 180 derece toprak,ateş,su ve hava elementi '\n" + \
                   "2. Esma Önerisi:\n" + \
                   "   - 'koruma için esma'\n" + \
                   "   - 'bereket için esma'\n" + \
                   "   - 'şifa için esma'\n" + \
                   "   - 'rızık için esma'\n" + \
                   "   - 'güç için esma'\n" + \
                   "   - 'ilim için esma'\n" + \
                   "   - 'sevgi için esma'\n" + \
                   "   - 'zenginlik için esma'\n" + \
                   "   - 'kahır için esma'\n" + \
                   "   - 'başarı için esma'\n" + \
                   "   - 'yaratma için esma'\n" + \
                   "   - 'yardım için esma'\n" + \
                   "   - 'af için esma'\n" + \
                   "   - 'dilek için esma'\n" + \
                   "   - 'hacet için esma'\n\n" + \
                   "3. Psikolojik ve Duygusal Destek:\n" + \
                   "   - 'depresyon için esma'\n" + \
                   "   - 'anksiyete için esma'\n" + \
                   "   - 'panik atak için esma'\n" + \
                   "   - 'stres için esma'\n" + \
                   "   - 'kaygı için esma'\n" + \
                   "   - 'yalnızlık için esma'\n" + \
                   "   - 'aşk acısı için esma'\n" + \
                   "   - 'kırgınlık için esma'\n" + \
                   "   - 'güven sorunu için esma'\n\n" + \
                   "4. Ailevi ve Sosyal Sorunlar:\n" + \
                   "   - 'ailevi sorun için esma'\n" + \
                   "   - 'geçimsizlik için esma'\n" + \
                   "   - 'miras için esma'\n" + \
                   "   - 'dışlanma için esma'\n" + \
                   "   - 'mobbing için esma'\n" + \
                   "   - 'taciz için esma'\n" + \
                   "   - 'istismar için esma'\n\n" + \
                   "5. İş ve Ekonomik Sorunlar:\n" + \
                   "   - 'işsizlik için esma'\n" + \
                   "   - 'maddi sıkıntı için esma'\n" + \
                   "   - 'borç için esma'\n" + \
                   "   - 'iflas için esma'\n" + \
                   "   - 'başarısızlık için esma'\n" + \
                   "   - 'kariyer için esma'\n\n" + \
                   "6. Manevi Sorunlar:\n" + \
                   "   - 'iman için esma'\n" + \
                   "   - 'inanç için esma'\n" + \
                   "   - 'ibadet için esma'\n" + \
                   "   - 'günah için esma'\n" + \
                   "   - 'kader için esma'\n" + \
                   "   - 'kısmet için esma'\n\n" + \
                   "7. Esma Bilgisi:\n" + \
                   "   - 'er-rahman esması nedir'\n" + \
                   "   - 'el-hafız ne demek'\n\n" + \
                   "8. Davet Hesaplama:\n" + \
                   "   - 'vefkin davetini ver'\n" + \
                   "   - 'davet oluştur'\n\n" + \
                   "9. Ebced Analizi:\n" + \
                   "   - '100 ebced'\n" + \
                   "   - 'ebced 100'\n\n" + \
                   "10. Genel Sorunlar:\n" + \
                   "    - 'hasta oldum'\n" + \
                   "    - 'moralim bozuk'\n" + \
                   "    - 'korkuyorum'\n" + \
                   "    - 'yardıma ihtiyacım var'\n" + \
                   "    - 'dua istiyorum'\n" + \
                   "    - 'vefk istiyorum'"
        
        # Sağlık/hastalık ile ilgili niyetleri algıla
        saglik_kelimeleri = [
            # Hastalık durumları
            "hasta", "hastalık", "rahatsız", "acı", "ağrı", "sancı", "sızı", "yanma", "kaşıntı",
            "ateş", "üşüme", "titreme", "halsiz", "yorgun", "bitkin", "dermansız", "güçsüz",
            "baş ağrısı", "baş dönmesi", "mide bulantısı", "kusma", "ishal", "kabız",
            "nezle", "grip", "öksürük", "burun akıntısı", "boğaz ağrısı",
            
            # Kronik durumlar
            "kronik", "sürekli", "devamlı", "geçmeyen", "iyileşmeyen", "tedavi", "ilaç",
            "doktor", "hastane", "muayene", "kontrol", "tahlil", "test", "teşhis",
            
            # Duygusal durumlar
            "moral", "motivasyon", "umut", "ümit", "moral bozukluğu", "depresyon", "stres",
            "kaygı", "endişe", "panik", "korku", "üzüntü", "hüzün",
            
            # Şifa ve iyileşme
            "şifa", "sağlık", "iyileşme", "iyileşmek", "düzelme", "düzelmek", "geçme", "geçmek",
            "kurtulma", "kurtulmak", "rahatlama", "rahatlamak", "ferahlama", "ferahlamak",
            
            # Alternatif tedaviler
            "dua", "zikir", "vefk", "ruhani", "manevi", "spiritüel", "enerji", "şifa duası",
            "okuma", "okumak", "yazma", "yazmak", "çizme", "çizmek",
            
            # Genel sağlık
            "sağlıklı", "zinde", "dinç", "enerjik", "güçlü", "kuvvetli", "dayanıklı",
            "bağışıklık", "direnç", "metabolizma", "vücut", "beden", "organ", "sistem"
        ]

        # Psikolojik sorunlar ve dertler için kelime listesi
        psikolojik_kelimeleri = [
            # Psikolojik rahatsızlıklar
            "depresyon", "anksiyete", "panik atak", "obsesif", "kompulsif", "fobi", "kaygı bozukluğu",
            "stres bozukluğu", "travma", "ptsd", "bipolar", "şizofreni", "psikoz", "manik",
            "duygu durum bozukluğu", "kişilik bozukluğu", "borderline", "narsist", "antisosyal",
            
            # Duygusal durumlar
            "mutsuz", "hüzünlü", "karamsar", "umutsuz", "çaresiz", "yorgun", "bitkin", "tükenmiş",
            "kaygılı", "endişeli", "korkulu", "panik", "gergin", "sinirli", "öfkeli", "huzursuz",
            "yalnız", "terk edilmiş", "değersiz", "başarısız", "yetersiz", "suçlu", "pişman",
            
            # İlişki sorunları
            "aşk acısı", "kırgın", "kırılmış", "alıngan", "kıskanç", "kızgın", "küs", "barışmak",
            "ayrılık", "boşanma", "terk edilme", "aldatılma", "güven sorunu", "iletişim sorunu",
            
            # Ailevi sorunlar
            "ailevi", "aile içi", "anne baba", "evlat", "kardeş", "akraba", "miras", "geçimsizlik",
            "tartışma", "kavga", "huzursuzluk", "anlaşmazlık", "çatışma", "uzaklaşma",
            
            # İş/okul sorunları
            "işsiz", "işsizlik", "başarısız", "başarısızlık", "sınav", "sınıfta kalma", "kariyer",
            "terfi", "maaş", "ekonomik", "maddi", "borç", "iflas", "batık", "kredi",
            
            # Sosyal sorunlar
            "yalnızlık", "sosyal fobi", "çekingen", "utangaç", "asosyal", "dışlanma", "alay",
            "dalga geçme", "mobbing", "psikolojik şiddet", "taciz", "istismar",
            
            # Manevi sorunlar
            "iman", "inanç", "ibadet", "dua", "zikir", "maneviyat", "ruhani", "spiritüel",
            "günah", "sevap", "kader", "kısmet", "nasip", "hayır", "şer", "musibet", "belâ"
        ]

        # Dertler için tema eşleştirmeleri
        dert_temalar = {
            # Psikolojik dertler
            "depresyon": "şifa",
            "anksiyete": "şifa",
            "panik atak": "şifa",
            "kaygı": "şifa",
            "stres": "şifa",
            "korku": "koruma",
            "yalnızlık": "sevgi",
            "aşk acısı": "sevgi",
            "kırgınlık": "af",
            "güven sorunu": "koruma",
            
            # Ailevi dertler
            "ailevi sorun": "yardım",
            "geçimsizlik": "yardım",
            "miras": "yardım",
            "huzursuzluk": "yardım",
            
            # İş/ekonomik dertler
            "işsizlik": "rızık",
            "maddi sıkıntı": "rızık",
            "borç": "rızık",
            "iflas": "rızık",
            "başarısızlık": "başarı",
            "kariyer": "başarı",
            
            # Sosyal dertler
            "dışlanma": "yardım",
            "mobbing": "koruma",
            "taciz": "koruma",
            "istismar": "koruma",
            
            # Manevi dertler
            "iman": "ilim",
            "inanç": "ilim",
            "ibadet": "ilim",
            "günah": "af",
            "kader": "yardım",
            "kısmet": "yardım",
            
            # Genel dertler
            "maddi sıkıntı": "rızık",
            "paraya ihtiyacım": "rızık",
            "fakirlik": "rızık",
            "borç": "rızık",
            "düşman": "kahır",
            "haset": "kahır",
            "kötülük": "kahır",
            "ilim": "ilim",
            "öğrenmek": "ilim",
            "bilgi": "ilim",
            "hastalık": "şifa",
            "sağlık": "şifa",
            "korunma": "koruma",
            "korku": "koruma",
            "sevgi": "sevgi",
            "aşk": "sevgi",
            "muhabbet": "sevgi",
            "başarı": "başarı",
            "zenginlik": "zenginlik",
            "bereket": "bereket",
            "yardım": "yardım",
            "af": "af",
            "dilek": "dilek",
            "hacet": "hacet"
        }

        # Psikolojik veya sağlık sorunlarını algıla
        if any(kelime in user_input.lower() for kelime in saglik_kelimeleri + psikolojik_kelimeleri):
            # Kullanıcının derdini analiz et
            for dert, tema in dert_temalar.items():
                if dert in user_input.lower():
                    esma_onerisi = self.amac_icin_esma_oner(tema)
                    vefk_olusturma = f"\n\n🌟 Vefk ve Davet Oluşturma:\n" + \
                                   f"Önerilen esmalardan birini veya birkaçını seçerek vefk ve davet oluşturabilirsiniz.\n" + \
                                   f"Örnek: 'eş-Şafi için 3lü vefk' veya 'eş-Şafi, el-Muhyi için davet'\n" + \
                                   f"Veya: 'vefkin davetini ver' (son yapılan vefk için)"
                    return esma_onerisi + vefk_olusturma
            
            # Eğer spesifik bir dert bulunamazsa genel şifa öner
            esma_onerisi = self.amac_icin_esma_oner("şifa")
            vefk_olusturma = f"\n\n🌟 Vefk ve Davet Oluşturma:\n" + \
                           f"Önerilen esmalardan birini veya birkaçını seçerek vefk ve davet oluşturabilirsiniz.\n" + \
                           f"Örnek: 'eş-Şafi için 3lü vefk' veya 'eş-Şafi, el-Muhyi için davet'\n" + \
                           f"Veya: 'vefkin davetini ver' (son yapılan vefk için)"
            return esma_onerisi + vefk_olusturma
        
        # Esma önerisi
        # Önce doğrudan tema kelimelerini kontrol et
        for tema in ["koruma", "rızık", "muhabbet", "sevgi", "kahır", "şifa", "bereket", "zenginlik", "başarı", "yaratma", "yardım", "af", "dilek", "hacet", "ilim", "güç"]:
            if tema in user_input.lower():
                # Değişiklik burada: "için esma" veya "esma öner" gibi çeşitli varyasyonları algıla
                if any(phrase in user_input.lower() for phrase in [f"{tema} için esma", f"{tema} için esmalar", f"{tema} esması", f"{tema} esmaları", 
                                                                  f"{tema} amaçlı esma", f"{tema} için önerilen esmalar", 
                                                                  f"{tema} için esma öner", f"öner {tema}", f"{tema} göster"]):
                    esma_onerisi = self.amac_icin_esma_oner(tema)
                    vefk_olusturma = f"\n\n🌟 Vefk ve Davet Oluşturma:\n" + \
                                   f"Önerilen esmalardan birini veya birkaçını seçerek vefk ve davet oluşturabilirsiniz.\n" + \
                                   f"Örnek: 'er-Rezzak için 3lü vefk' veya 'er-Rezzak, el-Vehhab için davet'\n" + \
                                   f"Veya: 'vefkin davetini ver' (son yapılan vefk için)"
                    return esma_onerisi + vefk_olusturma
        
        # Sonra eşdeğer kelimeleri kontrol et (bilgi -> ilim gibi)
        for dert, tema in dert_temalar.items():
            if dert in user_input.lower() and "için" in user_input.lower():
                if any(phrase in user_input.lower() for phrase in ["esma", "esmalar", "öner", "göster"]):
                    esma_onerisi = self.amac_icin_esma_oner(tema)
                    vefk_olusturma = f"\n\n🌟 Vefk ve Davet Oluşturma:\n" + \
                                   f"Önerilen esmalardan birini veya birkaçını seçerek vefk ve davet oluşturabilirsiniz.\n" + \
                                   f"Örnek: 'er-Rezzak için 3lü vefk' veya 'er-Rezzak, el-Vehhab için davet'\n" + \
                                   f"Veya: 'vefkin davetini ver' (son yapılan vefk için)"
                    return esma_onerisi + vefk_olusturma
        
        # Esma bilgisi
        esma_match = re.search(r'(el-|er-|es-|eş-|en-|ed-|ez-|et-|allah|zü\'l-)\w+', user_input, re.IGNORECASE)
        if esma_match and any(x in user_input.lower() for x in ["nedir", "ne demek", "anlamı", "özellikleri"]):
            return self.esma_bilgisi_ver(esma_match.group(0))
        
        # Davet hesaplama
        if "davet" in user_input.lower():
            if "vefkin davetini ver" in user_input.lower():
                return self.davet_olustur_tema()
            elif any(x in user_input.lower() for x in ["için davet", "daveti ver", "davet oluştur", "davet hazırla"]):
                esma_match = re.findall(r'(el-|er-|es-|eş-|en-|ed-|ez-|et-|allah|zü\'l-)\w+', user_input, re.IGNORECASE)
                if esma_match:
                    return self.coklu_davet_olustur(esma_match)
        
        return "❌ Anlayamadım. Lütfen yardım menüsünü görmek için 'yardım' yazın."

    def esmalari_listele(self):
        liste = "Esma-ül Hüsna Listesi:\n\n"
        for gezegen, esmalar in self.esma_data.items():
            liste += f"\n{gezegen} Grubu Esmaları ({self.gezegen_gun_eslesmeleri[gezegen]} günü):\n"
            liste += "─" * 40 + "\n"
            for esma in esmalar:
                liste += f"• {esma['isim']} (Ebced: {esma['ebced']}) - {esma['tema']}\n"
            liste += "\n"
        return liste

    def ebced_analizi(self, sayi):
        # Ebced değerinden esma bulma
        bulunan_esmalar = []
        for gezegen, esmalar in self.esma_data.items():
            for esma in esmalar:
                if esma["ebced"] == sayi:
                    bulunan_esmalar.append((esma["isim"], gezegen))
        
        if bulunan_esmalar:
            sonuc = f"({sayi} ebced değerine sahip esmalar: "
            for i, (isim, gezegen) in enumerate(bulunan_esmalar):
                if i > 0:
                    sonuc += ", "
                sonuc += f"{isim} ({gezegen})"
            sonuc += ")"
            return sonuc
        
        # Eğer tam eşleşme bulunamazsa, en yakın ebced değerlerine sahip esmaları bul
        yakin_esmalar = []
        for gezegen, esmalar in self.esma_data.items():
                for esma in esmalar:
                    fark = abs(esma["ebced"] - sayi)
                if fark <= 10:  # 10 birim farkla yakın kabul et
                    yakin_esmalar.append((esma["isim"], esma["ebced"], gezegen, fark))
        
        if yakin_esmalar:
            # Farka göre sırala
            yakin_esmalar.sort(key=lambda x: x[3])
            sonuc = f"({sayi} ebced değerine yakın esmalar: "
            for i, (isim, ebced, gezegen, fark) in enumerate(yakin_esmalar[:3]):  # En yakın 3 esmayı göster
                if i > 0:
                    sonuc += ", "
                sonuc += f"{isim} ({gezegen}, ebced: {ebced})"
            sonuc += ")"
            return sonuc
        
        return f"({sayi})"

    def ebced_to_esma(self, ebced_degeri):
        # Ebced değerinden esma bulma
        bulunan_esmalar = []
        
        # Diğer esmalar için normal arama
        for gezegen, esmalar in self.esma_data.items():
            for esma in esmalar:
                # Tam eşleşme veya yakın değer kontrolü
                if abs(esma["ebced"] - ebced_degeri) <= 10:  # 10 birim tolerans
                    bulunan_esmalar.append((esma, gezegen))
        
        if not bulunan_esmalar:
            return f"❌ {ebced_degeri} ebced değerine sahip veya yakın esma bulunamadı. Farklı bir değer deneyebilirsiniz."
        
        yanit = f"🔍 {ebced_degeri} ebced değerine sahip veya yakın esmalar:\n\n"
        for esma, gezegen in bulunan_esmalar:
            yanit += f"İsim: {esma['isim']}\n"
            yanit += f"Tema: {esma['tema']}\n"
            yanit += f"Ebced: {esma['ebced']}\n"
            yanit += f"Gezegen: {gezegen} ({self.gezegen_gun_eslesmeleri[gezegen]} günü)\n"
            yanit += f"Harf Sayısı: {esma['harf_sayisi']}\n"
            yanit += "---\n"
        
        yanit += "\nBu esmalarla vefk oluşturmak için: '[esma_ismi] için 3lü vefk' yazabilirsiniz."
        return yanit

    def dort_vefk_olustur(self, user_input, ebced_toplam):
        # Element desenleri
        element_patterns = {
            "ateş": [4, 14, 15, 1, 9, 7, 6, 12, 5, 11, 10, 8, 16, 2, 3, 13],
            "toprak": [15, 10, 3, 6, 4, 5, 16, 9, 14, 11, 2, 7, 1, 8, 13, 12],
            "hava": [7, 12, 1, 14, 2, 13, 8, 11, 16, 3, 10, 5, 9, 6, 15, 4],
            "su": [13, 3, 8, 10, 12, 6, 11, 5, 7, 15, 14, 2, 4, 9, 1, 16]
        }
        
        # Elementi bulun
        element = None
        for e in element_patterns.keys():
            if e in user_input:
                element = e
                break
        
        if not element:
            element = random.choice(list(element_patterns.keys()))
            
        # Vefk matrisini oluştur
        vefk = [[0 for _ in range(4)] for _ in range(4)]
        pattern = element_patterns[element]
        
        # Başlangıç sayısını hesapla
        start_num = ebced_toplam // 4
        remainder = ebced_toplam % 4
        
        # Artırma değerini hesapla
        increment = 1 if remainder > 0 else 0
        
        # Vefk'i doldur
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                pos = pattern[idx] - 1  # 1-tabanlıdan 0-tabanlıya çevir
                row = pos // 4
                col = pos % 4
                vefk[row][col] = start_num + idx * increment
        
        return vefk, element

    def ebced_to_isim(self, ebced_degeri):
        """Ebced değerini harflere çevirir"""
        sonuc = ""
        deger = ebced_degeri
        bin = 1000
        
        # Basamak değerlerini hesapla
        birler = deger % 10
        onlar = (deger % 100) - birler
        yuzler = (deger % 1000) - (birler + onlar)
        binler = ((deger % 10000) - (yuzler + onlar + birler)) // 1000
        onbinler = ((deger % 100000) - (deger % 10000)) // 1000
        yuzbinler = ((deger % 1000000) - (deger % 100000)) // 1000
        milyonlar = ((deger % 10000000) - (deger % 1000000)) // 1000000

        if deger < 100:
            sonuc = f"{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        elif deger < 1000:
            sonuc = f"{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        elif deger > 1000 and deger < 2000:
            sonuc = f"{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        elif deger > 2000 and deger < 10000:
            sonuc = f"{self.ucbasamak(self.ebced_harf(binler), True)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        elif deger > 10000 and deger < 100000:
            sonuc = f"{self.ucbasamak(self.ebced_harf(onbinler), True)}{self.ucbasamak(self.ebced_harf(binler), False)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        elif deger > 100000 and deger < 1000000:
            sonuc = f"{self.ucbasamak(self.ebced_harf(yuzbinler), True)}{self.ucbasamak(self.ebced_harf(onbinler), True)}{self.ucbasamak(self.ebced_harf(binler), False)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        elif deger > 1000000 and deger < 1002000:
            sonuc = f"{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(bin), False)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        elif deger > 1002000 and deger < 1010000:
            sonuc = f"{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(bin), False)}{self.ucbasamak(self.ebced_harf(yuzbinler), True)}{self.ucbasamak(self.ebced_harf(onbinler), True)}{self.ucbasamak(self.ebced_harf(binler), True)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        elif deger > 1010000 and deger < 2000000:
            sonuc = f"{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(bin), False)}{self.ucbasamak(self.ebced_harf(yuzbinler), True)}{self.ucbasamak(self.ebced_harf(onbinler), True)}{self.ucbasamak(self.ebced_harf(binler), False)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        elif deger > 2000000 and deger < 2002000:
            sonuc = f"{self.ucbasamak(self.ebced_harf(milyonlar), True)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(onbinler), True)}{self.ucbasamak(self.ebced_harf(binler), False)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        elif deger > 2002000 and deger < 2010000:
            sonuc = f"{self.ucbasamak(self.ebced_harf(milyonlar), True)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(bin), False)}{self.ucbasamak(self.ebced_harf(yuzbinler), True)}{self.ucbasamak(self.ebced_harf(onbinler), True)}{self.ucbasamak(self.ebced_harf(binler), True)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
        
        else:
            sonuc = f"{self.ucbasamak(self.ebced_harf(milyonlar), True)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(bin), False)}{self.ucbasamak(self.ebced_harf(yuzbinler), True)}{self.ucbasamak(self.ebced_harf(onbinler), True)}{self.ucbasamak(self.ebced_harf(binler), False)}{self.ucbasamak(self.ebced_harf(bin), True)}{self.ucbasamak(self.ebced_harf(yuzler), True)}{self.ucbasamak(self.ebced_harf(onlar), True)}{self.ucbasamak(self.ebced_harf(birler), False)}in"
            
        return sonuc

    def ucbasamak(self, input_str, iki_harf):
        """String işleme yardımcı fonksiyonu"""
        if not input_str:
            return ""
        
        if iki_harf and len(input_str) >= 2:
            return input_str[:2]
        return input_str[:1]

    def ebced_harf(self, rakam):
        """Rakamları Arapça harflere çevirir"""
        ebced_map = {
            1: "â", 2: "Be", 3: "Cim", 4: "Dal", 5: "He",
            6: "Vav", 7: "Ze", 8: "Ha", 9: "Tı", 10: "Ye",
            20: "Kef", 30: "Lam", 40: "Mim", 50: "Nun", 60: "Sin",
            70: "Ayn", 80: "Fe", 90: "Sad", 100: "Kaf", 200: "Re",
            300: "Şin", 400: "Te", 500: "Se", 600: "Hı", 700: "Zel",
            800: "Dad", 900: "Zı", 1000: "Ğayın"
        }
        return ebced_map.get(rakam, "")

    def davet_olustur(self, esma_isimleri=None):
        """Verilen esmalar veya son vefk esmalarından davet oluşturur"""
        if not esma_isimleri and not self.son_vefk_esmalari:
            return "❌ Lütfen esma isimleri girin veya önce bir vefk hesaplaması yapın."
            
        if not esma_isimleri:
            esma_isimleri = [esma[0] for esma in self.son_vefk_esmalari]
            
        if len(esma_isimleri) > 7:
            return "❌ En fazla 7 esma için davet oluşturulabilir."
            
        cevap = "🌟 Esma Daveti\n"
        cevap += "=" * 40 + "\n\n"
        
        bulunan_esmalar = []
        for esma_adi in esma_isimleri:
            for gezegen, esmalar in self.esma_data.items():
                for esma in esmalar:
                    if esma["isim"].lower() == esma_adi.lower():
                        bulunan_esmalar.append((esma, gezegen))
                        break
                        
        if not bulunan_esmalar:
            return "❌ Hiçbir esma bulunamadı."
        
        # Toplam ebced değeri hesaplama
        toplam_ebced = 0
        for esma, _ in bulunan_esmalar:
            toplam_ebced += esma["ebced"]
            
        # Toplam okuma sayısı hesaplama
        okuma_sayisi = toplam_ebced * len(bulunan_esmalar)
            
        for i, (esma, gezegen) in enumerate(bulunan_esmalar, 1):
            ebced = esma["ebced"]
            harf_sayisi = esma["harf_sayisi"]
            gezegen_sayisi = self.gezegen_sayilari[gezegen]
            
            # Hesaplamalar
            hesaplama1 = ebced
            hesaplama2 = ebced * gezegen_sayisi
            hesaplama3 = ebced * harf_sayisi
            hesaplama4 = ebced * ebced
            hesaplama5 = hesaplama1 + hesaplama2 + hesaplama3 + hesaplama4
            
            cevap += f"🔹 {i}. ESMA: {esma['isim']}\n"
            cevap += f"Gezegen: {gezegen} (Sayı: {gezegen_sayisi})\n"
            cevap += f"Ebced: {ebced}, Harf Sayısı: {harf_sayisi}\n\n"
            
            # Her hesaplama için isim oluştur
            cevap += "📝 Oluşturulan İsimler:\n"
            cevap += f"1. İsim: {self.ebced_to_isim(hesaplama1)}\n"
            cevap += f"2. İsim: {self.ebced_to_isim(hesaplama2)}\n"
            cevap += f"3. İsim: {self.ebced_to_isim(hesaplama3)}\n"
            cevap += f"4. İsim: {self.ebced_to_isim(hesaplama4)}\n"
            cevap += f"5. İsim: {self.ebced_to_isim(hesaplama5)}\n\n"
            cevap += "-" * 40 + "\n\n"
        
        # Esma kombinasyonunu oluştur
        esma_kombinasyonu = ""
        if len(bulunan_esmalar) == 1:
            # Tek esma varsa, tüm ön ekleri kaldır ('el-', 'er-', 'ya-' vb.)
            esma_ismi = bulunan_esmalar[0][0]['isim']
            esma_ismi = esma_ismi.replace('el-', '').replace('er-', '').replace('es-', '').replace('eş-', '').replace('en-', '').replace('ed-', '').replace('ez-', '').replace('et-', '').replace('zü\'l-', '')
            
            # "ya-" veya "ya " ön eklerini kaldır
            if esma_ismi.lower().startswith("ya-"):
                esma_ismi = esma_ismi[3:]
            elif esma_ismi.lower().startswith("ya "):
                esma_ismi = esma_ismi[3:]
                
            # Başa "Ya" ekle
            esma_kombinasyonu = f"Ya {esma_ismi}"
        else:
            # Birden fazla esma varsa
            esma_isimleri = []
            for i, esma in enumerate(bulunan_esmalar):
                esma_ismi = esma[0]['isim']
                # Tüm ön ekleri kaldır ('el-', 'er-', 'ya-' vb.)
                esma_ismi = esma_ismi.replace('el-', '').replace('er-', '').replace('es-', '').replace('eş-', '').replace('en-', '').replace('ed-', '').replace('ez-', '').replace('et-', '').replace('zü\'l-', '')
                
                # "ya-" veya "ya " ön eklerini kaldır
                if esma_ismi.lower().startswith("ya-"):
                    esma_ismi = esma_ismi[3:]
                elif esma_ismi.lower().startswith("ya "):
                    esma_ismi = esma_ismi[3:]
                
                # Son esma hariç hepsine 'ul' ekle
                if i < len(bulunan_esmalar) - 1:
                    esma_isimleri.append(f"{esma_ismi}ul")
                else:
                    esma_isimleri.append(esma_ismi)
            
            # Başa her zaman "Ya" ekle
            esma_kombinasyonu = "Ya " + " ".join(esma_isimleri)
        
        cevap += "\n📌 DAVET TALİMATLARI:\n"
        cevap += f"1. Okuma Sayısı: {okuma_sayisi} (Toplam Ebced × Esma Sayısı)\n"
        cevap += f"2. Okuma: '{esma_kombinasyonu}' şeklinde okunmalı\n"
        cevap += "3. Her 100 de kasem(isimler x2 defa)okunur\n"
        cevap += "4. Zikir sırasında kıbleye dönük oturulmalı\n"
        cevap += "5. Zikre başlamadan önce abdest alınmalı"
        
        return cevap

if __name__ == "__main__":
    root = tk.Tk()
    app = VefkEsmaUygulamasi(root)
    root.mainloop()