# Vefk ve Esma-ül Hüsna Chatbot Projesi

Bu proje, İslami literatürde önemli bir yere sahip olan Esma-ül Hüsna ve Vefk hesaplamalarını otomatize eden bir chatbot uygulamasıdır. Python programlama dili kullanılarak geliştirilmiş olup, modern doğal dil işleme teknikleri ve makine öğrenmesi yöntemleri entegre edilmiştir.

## Proje Özellikleri

- **Esma-ül Hüsna Veritabanı**: 99 temel Esma + 19 ek Esma (toplam 118 Esma) içeren kapsamlı bir veritabanı
- **Vefk Hesaplamaları**: 3'lü (3x3), 4'lü (4x4) ve 5'li (5x5) vefk matrislerinin otomatik hesaplanması
- **Doğal Dil İşleme**: BERT tabanlı NLP modeli ile kullanıcı sorularının anlaşılması
- **Ebced Analizi**: Harflerin ebced değerlerinin hesaplanması ve analizi
- **Tema Bazlı Esma Önerileri**: Koruma, şifa, bereket gibi farklı temalar için özel esma önerileri
- **Davet Talimatları**: Vefk hesaplamalarına göre özel davet ve zikir talimatları
- **Kullanıcı Dostu Arayüz**: Tkinter ile geliştirilmiş sezgisel kullanıcı arayüzü

## Kurulum

### Gereksinimler

- Python 3.9 veya üzeri
- PyTorch 1.10.0 veya üzeri
- Transformers 4.15.0 veya üzeri
- NLTK 3.6.5 veya üzeri
- Tkinter (genellikle Python ile birlikte gelir)

### Kurulum Adımları

1. Projeyi klonlayın:
   ```bash
   git clone https://github.com/kullaniciadi/vefk-esma-chatbot.git
   cd vefk-esma-chatbot
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. NLTK veri paketlerini indirin:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

4. Uygulamayı başlatın:
   ```bash
   python main.py
   ```

## Kullanım

### Vefk Hesaplama

Uygulama, kullanıcının doğal dille ifade ettiği vefk hesaplama taleplerini anlayabilmektedir:

- "el-Rahman için 3lü vefk hesapla"
- "4lü Hafız vefki oluştur"
- "5li ateş vefki el-Vedud için"

### Esma Bilgisi Sorgulama

Belirli bir esma hakkında bilgi almak için:

- "el-Vedud nedir?"
- "Ya-Rahman hakkında bilgi ver"
- "el-Hafız esmasının özellikleri neler?"

### Amaca Yönelik Esma Önerisi

Belirli bir amaç için uygun esmaları öğrenmek için:

- "Koruma için hangi esmalar kullanılır?"
- "Şifa için esma öner"
- "Rızık için hangi esmalar okunmalı?"

### Ebced Analizi

Ebced değerine göre esma analizi için:

- "329 ebced değerine sahip esmalar"
- "66 ebced değeri analiz et"

### Davet Talimatları

Bir vefk için özel davet talimatları almak için:

- "el-Hafız vefki için davet oluştur"
- "Koruma vefki için zikir talimatları"

## Proje Yapısı

```
vefk-esma-chatbot/
├── main.py                  # Ana uygulama başlatma dosyası
├── vefk_nlp.py              # NLP modeli ve işlevleri
├── vefk_hesaplama.py        # Vefk hesaplama algoritmaları
├── esma_veritabani.py       # Esma veritabanı ve işlevleri
├── arayuz.py                # Tkinter arayüz kodları
├── model/                   # Eğitilmiş model dosyaları
├── data/                    # Veri dosyaları
│   ├── esmalar.json         # Esma verileri
│   ├── vefk_desenleri.json  # Vefk element desenleri
│   └── egitim_verileri.json # Model eğitim verileri
├── utils/                   # Yardımcı işlevler
└── docs/                    # Dokümantasyon
```

## Teknik Detaylar

### Model Mimarisi

Proje, BERT (Bidirectional Encoder Representations from Transformers) tabanlı bir doğal dil işleme modeli kullanmaktadır. Türkçe dil desteği için "dbmdz/bert-base-turkish-cased" ön eğitimli modeli kullanılmıştır.

Model, kullanıcı sorguları üzerinde şu niyet kategorilerini tespit edebilmektedir:
- vefk_hesapla
- esma_oner
- esma_bilgi
- davet_hesapla
- ebced_analiz

### Vefk Hesaplama Algoritması

Vefk hesaplamaları, geleneksel İslami kaynaklarda belirtilen yöntemlere uygun olarak geliştirilmiştir:

1. Esma tespiti ve ebced değeri hesaplama
2. Element ve derece parametrelerini belirleme
3. Temel değer hesaplama: Ebced - (Derece Çıkarma Değeri)
4. Bölüm ve kalan hesaplama: Temel Değer / 9
5. Element desenine göre matris oluşturma
6. Hücre değerlerini hesaplama: Bölüm + (Artırma * (Desen Değeri <= Kalan))

## Katkıda Bulunma

Projeye katkıda bulunmak için lütfen bir Pull Request oluşturun veya önerilerinizi [Issues](https://github.com/kullaniciadi/vefk-esma-chatbot/issues) bölümünde paylaşın.

Özellikle şu alanlarda katkılar beklenmektedir:
- Esma veritabanının genişletilmesi
- Yeni vefk tiplerinin eklenmesi
- NLP modelinin iyileştirilmesi
- Arayüz geliştirmeleri
- Farklı dil destekleri eklenmesi

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

## İletişim

Sorularınız ve önerileriniz için [email@example.com](mailto:email@example.com) adresine e-posta gönderebilirsiniz.

## Teşekkür

- [Transformers](https://github.com/huggingface/transformers) kütüphanesi için Hugging Face ekibine
- BERT modelini geliştiren Google AI ekibine
- Türkçe BERT modelini geliştiren Digitale Bibliothek München ekibine
- İslami kaynaklar ve vefk hesaplamaları konusunda yardımcı olan uzmanlara 