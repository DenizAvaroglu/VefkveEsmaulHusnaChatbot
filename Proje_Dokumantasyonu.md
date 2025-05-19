# Vefk ve Esma-ül Hüsna Hesaplama Uygulaması Dokümantasyonu

## İçindekiler
1. [Proje Yapısı](#proje-yapısı)
2. [Sınıf Yapısı](#sınıf-yapısı)
3. [Önemli Fonksiyonlar](#önemli-fonksiyonlar)
4. [Vefk Hesaplama Mantığı](#vefk-hesaplama-mantığı)
5. [Davet Oluşturma Süreci](#davet-oluşturma-süreci)
6. [Arayüz Bileşenleri](#arayüz-bileşenleri)

## Proje Yapısı

Bu proje, Vefk hesaplamaları ve Esma-ül Hüsna ile ilgili işlemleri gerçekleştiren bir uygulamadır. Projede kullanılan ana bileşenler:

- **vefk_chatbot.py**: Ana kod dosyası, tüm uygulama mantığını içerir
- **NLP Modeli**: Kullanıcı isteklerini anlamak için basit bir doğal dil işleme modeli
- **Vefk Hesaplamaları**: 3x3, 4x4 ve 5x5 Vefk matrislerinin hesaplanması
- **Esma-ül Hüsna Veri Tabanı**: 99 esma ve özellikleri

## Sınıf Yapısı

### VefkDataset ve VefkTransformer Sınıfları
Bu sınıflar, NLP modeli için veri yapısını ve model mimarisini tanımlar. Kullanıcının girdiği metinleri analiz edip, niyetini belirlemek için kullanılır.

### VefkNLP Sınıfı
Metin tabanlı kullanıcı girdilerini işleyerek, kullanıcının ne yapmak istediğini tahmin eden NLP sistemini içerir.

### VefkEsmaUygulamasi Sınıfı
Uygulamanın ana sınıfıdır ve şu temel özellikleri içerir:
- Kullanıcı arayüzü oluşturma
- Esma-ül Hüsna veritabanını yükleme
- Vefk hesaplama algoritmaları
- Davet (zikir) oluşturma
- Tema bazlı esma önerileri

## Önemli Fonksiyonlar

### 1. Vefk Hesaplama Fonksiyonları

- **uclu_vefk_hesapla**: 3x3 vefk matrisi oluşturur
```python
def uclu_vefk_hesapla(self, user_input):
    # Elementleri ve dereceleri belirler
    # Ebced değerlerini toplayarak vefk matrisini oluşturur
```

- **dortlu_vefk_hesapla**: 4x4 vefk matrisi oluşturur
```python
def dortlu_vefk_hesapla(self, user_input):
    # Sabit vefk deseni kullanır
    # Ebced değerlerine göre matris doldurur
```

- **besli_vefk_hesapla**: 5x5 vefk matrisi oluşturur
```python
def besli_vefk_hesapla(self, user_input):
    # Element bazlı desenleri kullanır
    # Toplam ebced değerine göre hesaplama yapar
```

### 2. Esma İşleme Fonksiyonları

- **find_esma**: Girilen isme göre esma bilgilerini bulur
```python
def find_esma(self, esma_adi):
    # Metin normalleştirme yapar
    # Esma veritabanında arama yapar
    # Bulunan esmayı ve gezegenini döndürür
```

- **esma_bilgisi_ver**: Bir esma hakkında detaylı bilgi verir
```python
def esma_bilgisi_ver(self, esma_adi):
    # Esmanın ebced değeri, tema, harf sayısı gibi bilgilerini içerir
```

- **tema_esma_oner**: Belirli bir amaç/tema için uygun esmaları önerir
```python
def tema_esma_oner(self, tema):
    # Her esmanın tema bilgisini kontrol eder
    # Benzerlik skorlarına göre en uygun esmaları listeler
```

### 3. Davet (Zikir) Fonksiyonları

- **davet_olustur**: Tek bir esma için davet talimatları oluşturur
```python
def davet_olustur(self, esma_isimleri=None):
    # Esma bilgilerini getirir
    # Ebced değerleri ve gezegen bilgilerini hesaplar
    # Zikir sayılarını ve örüntülerini oluşturur
```

- **coklu_davet_olustur**: Birden fazla esma için davet talimatları
```python
def coklu_davet_olustur(self, esma_isimleri):
    # Birden fazla esmanın kombinasyonunu hesaplar
    # Her esma için ayrı hesaplamalar yapar
    # Toplam değerleri oluşturur
```

- **davet_olustur_tema**: Vefk hesabı sonrası için davet talimatları
```python
def davet_olustur_tema(self, tema=None):
    # Son hesaplanan vefke göre özel pozisyonlar belirler
    # Vefk türüne (3lü, 4lü, 5li) göre yönergeler oluşturur
```

### 4. Kullanıcı Arayüzü Fonksiyonları

- **create_widgets**: Arayüz bileşenlerini oluşturur
```python
def create_widgets(self):
    # Sohbet alanı, giriş kutusu, butonlar vb. oluşturur
```

- **process_prompt**: Kullanıcı girdisini işler ve uygun yanıtı üretir
```python
def process_prompt(self, user_input):
    # Regex ile kullanıcı amacını tespit eder
    # İlgili fonksiyonu çağırır (vefk hesaplama, esma önerme vb.)
```

## Vefk Hesaplama Mantığı

Vefk hesaplamaları şu temel adımları içerir:

1. **Esma Tespiti**: Kullanıcının belirttiği esma isimleri tespit edilir
2. **Ebced Hesabı**: Her esmanın ebced değeri toplanır
3. **Matris Oluşturma**: 
   - 3x3 vefk için 9 hücreli matris
   - 4x4 vefk için 16 hücreli matris
   - 5x5 vefk için 25 hücreli matris
4. **Değer Hesaplama**: 
   - Toplam ebced değerinden çıkarma değeri çıkarılır
   - Sonuç, matris boyutuna bölünür ve her hücreye temel değer atanır
   - Artırma değeri ve element/vefk deseni kullanılarak her hücre hesaplanır

## Davet Oluşturma Süreci

Davet (zikir) hesaplamaları şu adımları içerir:

1. **Esma Kombinasyonu**: Birden fazla esma için "Ya X-ul Y-ul Z" formatında birleştirme
2. **Zikir Sayısı**: Toplam ebced değeri × esma sayısı formülü ile hesaplanır
3. **Talimatlar**: Abdest, kıbleye yönelme, ara verme gibi ritüel talimatları
4. **Pozisyonlar**: Vefk türüne göre özel pozisyonlar ve her pozisyon için zikir sayıları

## Arayüz Bileşenleri

Uygulama arayüzü şu bileşenlerden oluşur:

1. **Sohbet Alanı**: Kullanıcı ve bot mesajlarının görüntülendiği alan
2. **Giriş Kutusu**: Kullanıcının komut veya soru yazabildiği alan
3. **Gönder Butonu**: Mesajı iletmek için buton
4. **Model Eğitim Butonu**: NLP modelini yeniden eğitmek için buton

Arayüz, kullanıcı dostu ve basit bir tasarım sunarak, kompleks hesaplamaları anlaşılır şekilde kullanıcıya sunar. 