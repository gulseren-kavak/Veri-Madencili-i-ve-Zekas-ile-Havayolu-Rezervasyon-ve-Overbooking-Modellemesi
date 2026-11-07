# 📋 Uçuş Overbooking ve Kapasite Risk Yönetim Sistemi — Proje Raporu

**Ders:** Veri Madenciliği  
**Tarih:** Kasım 2026  

---

## 1. Analiz Raporu

### 1.1 Veri Kaynakları

| Veri Seti | Kaynak | Açıklama |
|-----------|--------|----------|
| `flights.csv` | Kaggle (2015 Flight Delays) | Uçuş bilgileri, gecikmeler, iptaller |
| `airports.csv` | Kaggle | Havalimanı konum bilgileri |
| `weather_data.csv` | NOAA tarzı sentetik | Hava durumu verileri |
| `events.csv` | Sentetik | Özel etkinlikler (maç, festival, tatil) |
| `reservations.csv` | Sentetik | Rezervasyon ve kapasite bilgileri |

> [!IMPORTANT]
> **Veri Zenginleştirme Notu:** Orijinal Kaggle uçuş veri setinde **uçak kapasitesi (CAPACITY)** ve **rezerve edilen yolcu sayısı (TICKETS_SOLD)** sütunları bulunmamaktadır. Bu sütunlar, overbooking analizi için zorunlu olduğundan, aşağıdaki yöntemle sentetik olarak türetilmiştir:
> - **CAPACITY:** Uçak tipine göre (B737=160, A321=220, B777=350 vb.) gerçekçi kapasite değerleri atanmıştır.
> - **TICKETS_SOLD:** Havayolunun tarihsel overbooking oranı, etkinlik etkisi, mevsimsel faktör ve haftanın günü dikkate alınarak hesaplanmıştır.
> - **BOOKING_RATIO:** TICKETS_SOLD / CAPACITY oranı olarak türetilmiştir.
> - **NO_SHOW_RATE:** Normal dağılımla (ortalama %5) simüle edilmiştir.
> - **EXPECTED_PASSENGERS:** TICKETS_SOLD × (1 - NO_SHOW_RATE) formülüyle hesaplanmıştır.

### 1.2 Keşifsel Veri Analizi (EDA)

- **Toplam uçuş:** 15.000 kayıt
- **Havayolu sayısı:** 10
- **Havalimanı sayısı:** 20
- **Overbooked uçuş:** 6.295 (%42.0)
- **Tarih aralığı:** 2026 tam yıl (365 gün)

**Overbooking'i Etkileyen Faktörler:**
1. **Booking Ratio** — En güçlü belirleyici (%90.4 önem)
2. **No-Show Rate** — İkinci en önemli (%8.8)
3. **Tickets Sold** — Üçüncü (%0.6)
4. **Etkinlikler** — Super Bowl, Thanksgiving gibi özel günlerde risk %95'e çıkar
5. **Hava Durumu** — Fırtına ve sis gecikme riskini artırır
6. **Mevsimsellik** — Yaz aylarında ve tatil dönemlerinde artış

### 1.3 Model Performansı

| Metrik | GradientBoosting | RandomForest |
|--------|-----------------|--------------|
| **ROC-AUC** | **0.9998** | 0.9989 |
| **Accuracy** | **%99.17** | %98.33 |
| **Precision** | **%98.89** | %97.86 |
| **Recall** | **%99.13** | %98.17 |
| **F1-Score** | **%99.01** | %98.02 |

**Seçilen Model:** GradientBoosting Classifier (200 ağaç, max_depth=5)

### 1.4 Maliyet Analizi

Overbooking gerçekleştiğinde havayoluna çıkan maliyetler:

| Maliyet Kalemi | Açıklama | Tahmini Birim Maliyet |
|----------------|----------|----------------------|
| Business yükseltme | Ekonomiden business'a upgrade | $200 – $600/yolcu |
| Otel konaklatma | 5 yıldızlı otel + transfer | $150 – $400/yolcu |
| Tazminat | AB EC 261/2004 kapsamında | $300 – $1.200/yolcu |
| Yemek/İkram | Bekleme süresince | $30 – $80/yolcu |

**Toplam Tahmini Yıllık Maliyet:** ~$195 milyon (15.000 uçuş bazında)

---

## 2. Müşteri Gereksinimi

### 2.1 Müşteri Profili
**Müşteri:** Havalimanı Yer Hizmetleri ve Havayolu Yolcu Hizmetleri Departmanı

### 2.2 İhtiyaçlar

| # | Gereksinim | Öncelik |
|---|-----------|---------|
| MG-01 | Sistem, her uçuş için overbooking riskini otomatik hesaplamalıdır | Yüksek |
| MG-02 | Riskli uçuşlar uçuş öncesinde (en az 2 saat) tespit edilmelidir | Yüksek |
| MG-03 | Operasyon ekibine Türkçe, anlaşılır uyarı mesajları iletilmelidir | Yüksek |
| MG-04 | Maliyet analizi ile potansiyel zarar önceden tahmin edilmelidir | Orta |
| MG-05 | Hava durumu ve özel etkinliklerin risk üzerindeki etkisi gösterilmelidir | Orta |
| MG-06 | Sistem web tabanlı dashboard üzerinden kullanılabilir olmalıdır | Yüksek |
| MG-07 | Tüm havayolları ve havalimanları için tek merkezden izleme yapılabilmelidir | Orta |
| MG-08 | Geçmiş verilere dayalı raporlama yapılabilmelidir | Düşük |

---

## 3. Fonksiyonel Gereksinimler

| # | Fonksiyon | Girdi | Çıktı | Açıklama |
|---|----------|-------|-------|----------|
| FR-01 | Risk Tahmini | Uçuş bilgileri (havayolu, rota, tarih, kapasite, satılan bilet) | Risk olasılığı (0-1), risk seviyesi (Düşük/Orta/Yüksek/Kritik) | ML modeli ile tahmin |
| FR-02 | Riskli Uçuş Listeleme | Eşik değer, limit | Filtrelenmiş uçuş listesi | Eşik üstü uçuşları sıralar |
| FR-03 | NLP Uyarı Üretimi | Riskli uçuş verisi | Türkçe doğal dil uyarı metni | Template-based NLG |
| FR-04 | Maliyet Hesaplama | Overbooked yolcu sayısı | Yükseltme, otel, tazminat maliyetleri | Birim maliyet × yolcu |
| FR-05 | Hava Durumu Etkisi | Havalimanı, tarih | Gecikme risk faktörü | Hava verisi entegrasyonu |
| FR-06 | Etkinlik Etkisi | Şehir, tarih | Etkinlik etki faktörü (0-1) | Etkinlik takvimi |
| FR-07 | Dashboard Görüntüleme | — | İstatistikler, grafikler, tablolar | Web arayüzü |
| FR-08 | Sistem Sağlık Kontrolü | — | API durumu, model bilgisi | /health endpoint |

---

## 4. Use Case Diyagramı

```
┌─────────────────────────────────────────────────────────────────┐
│                   OVERBOOKING RİSK YÖNETİM SİSTEMİ             │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐│
│  │  Dashboard Görme  │  │ Risk Tahmini Yap │  │ Uyarıları Görme  ││
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘│
│           │                     │                     │          │
│  ┌────────┴─────────┐  ┌───────┴──────────┐  ┌───────┴─────────┐│
│  │ Riskli Uçuşları  │  │ Maliyet Analizi  │  │ Hava Durumu     ││
│  │ Listeleme        │  │ Görüntüleme      │  │ Etkisi Görme    ││
│  └──────────────────┘  └──────────────────┘  └─────────────────┘│
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ Etkinlik Takvimi │  │ Ayarları         │                     │
│  │ Görüntüleme      │  │ Değiştirme       │                     │
│  └──────────────────┘  └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘

Aktörler:
  👤 Operasyon Personeli ──→ Dashboard, Riskli Uçuşlar, Uyarılar
  👤 Yönetici ──→ Maliyet Analizi, Dashboard, Raporlar
  🤖 ML Modeli ──→ Risk Tahmini, NLP Uyarı Üretimi
  🌐 Hava Durumu API ──→ Hava Durumu Etkisi
  📅 Etkinlik Veritabanı ──→ Etkinlik Etkisi
```

### Use Case Detayları

**UC-01: Risk Tahmini Yap**
- **Aktör:** Operasyon Personeli
- **Ön Koşul:** Sistem aktif, model eğitilmiş
- **Ana Akış:**
  1. Personel uçuş bilgilerini girer (havayolu, rota, tarih, kapasite, satılan bilet)
  2. Sistem ML modeli ile risk olasılığını hesaplar
  3. Risk seviyesi ve NLP uyarı metni gösterilir
  4. Maliyet tahmini hesaplanır
- **Son Koşul:** Risk raporu ekranda görüntülenir

**UC-02: Riskli Uçuşları İzle**
- **Aktör:** Operasyon Personeli, Yönetici
- **Ön Koşul:** Uçuş verileri yüklenmiş
- **Ana Akış:**
  1. Personel dashboard'u açar
  2. Sistem tüm uçuşları risk seviyesine göre sıralar
  3. Eşik değerin üzerindeki uçuşlar listelenir
  4. Her uçuş için detay görüntülenebilir
- **Son Koşul:** Riskli uçuş listesi görüntülenir

**UC-03: Maliyet Analizi Görüntüle**
- **Aktör:** Yönetici
- **Ön Koşul:** Rezervasyon verileri mevcut
- **Ana Akış:**
  1. Yönetici maliyet analizi sayfasını açar
  2. Havayoluna göre maliyet dağılımı gösterilir
  3. Yükseltme, otel, tazminat kalemleri ayrı ayrı listelenir
- **Son Koşul:** Maliyet raporu görüntülenir

---

## 5. Kabul Kriterleri

| # | Kriter | Doğrulama Yöntemi | Durum |
|---|--------|-------------------|-------|
| AK-01 | Model ROC-AUC değeri ≥ 0.90 olmalıdır | Model eğitim çıktısı | ✅ 0.9998 |
| AK-02 | API 5 saniye içinde yanıt vermelidir | /health endpoint testi | ✅ <1s |
| AK-03 | Risk tahmini 0-1 arası olasılık döndürmelidir | /predict endpoint testi | ✅ |
| AK-04 | NLP uyarıları Türkçe ve anlaşılır olmalıdır | Manuel inceleme | ✅ |
| AK-05 | Dashboard tüm bölümleri yükleyebilmelidir | Tarayıcı testi | ✅ |
| AK-06 | Maliyet analizi havayoluna göre gruplandırılmalıdır | /cost_analysis testi | ✅ |
| AK-07 | Hava durumu verisi risk faktörünü etkilemelidir | Feature importance | ✅ |
| AK-08 | Etkinlik verisi overbooking riskini artırmalıdır | Veri analizi | ✅ |
| AK-09 | Sistem en az 10.000 uçuşu işleyebilmelidir | Performans testi | ✅ 15.000 |
| AK-10 | Tüm API endpoint'leri CORS desteklemelidir | Tarayıcı testi | ✅ |

---

## 6. Test Case'ler

### TC-01: Sistem Sağlık Kontrolü
| Alan | Değer |
|------|-------|
| **Amaç** | API'nin çalıştığını doğrulamak |
| **Ön Koşul** | Sunucu başlatılmış |
| **Adımlar** | 1. GET /health isteği gönder |
| **Beklenen Sonuç** | `{"status":"ok", "threshold":0.5, "model":"GradientBoosting"}` |
| **Gerçek Sonuç** | ✅ Başarılı |

### TC-02: Risk Tahmini — Yüksek Risk
| Alan | Değer |
|------|-------|
| **Amaç** | Yüksek riskli uçuşun doğru tespit edilmesi |
| **Ön Koşul** | Model eğitilmiş |
| **Adımlar** | 1. POST /predict — capacity:180, tickets_sold:210, event_impact:0.9 |
| **Beklenen Sonuç** | risk_probability > 0.7, high_risk: true |
| **Gerçek Sonuç** | ✅ Başarılı |

### TC-03: Risk Tahmini — Düşük Risk
| Alan | Değer |
|------|-------|
| **Amaç** | Normal uçuşun düşük risk olarak sınıflandırılması |
| **Ön Koşul** | Model eğitilmiş |
| **Adımlar** | 1. POST /predict — capacity:220, tickets_sold:180, event_impact:0 |
| **Beklenen Sonuç** | risk_probability < 0.5, high_risk: false |
| **Gerçek Sonuç** | ✅ Başarılı |

### TC-04: Riskli Uçuş Listeleme
| Alan | Değer |
|------|-------|
| **Amaç** | Riskli uçuşların doğru listelenmesi |
| **Ön Koşul** | Uçuş verileri yüklenmiş |
| **Adımlar** | 1. GET /risky_flights?limit=10 |
| **Beklenen Sonuç** | En fazla 10 uçuş, booking_ratio > 1.0, sıralı |
| **Gerçek Sonuç** | ✅ Başarılı |

### TC-05: NLP Uyarı Üretimi
| Alan | Değer |
|------|-------|
| **Amaç** | Türkçe uyarı metninin doğru üretilmesi |
| **Ön Koşul** | Model eğitilmiş, riskli uçuş mevcut |
| **Adımlar** | 1. GET /alerts?limit=5 |
| **Beklenen Sonuç** | Her uyarıda: icon, level, message (Türkçe), risk_percent |
| **Gerçek Sonuç** | ✅ Başarılı |

### TC-06: Maliyet Analizi
| Alan | Değer |
|------|-------|
| **Amaç** | Maliyet hesaplamasının doğruluğu |
| **Ön Koşul** | Rezervasyon verileri mevcut |
| **Adımlar** | 1. GET /cost_analysis |
| **Beklenen Sonuç** | total_upgrade, total_hotel, total_compensation > 0, by_airline listesi |
| **Gerçek Sonuç** | ✅ Başarılı |

### TC-07: Hava Durumu Etkisi
| Alan | Değer |
|------|-------|
| **Amaç** | Hava koşullarının gecikme riskine etkisinin gösterilmesi |
| **Adımlar** | 1. GET /weather_impact |
| **Beklenen Sonuç** | by_risk_level ve by_condition verileri dolu |
| **Gerçek Sonuç** | ✅ Başarılı |

### TC-08: Etkinlik Verisi
| Alan | Değer |
|------|-------|
| **Amaç** | Etkinlik takviminin yüklenmesi |
| **Adımlar** | 1. GET /events |
| **Beklenen Sonuç** | En az 20 farklı etkinlik, impact_factor alanı mevcut |
| **Gerçek Sonuç** | ✅ Başarılı |

### TC-09: Dashboard Yükleme
| Alan | Değer |
|------|-------|
| **Amaç** | Web arayüzünün tam yüklenmesi |
| **Adımlar** | 1. Tarayıcıda http://localhost:8001 aç 2. Tüm sayfalara tıkla |
| **Beklenen Sonuç** | 8 sayfa hatasız yüklenmeli |
| **Gerçek Sonuç** | ✅ Başarılı |

### TC-10: Etkinlik Etkisi Doğrulama
| Alan | Değer |
|------|-------|
| **Amaç** | Thanksgiving gibi büyük etkinliklerde risk artışının doğrulanması |
| **Adımlar** | 1. event_impact=0.95 ile predict çağır 2. event_impact=0 ile predict çağır |
| **Beklenen Sonuç** | Etkinlikli tahminde risk belirgin şekilde daha yüksek |
| **Gerçek Sonuç** | ✅ Başarılı |

---

## 7. Teknik Mimari

```
┌─────────────┐    HTTP/JSON    ┌──────────────────┐
│  Web        │ ◄────────────► │  Python Backend   │
│  Dashboard  │                │  (HTTP Server)    │
│  (HTML/JS)  │                │                   │
└─────────────┘                │  ┌──────────────┐ │
                               │  │ ML Model     │ │
                               │  │ (Gradient    │ │
                               │  │  Boosting)   │ │
                               │  └──────────────┘ │
                               │  ┌──────────────┐ │
                               │  │ NLP Engine   │ │
                               │  │ (Template    │ │
                               │  │  NLG)        │ │
                               │  └──────────────┘ │
                               └────────┬─────────┘
                                        │
                               ┌────────┴─────────┐
                               │   Veri Katmanı    │
                               │  flights.csv      │
                               │  airports.csv     │
                               │  weather_data.csv │
                               │  events.csv       │
                               │  reservations.csv │
                               └──────────────────┘
```

---

## 8. Kullanılan Teknolojiler

| Teknoloji | Kullanım Alanı |
|-----------|---------------|
| Python 3.12 | Backend, ML, veri işleme |
| pandas | Veri manipülasyonu |
| scikit-learn | Makine öğrenmesi (GradientBoosting, RandomForest) |
| joblib | Model serileştirme |
| HTML5 / CSS3 / JavaScript | Frontend dashboard |
| HTTP Server (stdlib) | API sunucusu |

---

## 9. Veri Zenginleştirme Açıklaması (Rapor İçin Önemli)

Orijinal Kaggle veri setinde **bulunmayan** ancak overbooking analizi için **zorunlu olan** sütunlar şunlardır:

| Türetilen Sütun | Nasıl Üretildi | Neden Gerekli |
|-----------------|----------------|---------------|
| `CAPACITY` | Uçak tipine göre atandı (B737=160, A321=220 vb.) | Overbooking = yolcu > kapasite |
| `TICKETS_SOLD` | Kapasite × booking_ratio (havayolu overbooking oranı + etkinlik + mevsim etkisi) | Satılan bilet sayısı olmadan risk hesaplanamaz |
| `BOOKING_RATIO` | TICKETS_SOLD / CAPACITY | Ana risk göstergesi |
| `NO_SHOW_RATE` | Normal dağılım N(0.05, 0.02) | Gerçek binen yolcu tahmini için |
| `EXPECTED_PASSENGERS` | TICKETS_SOLD × (1 - NO_SHOW_RATE) | Gerçek doluluk tahmini |
| `HAS_EVENT` | Etkinlik takviminden (tarih + şehir eşleşmesi) | Etkinlik etkisi |
| `EVENT_IMPACT` | Etkinlik büyüklüğüne göre (0-1) | Risk artış faktörü |
| `WEATHER_DELAY_RISK` | Hava durumu verilerinden (yağış, rüzgar, görüş) | Gecikme riski |
| `SEASONAL_FACTOR` | Sinüs fonksiyonu (yaz=yüksek, kış=düşük) | Mevsimsel talep |

> Bu türetme işlemi `create_dummy_data.py` dosyasında detaylı şekilde kodlanmıştır.

