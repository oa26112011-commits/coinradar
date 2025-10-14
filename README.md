"""
# Fixed Trading Bot v2.0 - Deployment Guide

## 🚀 Render.com'a Deploy Etme

### 1. Render.com Hesabı
1. https://render.com adresine gidin
2. GitHub hesabınızla giriş yapın
3. Ücretsiz plan seçin

### 2. Repository Hazırlama
```bash
# Dosyaları organize edin
project/
├── akilli_tarayici_bot.py  (ana kod)
├── requirements.txt
├── render.yaml
├── runtime.txt
├── .env (sadece local)
└── .gitignore
```

### 3. GitHub'a Push
```bash
git init
git add .
git commit -m "Initial commit - Fixed Trading Bot v2.0"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

### 4. Render'da Deploy
1. Render Dashboard → "New" → "Blueprint"
2. GitHub repository'nizi seçin
3. `render.yaml` otomatik algılanacak
4. "Apply" butonuna basın
5. Deploy başlayacak (2-3 dakika)

### 5. Environment Variables Kontrolü
Render Dashboard'da:
- Settings → Environment
- Tüm değişkenlerin doğru olduğunu kontrol edin
- Telegram token'ı ve chat ID'yi doğrulayın

### 6. Health Check
Deploy tamamlandıktan sonra:
```bash
curl https://YOUR-APP-NAME.onrender.com/health
```

Yanıt:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-10T12:00:00",
  "strategy": "Fixed v2.0 - 45m Resample + Wilder RSI + Kademeli Ceza"
}
```

## 🔧 Local Test

### Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Locally
```bash
python akilli_tarayici_bot.py
```

## 📊 Monitoring

### Render Dashboard
- Logs: Real-time log stream
- Metrics: CPU, Memory kullanımı
- Events: Deploy history

### Telegram
- Bot sinyalleri Telegram grubunuza gelecek
- Her 10 dakikada bir tarama

## ⚙️ Configuration

### Önemli Parametreler
- `MIN_SIGNAL_SCORE=4.0` → Daha yüksek = daha az sinyal
- `MIN_VOLUME_RATIO=1.3` → Hacim filtresi
- `SIGNAL_COOLDOWN_H=24` → Aynı coin için bekleme süresi
- `MAX_SYMBOLS=100` → Taranacak maksimum coin sayısı

### Fine-tuning
1. İlk hafta varsayılan ayarlarla çalıştırın
2. Log'ları analiz edin
3. Gerekirse parametreleri ayarlayın
4. Render Dashboard → Settings → Environment → Restart

## 🐛 Troubleshooting

### Bot başlamıyor
- Environment variables'ı kontrol edin
- Logs'ta hata mesajlarına bakın
- Health endpoint'i test edin

### Telegram mesajları gelmiyor
- Token ve Chat ID'yi doğrulayın
- Bot'u gruba admin olarak ekleyin
- `/start` komutu gönderin

### Çok fazla sinyal
- `MIN_SIGNAL_SCORE` değerini artırın (5.0-6.0)
- `MIN_VOLUME_RATIO` değerini artırın (1.5-2.0)

### Hiç sinyal yok
- `MIN_SIGNAL_SCORE` değerini düşürün (3.5-4.0)
- Log'larda "signals_found" sayısını kontrol edin

## 📈 Performance

### Free Tier Limits (Render)
- 750 saat/ay (yeterli)
- 512 MB RAM
- Shared CPU
- Auto-sleep after 15 min inactivity (web request ile uyanır)

### Keep-alive (Optional)
Ücretsiz planda sleep'i önlemek için:
- UptimeRobot ile 10 dakikada bir ping
- Veya cron job ile health endpoint'e istek

## 🔒 Security

### Secrets Management
- Telegram token'ı asla GitHub'a push etmeyin
- `.env` dosyası `.gitignore`'da
- Render'da Environment Variables kullanın

### API Rate Limits
- Binance: 1200 request/min
- Bot: 0.1s delay between requests
- Safe ✅

## 📝 Maintenance

### Updates
```bash
# Kodu güncelleyin
git add .
git commit -m "Update: XYZ"
git push

# Render otomatik deploy edecek (30-60 saniye)
```

### Database Backup
```bash
# Render Dashboard → Shell
cd /data
cat trading_bot.db > backup.db
```

### Logs Export
```bash
# Render Dashboard → Logs → Download
```

## 💡 Tips

1. **İlk 24 Saat**: Parametreleri değiştirmeyin, gözlemleyin
2. **Backtesting**: Tarihi verileri analiz edin
3. **Risk Yönetimi**: Her sinyali manuel kontrol edin
4. **Diversifikasyon**: Tek bir sinyale güvenmeyin
5. **Stop-Loss**: Mutlaka kullanın

## 🆘 Support

- GitHub Issues: Bug report için
- Telegram: Sinyal bildirimleri
- Render Support: Deploy sorunları için

## 📄 License

MIT License - Use at your own risk

## ⚠️ Disclaimer

Bu bot eğitim amaçlıdır. Finansal tavsiye değildir.
Gerçek parayla trade yapmadan önce:
- Stratejiyi anlayın
- Paper trading yapın
- Risk yönetimi uygulayın
"""
