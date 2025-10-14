# -*- coding: utf-8 -*-
"""
Professional Buy-Only Trading Bot - Fixed Version 2.0
Tüm kritik hatalar düzeltildi
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import logging
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from aiohttp import web
import signal
import sys
import traceback
from contextlib import asynccontextmanager

# ---------- Enhanced Config ----------
@dataclass
class BotConfig:
    BINANCE_BASE: str = "https://api.binance.com"
    TELEGRAM_TOKEN: Optional[str] = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    MIN_VOLUME_24H: float = float(os.getenv("MIN_VOLUME_24H", "100000"))
    MAX_SYMBOLS: int = int(os.getenv("MAX_SYMBOLS", "100"))
    MIN_SIGNAL_SCORE: float = float(os.getenv("MIN_SIGNAL_SCORE", "4.0"))
    MIN_ACTIVE_SIGNALS: int = 3
    MIN_VOLUME_RATIO: float = 1.3  # 2.0'dan düşürüldü
    SIGNAL_COOLDOWN_HOURS: int = int(os.getenv("SIGNAL_COOLDOWN_H", "24"))
    TIMEFRAMES: Tuple[str, ...] = ("15m", "30m", "1h", "4h", "1d")  # 45m → 30m
    CONCURRENCY_LIMIT: int = 3
    PORT: int = int(os.getenv("PORT", "10000"))
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 5.0
    REQUEST_TIMEOUT: int = 30
    RATE_LIMIT_DELAY: float = 0.1
    DATABASE_PATH: str = "trading_bot.db"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_BACKTESTING: bool = os.getenv("ENABLE_BACKTESTING", "false").lower() == "true"

CONFIG = BotConfig()

# ---------- Enhanced Logging System ----------
class BotLogger:
    def __init__(self, config: BotConfig):
        self.config = config
        self.setup_logging()

    def setup_logging(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        try:
            file_handler = logging.FileHandler('bot.log')
            file_handler.setFormatter(formatter)
        except Exception:
            file_handler = logging.StreamHandler(sys.stdout)
            file_handler.setFormatter(formatter)

        logging.root.setLevel(getattr(logging, self.config.LOG_LEVEL.upper()))
        logging.root.addHandler(console_handler)
        logging.root.addHandler(file_handler)

        logging.getLogger('aiohttp').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------- FIXED Technical Indicators ----------
class StrategyAlignedIndicators:
    @staticmethod
    def to_dataframe(klines: List) -> pd.DataFrame:
        if not klines:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'buy_base', 'buy_quote', 'ignore'
            ])
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=numeric_cols)
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            return df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"DataFrame conversion error: {e}")
            return pd.DataFrame()

    @staticmethod
    def resample_to_45m(df: pd.DataFrame) -> pd.DataFrame:
        """15m veriyi 45m'ye resample et"""
        if df.empty or len(df) < 3:
            return pd.DataFrame()
        
        try:
            df = df.set_index('timestamp')
            
            resampled = pd.DataFrame()
            resampled['open'] = df['open'].resample('45T').first()
            resampled['high'] = df['high'].resample('45T').max()
            resampled['low'] = df['low'].resample('45T').min()
            resampled['close'] = df['close'].resample('45T').last()
            resampled['volume'] = df['volume'].resample('45T').sum()
            
            resampled = resampled.dropna()
            resampled = resampled.reset_index()
            
            return resampled
            
        except Exception as e:
            logger.error(f"45m resample error: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_strategy_config(timeframe: str) -> Dict:
        """Strateji notlarına göre timeframe konfigürasyonu"""
        configs = {
            "15m": {
                "ema_periods": [9, 21],
                "rsi_period": 5,
                "macd_params": (6, 13, 4),
                "volume_period": 5,
                "min_rsi": 65,  # 70'ten düşürüldü
                "rsi_oversold": 30,
                "rsi_bullish": 40
            },
            "30m": {  # 45m yerine
                "ema_periods": [21, 65, 200],
                "rsi_period": 14,
                "macd_params": (12, 26, 9),
                "volume_period": 5,
                "min_rsi": 60,
                "rsi_oversold": 30,
                "rsi_bullish": 55
            },
            "1h": {
                "ema_periods": [21, 50],
                "rsi_period": 14,
                "macd_params": (12, 26, 9),
                "volume_period": 5,
                "min_rsi": 60,
                "rsi_oversold": 30,
                "rsi_bullish": 55
            },
            "4h": {
                "ema_periods": [13, 21, 34],
                "rsi_period": 14,
                "macd_params": (12, 26, 9),
                "volume_period": 20,
                "min_rsi": 55,
                "rsi_oversold": 30,
                "rsi_bullish": 50
            },
            "1d": {
                "ema_periods": [9, 21, 50],
                "rsi_period": 14,
                "macd_params": (12, 26, 9),
                "volume_period": 20,
                "min_rsi": 55,
                "rsi_oversold": 30,
                "rsi_bullish": 50
            }
        }
        return configs.get(timeframe, configs["1h"])

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's Smoothed RSI - daha doğru"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # İlk ortalama
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Wilder smoothing
        for i in range(period, len(gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
        if len(df) < 20:
            return df
            
        try:
            close = df['close']
            config = StrategyAlignedIndicators.get_strategy_config(timeframe)
            
            # EMA'lar
            for period in config["ema_periods"]:
                if len(df) >= period:
                    df[f'ema_{period}'] = StrategyAlignedIndicators.calculate_ema(close, period)
                else:
                    df[f'ema_{period}'] = np.nan
            
            df['ema_20'] = StrategyAlignedIndicators.calculate_ema(close, 20)
            
            # Hacim analizleri
            volume_period = config["volume_period"]
            df[f'volume_sma_{volume_period}'] = df['volume'].rolling(volume_period, min_periods=1).mean()
            df['volume_sma_20'] = df['volume'].rolling(20, min_periods=1).mean()
            
            # Wilder RSI (düzeltilmiş)
            rsi_period = config["rsi_period"]
            if len(df) >= rsi_period:
                df['rsi'] = StrategyAlignedIndicators.calculate_wilder_rsi(close, rsi_period)
            else:
                df['rsi'] = 50.0
            
            # MACD
            if len(df) >= max(config["macd_params"]):
                fast, slow, signal = config["macd_params"]
                macd, signal_line, hist = StrategyAlignedIndicators.calculate_macd(close, fast, slow, signal)
                df['macd'] = macd
                df['macd_signal'] = signal_line
                df['macd_histogram'] = hist
            
            # Support/Resistance
            df['support_level'] = df['low'].rolling(10, min_periods=1).min()
            df['resistance_level'] = df['high'].rolling(10, min_periods=1).max()
            
            # OBV ve VWAP
            df['obv'] = StrategyAlignedIndicators.calculate_obv(df)
            df['vwap'] = StrategyAlignedIndicators.calculate_vwap(df)
            
            return df.replace([np.inf, -np.inf], np.nan)
            
        except Exception as e:
            logger.error(f"Indicators calculation error: {e}")
            return df

# ---------- Database Manager ----------
class DatabaseManager:
    def __init__(self, db_path: str = CONFIG.DATABASE_PATH):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        score REAL NOT NULL,
                        price REAL NOT NULL,
                        rsi REAL,
                        volume_ratio REAL,
                        signals_json TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        sent_telegram BOOLEAN DEFAULT FALSE
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def log_signal(self, signal_data: Dict) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO signals (symbol, timeframe, score, price, rsi, volume_ratio, signals_json, sent_telegram)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_data.get('symbol'),
                    signal_data.get('timeframe'),
                    signal_data.get('score'),
                    signal_data.get('current_values', {}).get('price'),
                    signal_data.get('current_values', {}).get('rsi'),
                    signal_data.get('current_values', {}).get('volume_ratio'),
                    json.dumps(signal_data.get('signals', {})),
                    signal_data.get('sent_telegram', False)
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Database signal logging error: {e}")
            return False

db_manager = DatabaseManager()

# ---------- Binance Data Provider ----------
class BinanceDataProvider:
    def __init__(self, config: BotConfig):
        self.config = config
        self.base_url = config.BINANCE_BASE
        self.session_timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)

    @asynccontextmanager
    async def get_session(self):
        session = None
        try:
            connector = aiohttp.TCPConnector(limit=10)
            session = aiohttp.ClientSession(timeout=self.session_timeout, connector=connector)
            yield session
        finally:
            if session:
                await session.close()

    async def get_24h_tickers(self, session: aiohttp.ClientSession) -> List[Dict]:
        try:
            async with session.get(f"{self.base_url}/api/v3/ticker/24hr") as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
        except Exception as e:
            logger.error(f"Binance ticker error: {e}")
            return []

    async def get_klines(self, session: aiohttp.ClientSession, symbol: str, interval: str = '1h', limit: int = 100) -> List:
        try:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            async with session.get(f"{self.base_url}/api/v3/klines", params=params) as response:
                if response.status == 200:
                    return await response.json()
                return []
        except Exception:
            return []

    async def get_multi_timeframe_data(self, session: aiohttp.ClientSession, symbol: str) -> Dict[str, pd.DataFrame]:
        data = {}
        
        # 15m'yi al, hem kendisi hem 45m için kullan
        klines_15m = await self.get_klines(session, symbol, "15m", 300)  # 45m için yeterli
        if klines_15m:
            df_15m = StrategyAlignedIndicators.to_dataframe(klines_15m)
            if not df_15m.empty and len(df_15m) >= 20:
                data["15m"] = df_15m
                
                # 45m'yi resample et
                df_45m = StrategyAlignedIndicators.resample_to_45m(df_15m)
                if not df_45m.empty and len(df_45m) >= 20:
                    data["45m"] = df_45m
        
        await asyncio.sleep(0.1)
        
        # Diğer timeframe'ler
        for tf in ["30m", "1h", "4h", "1d"]:
            try:
                limit = 100
                klines = await self.get_klines(session, symbol, tf, limit)
                if klines:
                    df = StrategyAlignedIndicators.to_dataframe(klines)
                    if not df.empty and len(df) >= 20:
                        data[tf] = df
                await asyncio.sleep(0.1)
            except Exception:
                continue
        
        return data

# ---------- FIXED Buy Signal Analyzer ----------
class StrategyAlignedBuyAnalyzer:
    def __init__(self, config: BotConfig):
        self.config = config
        self.indicators = StrategyAlignedIndicators()

    def analyze_symbol(self, df: pd.DataFrame, timeframe: str = "1h") -> Dict[str, Any]:
        if len(df) < 30:
            return {'score': 0.0, 'signals': {}, 'current_values': {}}

        try:
            df = self.indicators.calculate_all_indicators(df, timeframe)
            
            if df.empty:
                return {'score': 0.0, 'signals': {}, 'current_values': {}}
            
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else current
            
            if pd.isna(current['close']) or current['close'] <= 0:
                return {'score': 0.0, 'signals': {}, 'current_values': {}}
            
            # Kategori bazlı puanlama
            kategori_puanlari = {
                'rsi': 0.0,
                'macd': 0.0,
                'ema': 0.0,
                'hacim': 0.0,
                'destek': 0.0,
                'uyumsuzluk': 0.0
            }
            
            signals = {}
            kategori_signal_sayisi = {k: 0 for k in kategori_puanlari.keys()}
            
            # RSI Analizi
            rsi_puan, rsi_sinyaller = self._analyze_rsi_strategy(df, current, previous, timeframe)
            kategori_puanlari['rsi'] = min(rsi_puan, 2.0)
            kategori_signal_sayisi['rsi'] = sum(1 for v in rsi_sinyaller.values() if v is True)
            signals.update(rsi_sinyaller)
            
            # MACD Analizi
            macd_puan, macd_sinyaller = self._analyze_macd_strategy(df, current, previous, timeframe)
            kategori_puanlari['macd'] = min(macd_puan, 1.5)
            kategori_signal_sayisi['macd'] = sum(1 for v in macd_sinyaller.values() if v is True)
            signals.update(macd_sinyaller)
            
            # EMA Trend Analizi
            ema_puan, ema_sinyaller = self._analyze_ema_strategy(df, current, previous, timeframe)
            kategori_puanlari['ema'] = min(ema_puan, 2.0)
            kategori_signal_sayisi['ema'] = sum(1 for v in ema_sinyaller.values() if v is True)
            signals.update(ema_sinyaller)
            
            # Hacim Analizi
            hacim_puan, hacim_sinyaller = self._analyze_volume_strategy(df, current, previous, timeframe)
            kategori_puanlari['hacim'] = min(hacim_puan, 1.5)
            kategori_signal_sayisi['hacim'] = sum(1 for v in hacim_sinyaller.values() if v is True)
            signals.update(hacim_sinyaller)
            
            # Destek/Direnç
            destek_puan, destek_sinyaller = self._analyze_support_resistance(df, current, previous, timeframe)
            kategori_puanlari['destek'] = min(destek_puan, 1.0)
            kategori_signal_sayisi['destek'] = sum(1 for v in destek_sinyaller.values() if v is True)
            signals.update(destek_sinyaller)
            
            # Uyumsuzluk
            uyumsuzluk_puan, uyumsuzluk_sinyaller = self._detect_positive_divergence(df, current, previous, timeframe)
            kategori_puanlari['uyumsuzluk'] = min(uyumsuzluk_puan * 0.15, 0.5)
            kategori_signal_sayisi['uyumsuzluk'] = sum(1 for v in uyumsuzluk_sinyaller.values() if v is True)
            signals.update(uyumsuzluk_sinyaller)
            
            # Temel puan
            temel_puan = sum(kategori_puanlari.values())
            
            # Hacim oranı - kademeli ceza
            hacim_periyodu = 5 if timeframe in ['15m', '1h'] else 20
            hacim_ortalaması = current.get(f'volume_sma_{hacim_periyodu}', 1)
            hacim_oranı = current.get('volume', 0) / hacim_ortalaması if hacim_ortalaması > 0 else 0
            
            cezalar = 0.0
            bonuslar = 0.0
            
            # Kademeli hacim cezası (düzeltilmiş)
            if hacim_oranı < 1.0:
                cezalar += 3.0
                signals['cok_dusuk_hacim'] = True
            elif hacim_oranı < self.config.MIN_VOLUME_RATIO:
                eksiklik = (self.config.MIN_VOLUME_RATIO - hacim_oranı) / self.config.MIN_VOLUME_RATIO
                ceza = eksiklik * 1.5
                cezalar += ceza
                signals['dusuk_hacim_cezasi'] = round(ceza, 2)
            
            # Aşırı satış fırsatı bonusu - ERKEN HESAPLA
            current_rsi = current.get('rsi', 50)
            if current_rsi < 25 and hacim_oranı > 2.0:
                bonuslar += 0.3
                signals['asiri_satis_firsati'] = True
            
            # Kategori başına max 1 sinyal say
            aktif_sinyal_sayisi = sum(min(count, 1) for count in kategori_signal_sayisi.values())
            
            if aktif_sinyal_sayisi < self.config.MIN_ACTIVE_SIGNALS:
                ceza = (self.config.MIN_ACTIVE_SIGNALS - aktif_sinyal_sayisi) * 0.4
                cezalar += ceza
                signals['min_sinyal_cezasi'] = ceza
            
            # Zaman dilimi güvenilirlik
            zaman_dilimi_carpanlari = {
                "15m": 0.80,
                "30m": 0.95,  # 45m yerine
                "45m": 1.0,   # Resample edilmiş
                "1h": 0.95,
                "4h": 1.05,
                "1d": 1.08
            }
            
            guvenilirlik_carpani = zaman_dilimi_carpanlari.get(timeframe, 1.0)
            
            # Son puan hesaplama
            ayarlanmis_puan = (temel_puan * guvenilirlik_carpani) + bonuslar - cezalar
            final_puan = max(0.0, min(ayarlanmis_puan, 7.5))
            
            # Aşırı sinyalleşme
            toplam_sinyal = sum(1 for sig in signals.values() if sig is True)
            if toplam_sinyal > 10:
                final_puan *= 0.75
                signals['asiri_sinyallesme_cezasi'] = True
            
            result = {
                'score': round(final_puan, 2),
                'signals': signals,
                'current_values': {
                    'price': float(current['close']),
                    'rsi': float(current.get('rsi', 50)),
                    'volume_ratio': round(float(hacim_oranı), 2),
                    'active_signals_count': aktif_sinyal_sayisi,
                    'timeframe': timeframe,
                    'kategori_dagilimi': {k: round(v, 2) for k, v in kategori_puanlari.items()},
                    'guvenilirlik_carpani': guvenilirlik_carpani,
                    'toplam_ceza': round(cezalar, 2),
                    'toplam_bonus': round(bonuslar, 2)
                }
            }
            
            return result

        except Exception as e:
            logger.error(f"Sembol analiz hatası: {e}")
            return {'score': 0.0, 'signals': {}, 'current_values': {}}

    def _analyze_rsi_strategy(self, df: pd.DataFrame, current: pd.Series, previous: pd.Series, timeframe: str) -> Tuple[float, Dict]:
        score = 0.0
        signals = {}
        
        try:
            config = self.indicators.get_strategy_config(timeframe)
            current_rsi = current.get('rsi', 50)
            prev_rsi = previous.get('rsi', 50)
            
            if pd.isna(current_rsi) or pd.isna(prev_rsi):
                return 0.0, {}
            
            # RSI 30 kesişimi yukarı
            if prev_rsi <= 30 and current_rsi > 30:
                score += 1.2
                signals['rsi_30_yukari_kesisim'] = True
            
            # RSI boğa bölgesinde
            elif current_rsi >= config['rsi_bullish']:
                score += 0.6
                signals['rsi_boga_bolgesi'] = True
            
            # RSI güçlü momentum
            if current_rsi >= config['min_rsi']:
                score += 0.5
                signals['rsi_guclu_momentum'] = True
            
            # RSI yukarı kıvrılma
            if 30 <= current_rsi <= 45 and current_rsi > prev_rsi + 2:
                score += 0.3
                signals['rsi_yukari_kivrilma'] = True
            
            signals['mevcut_rsi'] = round(current_rsi, 2)
            return min(score, 2.0), signals
            
        except Exception:
            return 0.0, {}

    def _analyze_macd_strategy(self, df: pd.DataFrame, current: pd.Series, previous: pd.Series, timeframe: str) -> Tuple[float, Dict]:
        score = 0.0
        signals = {}
        
        try:
            current_macd = current.get('macd', 0)
            current_signal = current.get('macd_signal', 0)
            current_hist = current.get('macd_histogram', 0)
            prev_macd = previous.get('macd', 0)
            prev_signal = previous.get('macd_signal', 0)
            
            if any(pd.isna(x) for x in [current_macd, current_signal, prev_macd, prev_signal]):
                return 0.0, {}
            
            # MACD 0 üstü
            if current_macd > 0:
                score += 0.5
                signals['macd_sifir_ustu'] = True
            
            # MACD pozitif kesişim
            if (current_macd > current_signal and prev_macd <= prev_signal):
                if current_macd > 0:
                    score += 1.0
                    signals['macd_boga_kesisim_sifir_ustu'] = True
                else:
                    score += 0.7
                    signals['macd_boga_kesisim_sifir_alti'] = True
            
            return min(score, 1.5), signals
            
        except Exception:
            return 0.0, {}

    def _analyze_ema_strategy(self, df: pd.DataFrame, current: pd.Series, previous: pd.Series, timeframe: str) -> Tuple[float, Dict]:
        score = 0.0
        signals = {}
        
        try:
            config = self.indicators.get_strategy_config(timeframe)
            price = current['close']
            emas = {}
            
            for period in config["ema_periods"]:
                ema_val = current.get(f'ema_{period}', np.nan)
                if not pd.isna(ema_val) and ema_val > 0:
                    emas[period] = ema_val
            
            if len(emas) < 2:
                return 0.0, {}
            
            # Timeframe'e göre EMA analizleri
            if timeframe in ["30m", "45m"] and all(period in emas for period in [21, 65, 200]):
                if price > emas[200]:
                    score += 0.8
                    signals['fiyat_ema200_ustu'] = True
                
                if emas[21] > emas[65]:
                    score += 0.7
                    signals['ema21_ema65_ustu'] = True
            
            elif timeframe == "1h" and all(period in emas for period in [21, 50]):
                if price > emas[50] and price > emas[21]:
                    score += 0.8
                    signals['fiyat_anahtar_ema_ustu'] = True
                
                if emas[21] > emas[50]:
                    score += 0.6
                    signals['ema_boga_siralama'] = True
            
            # 20 EMA kırılımı
            ema_20 = current.get('ema_20', np.nan)
            if not pd.isna(ema_20):
                prev_ema20 = previous.get('ema_20', ema_20)
                prev_price = previous['close']
                
                if prev_price <= prev_ema20 and price > ema_20:
                    score += 0.8
                    signals['ema20_kirilim'] = True
            
            return min(score, 2.0), signals
            
        except Exception:
            return 0.0, {}

    def _analyze_volume_strategy(self, df: pd.DataFrame, current: pd.Series, previous: pd.Series, timeframe: str) -> Tuple[float, Dict]:
        score = 0.0
        signals = {}
        
        try:
            current_volume = current['volume']
            
            if pd.isna(current_volume) or current_volume <= 0:
                return 0.0, {}
            
            # OBV ve VWAP teyidi
            current_obv = current.get('obv', 0)
            prev_obv = previous.get('obv', 0)
            vwap = current.get('vwap', current['close'])
            price = current['close']
            
            if timeframe in ['15m', '1h']:
                vol_avg = current.get('volume_sma_5', 1)
                volume_ratio = current_volume / vol_avg
                
                if volume_ratio >= 1.5:
                    score += 0.5
                    signals['hacim_150pct_ustu'] = True
                
                if volume_ratio >= 2.0:
                    score += 0.8
                    signals['hacim_spike_2x'] = True
            else:
                vol_avg = current.get('volume_sma_20', 1)
                volume_ratio = current_volume / vol_avg
                
                if volume_ratio >= 1.5:
                    score += 0.6
                    signals['hacim_onay_150pct'] = True
                
                if volume_ratio >= 2.0:
                    score += 0.9
                    signals['hacim_onay_200pct'] = True
            
            # OBV yükseliş teyidi
            if not pd.isna(current_obv) and not pd.isna(prev_obv):
                if current_obv > prev_obv:
                    score += 0.2
                    signals['obv_yukselis'] = True
            
            # VWAP üstü
            if not pd.isna(vwap) and price > vwap:
                score += 0.2
                signals['vwap_ustu'] = True
            
            signals['hacim_orani'] = round(volume_ratio, 2)
            return min(score, 1.5), signals
            
        except Exception:
            return 0.0, {}

    def _analyze_support_resistance(self, df: pd.DataFrame, current: pd.Series, previous: pd.Series, timeframe: str) -> Tuple[float, Dict]:
        score = 0.0
        signals = {}
        
        try:
            price = current['close']
            support = current.get('support_level', 0)
            resistance = current.get('resistance_level', 0)
            
            if pd.isna(support) or support <= 0:
                return 0.0, {}
            
            # EMA trend kontrolü
            ema_21 = current.get('ema_21', np.nan)
            ema_50 = current.get('ema_50', np.nan)
            trend_down = False
            
            if not pd.isna(ema_21) and not pd.isna(ema_50):
                trend_down = ema_21 < ema_50
            
            # Destek seviyesine yakınlık
            distance_from_support = (price - support) / support
            if 0 <= distance_from_support <= 0.02:
                # Trend aşağıysa yarı puan
                if trend_down:
                    score += 0.4
                    signals['destek_yakin_zayif_trend'] = True
                else:
                    score += 0.8
                    signals['destek_yakin'] = True
                
                signals['destek_mesafe_yuzde'] = round(distance_from_support * 100, 2)
            
            # Direnç seviyesine uzaklık
            if not pd.isna(resistance) and resistance > 0:
                distance_to_resistance = (resistance - price) / price
                if distance_to_resistance >= 0.05:
                    score += 0.2
                    signals['yukari_potansiyel'] = True
            
            return min(score, 1.0), signals
            
        except Exception:
            return 0.0, {}

    def _detect_positive_divergence(self, df: pd.DataFrame, current: pd.Series, previous: pd.Series, timeframe: str) -> Tuple[float, Dict]:
        score = 0.0
        signals = {}
        
        try:
            if len(df) < 20:
                return 0.0, {}
            
            recent_data = df.tail(20)
            
            # RSI pozitif uyumsuzluk
            price_lows = []
            rsi_lows = []
            macd_lows = []
            
            for i in range(2, len(recent_data) - 2):
                if (recent_data.iloc[i]['close'] < recent_data.iloc[i-1]['close'] and 
                    recent_data.iloc[i]['close'] < recent_data.iloc[i+1]['close']):
                    price_lows.append((i, recent_data.iloc[i]['close']))
                
                if (recent_data.iloc[i]['rsi'] < recent_data.iloc[i-1]['rsi'] and 
                    recent_data.iloc[i]['rsi'] < recent_data.iloc[i+1]['rsi']):
                    rsi_lows.append((i, recent_data.iloc[i]['rsi']))
                
                # MACD histogram düşük noktaları
                macd_hist = recent_data.iloc[i].get('macd_histogram', 0)
                if not pd.isna(macd_hist):
                    if (macd_hist < recent_data.iloc[i-1].get('macd_histogram', 0) and 
                        macd_hist < recent_data.iloc[i+1].get('macd_histogram', 0)):
                        macd_lows.append((i, macd_hist))
            
            # RSI uyumsuzluk
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                last_price_low = price_lows[-1][1]
                prev_price_low = price_lows[-2][1]
                last_rsi_low = rsi_lows[-1][1]
                prev_rsi_low = rsi_lows[-2][1]
                
                if last_price_low < prev_price_low and last_rsi_low > prev_rsi_low:
                    score += 2.0
                    signals['rsi_pozitif_uyumsuzluk'] = True
                    
                    # MACD teyidi
                    if len(macd_lows) >= 2:
                        last_macd_low = macd_lows[-1][1]
                        prev_macd_low = macd_lows[-2][1]
                        if last_macd_low > prev_macd_low:
                            score += 0.5
                            signals['macd_uyumsuzluk_teyid'] = True
            
            return score, signals
            
        except Exception:
            return 0.0, {}

# ---------- FIXED Telegram Notifier ----------
class TelegramNotifier:
    def __init__(self, config: BotConfig):
        self.config = config
        self.token = config.TELEGRAM_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        
        if self.enabled:
            logger.info("Telegram bildirimleri aktif")

    async def send_buy_signal(self, session: aiohttp.ClientSession, symbol: str, 
                             score: float, analysis: Dict, timeframe: str = "1h") -> bool:
        if not self.enabled:
            return False
            
        try:
            current = analysis.get('current_values', {})
            price = current.get('price', 0)
            rsi = current.get('rsi', 50)
            volume_ratio = current.get('volume_ratio', 1)
            active_signals = current.get('active_signals_count', 0)
            
            kategori_dag = current.get('kategori_dagilimi', {})
            
            message = (
                f"🟢 <b>SATIN AL SİNYALİ - {symbol}</b>\n\n"
                f"💰 Fiyat: ${price:.6f}\n"
                f"🎯 Puan: <b>{score:.2f}/7.5</b>\n"
                f"📊 Zaman Dilimi: {timeframe}\n"
                f"📈 RSI: {rsi:.1f}\n"
                f"📊 Hacim: {volume_ratio:.2f}x\n"
                f"🔢 Aktif Sinyaller: {active_signals}\n\n"
                f"🎪 <b>Kategori Dağılımı:</b>\n"
            )
            
            for kategori, puan in kategori_dag.items():
                if puan > 0:
                    message += f"• {kategori.title()}: {puan:.1f}\n"
            
            signals_list = analysis.get('signals', {})
            active_signal_names = [k for k, v in signals_list.items() if v is True and not k.endswith('_cezasi')]
            
            if active_signal_names:
                message += f"\n✅ <b>Ana Sinyaller:</b>\n"
                for signal in active_signal_names[:4]:
                    signal_tr = signal.replace('_', ' ').title()
                    message += f"• {signal_tr}\n"
            
            message += f"\n⏰ Zaman: {datetime.now().strftime('%H:%M:%S')} UTC"
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id, 
                "text": message, 
                "parse_mode": "HTML"
            }
            
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    # Telegram API yanıt kontrolü
                    if result.get('ok') is True:
                        return True
                    else:
                        logger.error(f"Telegram hatası: {result.get('description', 'Bilinmeyen')}")
                        return False
                else:
                    logger.error(f"Telegram HTTP hatası: {resp.status}")
                    return False
                
        except Exception as e:
            logger.error(f"Telegram bildirim hatası: {e}")
            return False

# ---------- Ana Trading Bot ----------
class StrategyAlignedTradingBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.data_provider = BinanceDataProvider(config)
        self.analyzer = StrategyAlignedBuyAnalyzer(config)
        self.telegram = TelegramNotifier(config)
        self.signal_history = []
        self.cooldown_tracker = {}
        self.is_running = True
        self.scan_count = 0
        self.error_count = 0

    async def run_scan(self) -> Dict[str, Any]:
        scan_start_time = datetime.now()
        scan_stats = {
            'symbols_scanned': 0,
            'signals_found': 0,
            'high_quality_signals': 0,
            'errors': 0,
            'duration': 0.0
        }
        
        try:
            logger.info(f"🔍 Tarama başlatılıyor #{self.scan_count + 1}")
            
            async with self.data_provider.get_session() as session:
                tickers = await self.data_provider.get_24h_tickers(session)
                if not tickers:
                    return scan_stats
                
                symbols = self._filter_symbols(tickers)
                scan_stats['symbols_scanned'] = len(symbols)
                
                if not symbols:
                    return scan_stats
                
                buy_signals = []
                high_quality_signals = []
                
                # Timeframe öncelik sırası - 45m eklendi
                timeframe_priority = ["45m", "30m", "1h", "4h", "1d", "15m"]
                
                for symbol in symbols[:15]:
                    if not self.is_running:
                        break
                    
                    try:
                        multi_tf_data = await self.data_provider.get_multi_timeframe_data(session, symbol)
                        
                        if not multi_tf_data:
                            continue
                        
                        best_signal = None
                        best_score = 0
                        
                        for tf in timeframe_priority:
                            if tf in multi_tf_data:
                                df = multi_tf_data[tf]
                                result = self.analyzer.analyze_symbol(df, tf)
                                score = result.get('score', 0)
                                
                                if score > best_score:
                                    best_score = score
                                    best_signal = result
                                    best_signal['timeframe'] = tf
                                    best_signal['symbol'] = symbol
                        
                        if best_signal and best_score >= self.config.MIN_SIGNAL_SCORE:
                            buy_signals.append(best_signal)
                            
                            current_vals = best_signal.get('current_values', {})
                            kategori_dag = current_vals.get('kategori_dagilimi', {})
                            
                            logger.info(f"✅ {symbol} ({best_signal['timeframe']}): {best_score:.2f} - "
                                      f"Sinyaller: {current_vals.get('active_signals_count', 0)} - "
                                      f"RSI:{kategori_dag.get('rsi', 0):.1f} EMA:{kategori_dag.get('ema', 0):.1f} "
                                      f"Hacim:{kategori_dag.get('hacim', 0):.1f}")
                            
                            if (best_score >= 6.0 and 
                                current_vals.get('active_signals_count', 0) >= 4 and
                                current_vals.get('volume_ratio', 0) >= 2.0):
                                high_quality_signals.append(best_signal)
                        
                        await asyncio.sleep(0.15)
                        
                    except Exception as e:
                        scan_stats['errors'] += 1
                        logger.error(f"Sembol {symbol} analiz hatası: {e}")
                        continue
                
                scan_stats['signals_found'] = len(buy_signals)
                scan_stats['high_quality_signals'] = len(high_quality_signals)
                
                signals_to_process = high_quality_signals if high_quality_signals else buy_signals
                
                if signals_to_process:
                    signals_to_process.sort(key=lambda x: (
                        -x['score'],
                        0 if x.get('timeframe') == '45m' else 1
                    ))
                    
                    await self._process_buy_signals(session, signals_to_process)
                else:
                    logger.info("📊 Strateji kriterlerini karşılayan sinyal bulunamadı")
                
        except Exception as e:
            logger.error(f"Strateji tarama hatası: {e}")
            scan_stats['errors'] += 1
        
        finally:
            scan_duration = (datetime.now() - scan_start_time).total_seconds()
            scan_stats['duration'] = scan_duration
            self.scan_count += 1
            
            logger.info(f"Tarama #{self.scan_count} tamamlandı {scan_duration:.1f}s - "
                       f"Sinyaller: {scan_stats['signals_found']} (YK: {scan_stats['high_quality_signals']})")
            
        return scan_stats

    def _filter_symbols(self, tickers: List[Dict]) -> List[str]:
        filtered = []
        
        stable_coins = {
            'USDCUSDT', 'BUSDUSDT', 'DAIUSDT', 'FDUSDUSDT', 'USDEUSDT', 
            'TUSDUSDT', 'USTCUSDT', 'PAXGUSDT', 'USTUSDT', 'FRAXUSDT',
            'PYUSDUSDT', 'AEURUSDT'
        }
        
        for ticker in tickers:
            try:
                symbol = ticker.get('symbol', '')
                volume = float(ticker.get('quoteVolume', 0))
                
                if not symbol.endswith('USDT'):
                    continue
                
                if symbol in stable_coins:
                    continue
                
                if any(fiat in symbol for fiat in ['EUR', 'GBP', 'AUD', 'BRL', 'TRY']):
                    continue
                
                if volume < self.config.MIN_VOLUME_24H:
                    continue
                
                if not self._check_cooldown(symbol):
                    continue
                
                filtered.append(symbol)
                
            except (ValueError, TypeError):
                continue
        
        volume_map = {}
        for ticker in tickers:
            try:
                volume_map[ticker.get('symbol')] = float(ticker.get('quoteVolume', 0))
            except (ValueError, TypeError):
                continue
        
        filtered.sort(key=lambda s: volume_map.get(s, 0), reverse=True)
        return filtered[:self.config.MAX_SYMBOLS]

    def _check_cooldown(self, symbol: str) -> bool:
        if symbol not in self.cooldown_tracker:
            return True
            
        last_time = self.cooldown_tracker[symbol]
        cooldown_seconds = self.config.SIGNAL_COOLDOWN_HOURS * 3600
        return (datetime.now().timestamp() - last_time) > cooldown_seconds

    async def _process_buy_signals(self, session: aiohttp.ClientSession, signals: List[Dict]):
        for signal in signals[:3]:
            try:
                symbol = signal['symbol']
                score = signal['score']
                timeframe = signal.get('timeframe', '1h')
                
                logger.info(f"📤 Sinyal işleniyor: {symbol} ({timeframe}) - Puan: {score:.2f}")
                
                telegram_success = await self.telegram.send_buy_signal(session, symbol, score, signal, timeframe)
                
                if telegram_success:
                    self.cooldown_tracker[symbol] = datetime.now().timestamp()
                    signal_data = signal.copy()
                    signal_data.update({
                        'symbol': symbol,
                        'sent_telegram': telegram_success
                    })
                    db_manager.log_signal(signal_data)
                    logger.info(f"✅ Sinyal gönderildi: {symbol} ({timeframe}) - Puan: {score:.2f}")
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Sinyal işleme hatası: {e}")
                continue

    def stop(self):
        self.is_running = False
        logger.info("🛑 Trading bot durduruluyor...")

# ---------- Health Server ----------
class HealthServer:
    def __init__(self, port: int = CONFIG.PORT):
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        self.runner = None
        self.site = None

    def setup_routes(self):
        self.app.router.add_get('/', self.health_check)
        self.app.router.add_get('/health', self.health_check)

    async def health_check(self, request):
        return web.json_response({
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "strategy": "Fixed v2.0 - 45m Resample + Wilder RSI + Kademeli Ceza"
        })

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await self.site.start()
        logger.info(f"Health server başlatıldı port {self.port}")

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

# ---------- Main Application ----------
class TradingBotApp:
    def __init__(self):
        self.config = BotConfig()
        self.bot = StrategyAlignedTradingBot(self.config)
        self.health_server = HealthServer(self.config.PORT)
        self.is_running = True

    async def start(self):
        try:
            await self.health_server.start()
            logger.info("✅ Health server başlatıldı")
            
            self._print_strategy_info()
            
            logger.info("🚀 Fixed v2.0 Trading Bot başlatılıyor...")
            await self.run_bot()
            
        except Exception as e:
            logger.error(f"Başlatma hatası: {e}")
            await self.shutdown()

    def _print_strategy_info(self):
        logger.info("📊 DÜZELTMELER:")
        logger.info("✓ 45m: 15m veriyi resample ile oluşturuldu")
        logger.info("✓ RSI: Wilder's Smoothed RSI kullanılıyor")
        logger.info("✓ Hacim: Kademeli ceza sistemi (1.3x minimum)")
        logger.info("✓ Bonus: Erken hesaplama (final_puan'a dahil)")
        logger.info("✓ Telegram: API yanıt kontrolü eklendi")
        logger.info("✓ Sinyal Sayımı: Kategori başına max 1 sinyal")
        logger.info("✓ OBV/VWAP: Hacim teyidi için eklendi")
        logger.info("✓ Destek: Trend kontrolü ile yarı puan")
        logger.info("✓ Uyumsuzluk: MACD histogram teyidi")
        logger.info(f"Min Hacim Oranı: {self.config.MIN_VOLUME_RATIO}x (kademeli)")
        logger.info(f"Min Sinyal Puanı: {self.config.MIN_SIGNAL_SCORE}")

    async def run_bot(self):
        logger.info("🤖 Bot başlatıldı - SADECE SATIN AL SİNYALLERİ")
        
        while self.is_running:
            try:
                scan_stats = await self.bot.run_scan()
                
                if scan_stats['high_quality_signals'] > 0:
                    logger.info(f"🎯 {scan_stats['high_quality_signals']} YÜKSEK KALİTE sinyal bulundu!")
                elif scan_stats['signals_found'] > 0:
                    logger.info(f"📈 Toplam {scan_stats['signals_found']} sinyal bulundu")
                else:
                    logger.info("📊 Bu taramada sinyal bulunamadı")
                
                logger.info("⏰ Sonraki tarama için 10 dakika bekleniyor...")
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Bot döngü hatası: {e}")
                await asyncio.sleep(60)

    async def shutdown(self):
        self.is_running = False
        self.bot.stop()
        await self.health_server.stop()
        logger.info("✅ Bot kapatma tamamlandı")

# ---------- Main Function ----------
async def main():
    app = None
    try:
        logger.info("🚀 Fixed v2.0 Trading Bot Başlatılıyor")
        BotLogger(CONFIG)
        
        app = TradingBotApp()
        await app.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Klavye kesmesi")
    except Exception as e:
        logger.critical(f"💥 Kritik hata: {e}")
    finally:
        if app:
            await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"💥 Kritik hata: {e}")
