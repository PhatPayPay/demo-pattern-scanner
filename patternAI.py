import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
from datetime import datetime, timedelta

# Success rates from the uploaded image
success_rates = {
    'Inverse Head and Shoulders': 84,
    'Head and Shoulders': 82,
    'Double Bottom': 82,
    'Channel Up': 73,
    'Channel Down': 72,
    'Descending Triangle': 70,
    'Double Top': 69,
    'Ascending Triangle': 68,
    'Flag': 66,
    'Falling Wedge': 66,
    'Rising Wedge': 65,
    'Triangle': 62,
    'Rectangle': 58,
    'Pennant': 58
}

# Helper to get success rate for pattern (match base name)
def get_success_rate(pattern_str):
    for key in success_rates:
        if key.lower() in pattern_str.lower():
            return success_rates[key]
    return 'N/A'

# Styling functions for DataFrame (fixed to handle Series)
def color_market_cap(series):
    styles = np.full(len(series), '')
    is_na = pd.isna(series)
    lt_100m = series < 100_000_000
    styles[lt_100m] = 'font-weight: bold;'
    styles[is_na] = ''
    return styles

# Binance Alpha coins (hardcoded from recent listings, prioritize these)
binance_alpha_coins = [
    'XPIN', 'MIRROR', 'OPEN', 'PTB', 'ZKC', 'HOLO', 'UB',  # From September 2025 listings
    # Add more if needed, e.g., 'BTC', 'ETH' for testing
]

# Fetch top coins from CoinGecko
@st.cache_data(ttl=300)
def get_top_coins(n=50, vol_change_threshold=10):
    cg_url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=volume_desc&per_page={}&page=1&sparkline=false&price_change_percentage=24h".format(n)
    try:
        response = requests.get(cg_url, timeout=10)
        data = response.json()
        
        results = []
        for coin in data:
            price_change = coin.get('price_change_percentage_24h', 0)
            current_price = coin.get('current_price', 0)
            market_cap = coin.get('market_cap', np.nan)
            volume_24h = coin.get('total_volume', 0)
            
            # Simulate volume change (use price change as proxy or fetch historical if needed)
            vol_change = price_change  # Placeholder, as CoinGecko doesn't have direct vol change
            
            results.append({
                'symbol': coin['symbol'].upper() + 'USDT',  # Format for consistency
                'current_price': current_price,
                'price_change_24h': price_change,
                'volume_change_24h': vol_change,
                'market_cap': market_cap,
                'id': coin['id']  # For OHLC fetch
            })
        
        filtered = [r for r in results if abs(r['volume_change_24h']) > vol_change_threshold]
        if not filtered:
            st.warning(f"Không coin nào thỏa filter volume change > {vol_change_threshold}%. Hiển thị top mà không filter.")
            return pd.DataFrame(results)
        return pd.DataFrame(filtered)
    except Exception as e:
        st.error(f"Error fetching top coins: {e}")
        return pd.DataFrame()

# Fetch OHLCV from CoinGecko
def fetch_ohlcv(coin_id, timeframe='1d', days=90):
    # Map timeframe to days
    if timeframe == '5m':
        days = 1  # CoinGecko min is 1 day for 5m
    elif timeframe == '15m':
        days = 1
    elif timeframe == '1h':
        days = 7
    elif timeframe == '4h':
        days = 30
    elif timeframe == '1d':
        days = 90
    
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['volume'] = np.nan  # CoinGecko OHLC doesn't have volume, set to NaN
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV for {coin_id}: {e}")
        return pd.DataFrame()

def find_pivots(df, order=5):
    if len(df) < order * 2:
        return [], []
    highs_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
    lows_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
    return highs_idx, lows_idx

# Detect patterns with success rate
def detect_patterns(df, highs_idx, lows_idx):
    patterns = []
    current_price = df['close'].iloc[-1]
    
    highs = df['high'].iloc[highs_idx]
    lows = df['low'].iloc[lows_idx]
    
    # 1. Head and Shoulders / Inverse
    if len(highs_idx) >= 3:
        recent_highs = highs.tail(3).values
        if recent_highs[1] > recent_highs[0] * 1.02 and recent_highs[1] > recent_highs[2] * 1.02:
            p = 'Head and Shoulders - Tiềm năng giảm'
            sr = get_success_rate(p)
            patterns.append(f"{p} (Success Rate: {sr}%)")
    if len(lows_idx) >= 3:
        recent_lows = lows.tail(3).values
        if recent_lows[1] < recent_lows[0] * 0.98 and recent_lows[1] < recent_lows[2] * 0.98:
            p = 'Inverse Head and Shoulders - Tiềm năng tăng'
            sr = get_success_rate(p)
            patterns.append(f"{p} (Success Rate: {sr}%)")
    
    # 2. Channel
    channel = detect_channel(df, highs, lows, highs_idx, lows_idx)
    if channel:
        p = channel['direction']
        sr = get_success_rate(p)
        patterns.append(f"{p} (Slope: {channel['slope']:.4f}, Success Rate: {sr}%)")
    
    # 3. Triangles
    if len(highs_idx) >= 3 and len(lows_idx) >= 3:
        recent_highs_y = highs.tail(3).values
        recent_lows_y = lows.tail(3).values
        if np.std(recent_highs_y) / np.mean(recent_highs_y) < 0.01:
            slope_lows = linregress(range(3), recent_lows_y).slope
            if slope_lows > 0:
                p = 'Ascending Triangle - Tiềm năng tăng'
                sr = get_success_rate(p)
                patterns.append(f"{p} (Success Rate: {sr}%)")
        if np.std(recent_lows_y) / np.mean(recent_lows_y) < 0.01:
            slope_highs = linregress(range(3), recent_highs_y).slope
            if slope_highs < 0:
                p = 'Descending Triangle - Tiềm năng giảm'
                sr = get_success_rate(p)
                patterns.append(f"{p} (Success Rate: {sr}%)")
    
    # 4. Wedges
    if len(highs_idx) >= 3 and len(lows_idx) >= 3:
        slope_h = linregress(range(3), highs.tail(3).values).slope
        slope_l = linregress(range(3), lows.tail(3).values).slope
        if abs(slope_h - slope_l) < 0.01 and slope_h < 0 and slope_l < 0:
            p = 'Falling Wedge - Tiềm năng tăng (reversal)'
            sr = get_success_rate(p)
            patterns.append(f"{p} (Success Rate: {sr}%)")
        if abs(slope_h - slope_l) < 0.01 and slope_h > 0 and slope_l > 0:
            p = 'Rising Wedge - Tiềm năng giảm (reversal)'
            sr = get_success_rate(p)
            patterns.append(f"{p} (Success Rate: {sr}%)")
    
    # 5. Double Bottom/Top
    if len(lows_idx) >= 2:
        if abs(lows.tail(2).values[0] - lows.tail(2).values[1]) / current_price < 0.02:
            p = 'Double Bottom - Tiềm năng tăng'
            sr = get_success_rate(p)
            patterns.append(f"{p} (Success Rate: {sr}%)")
    if len(highs_idx) >= 2:
        if abs(highs.tail(2).values[0] - highs.tail(2).values[1]) / current_price < 0.02:
            p = 'Double Top - Tiềm năng giảm'
            sr = get_success_rate(p)
            patterns.append(f"{p} (Success Rate: {sr}%)")
    
    # 6. Triple
    if len(lows_idx) >= 3:
        recent_lows = lows.tail(3).values
        if np.std(recent_lows) / np.mean(recent_lows) < 0.02:
            p = 'Triple Bottom - Tiềm năng tăng'
            sr = get_success_rate(p)  # Map to Double Bottom if no exact
            patterns.append(f"{p} (Success Rate: {sr}%)")
    if len(highs_idx) >= 3:
        recent_highs = highs.tail(3).values
        if np.std(recent_highs) / np.mean(recent_highs) < 0.02:
            p = 'Triple Top - Tiềm năng giảm'
            sr = get_success_rate(p)
            patterns.append(f"{p} (Success Rate: {sr}%)")
    
    # 7. Flag
    body_pct = abs(df['close'] - df['open']) / df['open'] * 100
    if (body_pct > 3).any():
        # Volume not available, skip volume check for flag
        p = 'Bullish Flag - Tiềm năng tăng' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'Bearish Flag - Tiềm năng giảm'
        sr = get_success_rate(p)
        patterns.append(f"{p} (Success Rate: {sr}%)")
    
    # 8. Pennant
    if (body_pct > 3).any() and any('Triangle' in p for p in patterns):
        p = patterns[-1].replace('Triangle', 'Pennant')
        sr = get_success_rate(p)
        patterns[-1] = f"{p} (Success Rate: {sr}%)"
    
    # 9. Rectangle
    if len(highs_idx) >= 3 and len(lows_idx) >= 3:
        slope_h = linregress(range(3), highs.tail(3).values).slope
        slope_l = linregress(range(3), lows.tail(3).values).slope
        if abs(slope_h) < 0.001 and abs(slope_l) < 0.001:
            p = 'Rectangle - Sideways, chờ breakout'
            sr = get_success_rate(p)
            patterns.append(f"{p} (Success Rate: {sr}%)")
    
    # 10. Support/Resistance
    supp = lows.tail(5).mode().iloc[0] if len(lows) > 0 else None
    res = highs.tail(5).mode().iloc[0] if len(highs) > 0 else None
    if supp and abs(current_price - supp) / current_price < 0.01:
        p = 'Near Support - Tiềm năng bounce tăng'
        sr = get_success_rate(p)  # N/A
        patterns.append(f"{p} (Success Rate: {sr}%)")
    if res and abs(current_price - res) / current_price < 0.01:
        p = 'Near Resistance - Tiềm năng reject giảm'
        sr = get_success_rate(p)
        patterns.append(f"{p} (Success Rate: {sr}%)")
    
    # 11. Big Movement
    if body_pct.iloc[-1] > 3:
        p = 'Big Movement - Volatility cao'
        sr = get_success_rate(p)
        patterns.append(f"{p} (Success Rate: {sr}%)")
    
    # 12. Consecutive Candles
    df['body_dir'] = np.sign(df['close'] - df['open'])
    if (df['body_dir'].tail(3) == 1).all():
        p = 'Consecutive Bullish Candles - Tiềm năng tăng'
        sr = get_success_rate(p)
        patterns.append(f"{p} (Success Rate: {sr}%)")
    if (df['body_dir'].tail(3) == -1).all():
        p = 'Consecutive Bearish Candles - Tiềm năng giảm'
        sr = get_success_rate(p)
        patterns.append(f"{p} (Success Rate: {sr}%)")
    
    # 13-16. Harmonic
    harmonic = detect_harmonic(df, highs, lows, highs_idx, lows_idx)
    for h in harmonic:
        sr = get_success_rate(h)
        patterns.append(f"{h} (Success Rate: {sr}%)")
    
    return patterns, supp, res

def detect_channel(df, highs, lows, highs_idx, lows_idx, min_points=3):
    if len(highs_idx) < min_points or len(lows_idx) < min_points:
        return None
    x_range = range(min_points)
    y_h = highs.tail(min_points).values
    y_l = lows.tail(min_points).values
    res_h = linregress(x_range, y_h)
    res_l = linregress(x_range, y_l)
    if abs(res_h.slope - res_l.slope) < 0.01 and res_h.rvalue**2 > 0.8 and res_l.rvalue**2 > 0.8:
        direction = "Channel Up - Tiềm năng tăng" if res_h.slope > 0 else "Channel Down - Tiềm năng giảm"
        return {'pattern': 'Channel', 'direction': direction, 'slope': res_h.slope}
    return None

def detect_harmonic(df, highs, lows, highs_idx, lows_idx):
    patterns = []
    if len(highs_idx) < 3 or len(lows_idx) < 3:
        return patterns
    # Recent pivots (use numeric idx)
    recent_highs_idx = highs_idx[-3:]
    recent_lows_idx = lows_idx[-3:]
    pts_idx = sorted(list(recent_highs_idx) + list(recent_lows_idx))
    pts = [(i, df['close'].iloc[i], 'H' if i in recent_highs_idx else 'L') for i in pts_idx]
    
    if len(pts) >= 5:
        X, A, B, C, D = pts[-5:]
        if X[2]=='L' and A[2]=='H' and B[2]=='L' and C[2]=='H' and D[2]=='L':
            XA = A[1] - X[1]
            AB = B[1] - A[1]
            BC = C[1] - B[1]
            CD = D[1] - C[1]
            if 0.6 < abs(AB/XA) < 0.62 and 0.38 < abs(BC/AB) < 0.886 and 1.27 < abs(CD/BC) < 1.618:
                patterns.append('Bullish Gartley - Tiềm năng tăng')
            elif 0.78 < abs(AB/XA) < 0.79 and 0.38 < abs(BC/AB) < 0.886 and 1.61 < abs(CD/BC) < 2.618:
                patterns.append('Bullish Butterfly - Tiềm năng tăng')
    if len(pts) >= 4:
        A, B, C, D = pts[-4:]
        if A[2]=='L' and B[2]=='H' and C[2]=='L' and D[2]=='H':
            AB = B[1] - A[1]
            BC = C[1] - B[1]
            CD = D[1] - C[1]
            if 0.6 < abs(BC/AB) < 0.8 and 1.2 < abs(CD/BC) < 1.7:
                patterns.append('Bullish ABCD - Tiềm năng tăng')
    if len(pts) >= 3:
        A, B, C = pts[-3:]
        if A[2]=='L' and B[2]=='H' and C[2]=='L':
            AB = B[1] - A[1]
            retrace = (B[1] - C[1]) / AB
            if 0.5 < retrace < 0.618:
                patterns.append('3-Point Retracement - Tiềm năng tiếp tục tăng')
            ext = (C[1] - B[1]) / AB
            if 1.618 < abs(ext) < 2.618:
                patterns.append('3-Point Extension - Tiềm năng đảo chiều')
    
    return patterns

# Fixed plot with full NaN series for scatter
def plot_chart(df, highs_idx, lows_idx, patterns, supp=None, res=None):
    ap = []
    
    # Highs scatter
    high_plot = pd.Series(np.nan, index=df.index)
    high_plot.iloc[highs_idx] = df['high'].iloc[highs_idx]
    ap.append(mpf.make_addplot(high_plot, scatter=True, markersize=50, color='green', marker='^'))
    
    # Lows scatter
    low_plot = pd.Series(np.nan, index=df.index)
    low_plot.iloc[lows_idx] = df['low'].iloc[lows_idx]
    ap.append(mpf.make_addplot(low_plot, scatter=True, markersize=50, color='red', marker='v'))
    
    # Supp/Res hlines
    if supp:
        ap.append(mpf.make_addplot(pd.Series(supp, index=df.index), color='blue', linestyle='--', width=1))
    if res:
        ap.append(mpf.make_addplot(pd.Series(res, index=df.index), color='orange', linestyle='--', width=1))
    
    fig, axes = mpf.plot(df, type='candle', volume=True, addplot=ap, returnfig=True, 
                         title=f"Chart Patterns: {', '.join(patterns[:3])}...", style='charles')
    st.pyplot(fig)  # Fixed: use fig directly, as returnfig=True returns (fig, axes)

# Main app
st.title("Crypto Pattern Scanner GUI (Fixed with Success Rates)")

tab1, tab2 = st.tabs(["Top Coins", "Binance Alpha Coins"])

with tab1:
    vol_threshold = st.slider("Filter Volume Change 24h (%)", 0, 50, 10)

    with st.spinner("Đang load top coins..."):
        coins_df = get_top_coins(50, vol_threshold)
    if not coins_df.empty:
        # Apply styling
        def apply_styling(df):
            return (
                df.style
                .apply(color_market_cap, subset=['market_cap'], axis=0)
                .format({
                    'market_cap': '${:,.0f}',
                    'current_price': '${:,.4f}',
                    'price_change_24h': '{:.2f}%',
                    'volume_change_24h': '{:.2f}%',
                    'symbol': lambda x: x.replace('/USDT', '')
                })
            )
        
        styled_df = apply_styling(coins_df)
        st.dataframe(styled_df)

        selected_symbol = st.selectbox("Chọn Coin", coins_df['symbol'] if not coins_df.empty else [])
        timeframe = st.selectbox("Chọn Timeframe", ['5m', '15m', '1h', '4h', '1d'])

        if st.button("Scan Patterns"):
            with st.spinner("Đang fetch data và detect..."):
                # Use coin_id for fetch
                coin_row = coins_df[coins_df['symbol'] == selected_symbol].iloc[0]
                df = fetch_ohlcv(coin_row['id'], timeframe)
                if df.empty:
                    st.error(f"No data for {selected_symbol}")
                else:
                    highs_idx, lows_idx = find_pivots(df)
                    patterns, supp, res = detect_patterns(df, highs_idx, lows_idx)
                    
                    st.subheader("Patterns Tìm Thấy:")
                    if patterns:
                        for p in patterns:
                            st.write(f"- {p}")
                    else:
                        st.write("Không tìm thấy pattern nào rõ ràng.")
                    
                    if supp or res:
                        st.write(f"Support gần: ${supp:.2f} nếu có | Resistance gần: ${res:.2f} nếu có")
                    
                    # Plot fixed
                    plot_chart(df, highs_idx, lows_idx, patterns, supp, res)

with tab2:
    st.write("Danh sách Binance Alpha Coins (ưu tiên scan các coin mới list futures):")
    st.write(binance_alpha_coins)
    
    # Filter top coins for Alpha
    alpha_df = coins_df[coins_df['symbol'].str.replace('/USDT', '').str.upper().isin(binance_alpha_coins)]
    if not alpha_df.empty:
        styled_alpha = apply_styling(alpha_df)
        st.dataframe(styled_alpha)
        
        selected_alpha = st.selectbox("Chọn Binance Alpha Coin", alpha_df['symbol'] if not alpha_df.empty else [])
        timeframe_alpha = st.selectbox("Chọn Timeframe", ['5m', '15m', '1h', '4h', '1d'], key='alpha_tf')
        
        if st.button("Scan Binance Alpha Patterns"):
            with st.spinner("Đang fetch data và detect..."):
                coin_row = alpha_df[alpha_df['symbol'] == selected_alpha].iloc[0]
                df = fetch_ohlcv(coin_row['id'], timeframe_alpha)
                if df.empty:
                    st.error(f"No data for {selected_alpha}")
                else:
                    highs_idx, lows_idx = find_pivots(df)
                    patterns, supp, res = detect_patterns(df, highs_idx, lows_idx)
                    
                    st.subheader("Patterns Tìm Thấy cho Binance Alpha:")
                    if patterns:
                        for p in patterns:
                            st.write(f"- {p}")
                    else:
                        st.write("Không tìm thấy pattern nào rõ ràng.")
                    
                    if supp or res:
                        st.write(f"Support gần: ${supp:.2f} nếu có | Resistance gần: ${res:.2f} nếu có")
                    
                    # Plot fixed
                    plot_chart(df, highs_idx, lows_idx, patterns, supp, res)
    else:
        st.warning("Không tìm thấy Binance Alpha coins trong top list hiện tại. Danh sách có thể cần cập nhật.")
