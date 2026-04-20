import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings, io, os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Options Analytics Platform", page_icon="📈", layout="wide")
st.markdown("""<style>
.main{background-color:#0e1117;}
.stMetric{background-color:#1e2130;padding:10px;border-radius:5px;}
h1{color:#00ff00;text-align:center;}h2{color:#00d9ff;}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────────────────────
def bs_call(S,K,T,r,sigma):
    if T<=0: return max(S-K,0),0,0,0,0
    d1=(np.log(S/K)+(r+.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
    p=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return p,norm.cdf(d1),norm.pdf(d1)/(S*sigma*np.sqrt(T)),S*norm.pdf(d1)*np.sqrt(T)/100,(-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365

def bs_put(S,K,T,r,sigma):
    if T<=0: return max(K-S,0),0,0,0,0
    d1=(np.log(S/K)+(r+.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
    p=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return p,norm.cdf(d1)-1,norm.pdf(d1)/(S*sigma*np.sqrt(T)),S*norm.pdf(d1)*np.sqrt(T)/100,(-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365

def calc_iv(price,S,K,T,r,otype='call'):
    sigma=0.3
    for _ in range(100):
        fn=bs_call if otype=='call' else bs_put
        p,_,_,vega,_=fn(S,K,T,r,sigma)
        diff=p-price
        if abs(diff)<1e-5: return sigma
        if vega<1e-10: return sigma
        sigma=max(0.01,min(sigma-diff/(vega*100),5.0))
    return sigma

# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
def mc_option(S,K,T,r,sigma,otype='call',n=100000,barrier=None):
    dt=T/252; steps=int(T*252)
    Z=np.random.standard_normal((n,steps))
    paths=np.zeros((n,steps+1)); paths[:,0]=S
    for t in range(1,steps+1):
        paths[:,t]=paths[:,t-1]*np.exp((r-.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z[:,t-1])
    if barrier is None:
        pay=np.maximum(paths[:,-1]-K,0) if otype=='call' else np.maximum(K-paths[:,-1],0)
    else:
        hit=np.any(paths>barrier,axis=1) if otype=='call' else np.any(paths<barrier,axis=1)
        pay=np.where(hit,0,np.maximum(paths[:,-1]-K,0) if otype=='call' else np.maximum(K-paths[:,-1],0))
    return np.exp(-r*T)*np.mean(pay),np.std(pay)/np.sqrt(n),paths

def asian_mc(S,K,T,r,sigma,otype='call',n=50000):
    dt=T/252; steps=int(T*252)
    Z=np.random.standard_normal((n,steps))
    paths=np.zeros((n,steps+1)); paths[:,0]=S
    for t in range(1,steps+1):
        paths[:,t]=paths[:,t-1]*np.exp((r-.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z[:,t-1])
    avg=np.mean(paths,axis=1)
    pay=np.maximum(avg-K,0) if otype=='call' else np.maximum(K-avg,0)
    return np.exp(-r*T)*np.mean(pay),np.std(pay)/np.sqrt(n)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER — supports multiple Kaggle formats + built-in sample data
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_data():
    """
    Generate realistic built-in NIFTY option chain data so the app
    works immediately with zero downloads.
    Mimics a real NSE option chain snapshot with ~40 strikes around ATM.
    """
    np.random.seed(42)
    spot = 22450.0
    r    = 0.065
    # Use realistic recent vol regime
    base_iv = 0.14

    expiries = {
        "25-Apr-2024 (Near)":  7/365,
        "02-May-2024 (Weekly)": 14/365,
        "30-May-2024 (Monthly)": 40/365,
        "27-Jun-2024 (Far)":  68/365,
    }

    all_rows = []
    for expiry_label, tte in expiries.items():
        # Smile: higher IV for deep ITM/OTM
        strikes = np.arange(spot - 2000, spot + 2100, 100)
        for K in strikes:
            moneyness = K / spot
            # Skew: puts have higher IV (typical for indices)
            smile_ce = base_iv + 0.04 * (moneyness - 1)**2 - 0.01*(moneyness-1)
            smile_pe = base_iv + 0.05 * (moneyness - 1)**2 + 0.015*(1-moneyness)
            smile_ce = max(smile_ce, 0.05)
            smile_pe = max(smile_pe, 0.05)

            ce_price,_,_,_,_ = bs_call(spot, K, tte, r, smile_ce)
            pe_price,_,_,_,_ = bs_put(spot, K, tte, r, smile_pe)

            # Realistic OI — peaks at ATM, lower deep ITM/OTM
            atm_dist = abs(K - spot) / spot
            oi_scale  = np.exp(-8 * atm_dist) * np.random.uniform(0.7, 1.3)
            ce_oi = int(max(100, 800000 * oi_scale * np.random.uniform(0.8, 1.2)))
            pe_oi = int(max(100, 750000 * oi_scale * np.random.uniform(0.8, 1.2)))

            all_rows.append({
                "expiryDate":           expiry_label,
                "strikePrice":          float(K),
                "Underlying Value":     spot,
                "CE_lastPrice":         round(max(ce_price * np.random.uniform(0.97,1.03), 0.05), 2),
                "CE_openInterest":      ce_oi,
                "CE_changeinOI":        int(ce_oi * np.random.uniform(-0.1, 0.15)),
                "CE_totalTradedVolume": int(ce_oi * np.random.uniform(0.3, 0.8)),
                "CE_impliedVolatility": round(smile_ce * 100, 2),
                "PE_lastPrice":         round(max(pe_price * np.random.uniform(0.97,1.03), 0.05), 2),
                "PE_openInterest":      pe_oi,
                "PE_changeinOI":        int(pe_oi * np.random.uniform(-0.1, 0.15)),
                "PE_totalTradedVolume": int(pe_oi * np.random.uniform(0.3, 0.8)),
                "PE_impliedVolatility": round(smile_pe * 100, 2),
            })

    return pd.DataFrame(all_rows), spot

def transform_intraday_options(df_raw):
    cols = [c.lower() for c in df_raw.columns]

    if not all(x in cols for x in ["strike_price", "right", "close"]):
        return None, None

    df = df_raw.copy()

    # take latest snapshot
    if "datetime" in df.columns:
        df = df.sort_values("datetime")
        df = df.groupby(["strike_price", "right"]).tail(1)

    pivot = df.pivot_table(
        index="strike_price",
        columns="right",
        values="close",
        aggfunc="last"
    ).reset_index()

    pivot.columns.name = None

    pivot.rename(columns={
        "strike_price": "strikePrice",
        "CE": "CE_lastPrice",
        "PE": "PE_lastPrice"
    }, inplace=True)

    pivot["CE_lastPrice"] = pivot.get("CE_lastPrice", 0).fillna(0)
    pivot["PE_lastPrice"] = pivot.get("PE_lastPrice", 0).fillna(0)

    pivot["expiryDate"] = "Sample Expiry"
    spot = pivot["strikePrice"].median()

    return _fill_defaults(pivot), spot

def try_parse_kaggle(df_raw):
    """
    Try to normalise several known Kaggle NSE options dataset formats
    into the standard internal schema.
    Returns (df_normalised, spot_price) or raises ValueError with a helpful message.
    """
    cols = [c.strip().upper() for c in df_raw.columns]
    df_raw.columns = [c.strip() for c in df_raw.columns]

    # ── Format A: Indian Nifty and Banknifty Options Data 2020-2024
    # Columns: Date, Symbol, Expiry, Strike, CE_LTP, CE_OI, CE_IV, PE_LTP, PE_OI, PE_IV, Underlying
    if any('CE_LTP' in c.upper() or 'CE LTP' in c.upper() for c in df_raw.columns):
        col_map = {}
        for c in df_raw.columns:
            cu = c.upper().replace(" ","_")
            if 'STRIKE' in cu:              col_map[c] = 'strikePrice'
            elif 'EXPIRY' in cu:            col_map[c] = 'expiryDate'
            elif 'CE_LTP' in cu or ('CE' in cu and 'LTP' in cu):   col_map[c] = 'CE_lastPrice'
            elif 'PE_LTP' in cu or ('PE' in cu and 'LTP' in cu):   col_map[c] = 'PE_lastPrice'
            elif 'CE_OI'  in cu or ('CE' in cu and cu.endswith('OI')): col_map[c] = 'CE_openInterest'
            elif 'PE_OI'  in cu or ('PE' in cu and cu.endswith('OI')): col_map[c] = 'PE_openInterest'
            elif 'CE_IV'  in cu or ('CE' in cu and 'IV' in cu):    col_map[c] = 'CE_impliedVolatility'
            elif 'PE_IV'  in cu or ('PE' in cu and 'IV' in cu):    col_map[c] = 'PE_impliedVolatility'
            elif 'UNDERLYING' in cu or 'SPOT' in cu or 'CLOSE' in cu: col_map[c] = 'Underlying Value'
        df = df_raw.rename(columns=col_map)
        spot = df['Underlying Value'].dropna().iloc[-1] if 'Underlying Value' in df.columns else None
        return _fill_defaults(df), spot

    # ── Format B: NSE F&O Bhavcopy style
    # Columns: SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, CLOSE, OPEN_INT, ...
    if 'OPTION_TYP' in cols or 'OPTIONTYPE' in cols:
        ot_col = next(c for c in df_raw.columns if c.upper().replace("_","") in ('OPTIONTYP','OPTIONTYPE'))
        str_col = next((c for c in df_raw.columns if 'STRIKE' in c.upper()), None)
        exp_col = next((c for c in df_raw.columns if 'EXPIRY' in c.upper()), None)
        close_col = next((c for c in df_raw.columns if c.upper() in ('CLOSE','LTP','SETTLE_PR','SETTLEPRICE')), None)
        oi_col  = next((c for c in df_raw.columns if 'OPEN_INT' in c.upper() or c.upper()=='OPEN_INT'), None)

        if not all([str_col, exp_col, close_col]):
            raise ValueError("Could not identify required columns in bhavcopy format.")

        calls = df_raw[df_raw[ot_col].str.upper().str.strip()=='CE'].copy()
        puts  = df_raw[df_raw[ot_col].str.upper().str.strip()=='PE'].copy()

        def agg(side):
            g = side.groupby([exp_col, str_col]).agg(
                ltp=(close_col,'last'),
                oi=(oi_col,'last') if oi_col else (close_col,'count')
            ).reset_index()
            g.columns = ['expiryDate','strikePrice','lastPrice','openInterest']
            return g

        ce = agg(calls); pe = agg(puts)
        merged = pd.merge(ce, pe, on=['expiryDate','strikePrice'], suffixes=('_CE','_PE'))
        merged = merged.rename(columns={
            'lastPrice_CE':'CE_lastPrice','openInterest_CE':'CE_openInterest',
            'lastPrice_PE':'PE_lastPrice','openInterest_PE':'PE_openInterest',
        })
        return _fill_defaults(merged), None

    # ── Format C: Historical Nifty Options 2024 All Expiries
    # Columns: Date, Expiry, Strike Price, CE LTP, CE OI, CE Volume, CE IV, PE LTP, PE OI, PE Volume, PE IV, Nifty
    col_lower = {c: c.lower().replace(" ","").replace("_","") for c in df_raw.columns}
    rev = {v: k for k, v in col_lower.items()}

    if 'celtp' in rev or 'celtprice' in rev:
        mapper = {
            rev.get('strikeprice', rev.get('strike','')): 'strikePrice',
            rev.get('expiry', rev.get('expirydate','')): 'expiryDate',
            rev.get('celtp', rev.get('celtprice','')): 'CE_lastPrice',
            rev.get('peltp', rev.get('peltprice','')): 'PE_lastPrice',
            rev.get('ceoi',''):  'CE_openInterest',
            rev.get('peoi',''):  'PE_openInterest',
            rev.get('ceiv',''):  'CE_impliedVolatility',
            rev.get('peiv',''):  'PE_impliedVolatility',
            rev.get('nifty', rev.get('underlying', rev.get('close',''))): 'Underlying Value',
        }
        mapper = {k: v for k,v in mapper.items() if k}
        df = df_raw.rename(columns=mapper)
        spot = df['Underlying Value'].dropna().iloc[-1] if 'Underlying Value' in df.columns else None
        return _fill_defaults(df), spot

    raise ValueError(
        "Unrecognised CSV format.\n\n"
        "Supported datasets:\n"
        "• Indian Nifty and Banknifty Options Data 2020-2024\n"
        "• Historical Nifty Options 2024 All Expiries\n"
        "• NSE F&O Bhavcopy CSV\n\n"
        f"Columns found: {list(df_raw.columns[:10])}"
    )

def _fill_defaults(df):
    """Ensure all expected columns exist, filling missing ones with 0."""
    needed = [
        'strikePrice','expiryDate',
        'CE_lastPrice','CE_openInterest','CE_changeinOI','CE_totalTradedVolume','CE_impliedVolatility',
        'PE_lastPrice','PE_openInterest','PE_changeinOI','PE_totalTradedVolume','PE_impliedVolatility',
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = 0
    df['strikePrice'] = pd.to_numeric(df['strikePrice'], errors='coerce')
    df = df.dropna(subset=['strikePrice'])
    return df

def compute_iv_if_missing(df, spot, r=0.065):
    """If IV columns are all zero, compute via Black-Scholes."""
    if df['CE_impliedVolatility'].sum() == 0 and 'expiryDate' in df.columns:
        for idx, row in df.iterrows():
            try:
                expiry_str = str(row['expiryDate'])
                for fmt in ('%d-%b-%Y','%Y-%m-%d','%d/%m/%Y','%d-%b-%y'):
                    try:
                        exp_dt = datetime.strptime(expiry_str.split()[0], fmt)
                        tte = max((exp_dt - datetime.now()).days / 365, 0.003)
                        break
                    except: tte = 0.08
                if row['CE_lastPrice'] > 0:
                    df.at[idx,'CE_impliedVolatility'] = round(calc_iv(row['CE_lastPrice'], spot, row['strikePrice'], tte, r, 'call')*100, 2)
                if row['PE_lastPrice'] > 0:
                    df.at[idx,'PE_impliedVolatility'] = round(calc_iv(row['PE_lastPrice'], spot, row['strikePrice'], tte, r, 'put')*100, 2)
            except: pass
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
st.title("Options Analytics Platform")
st.markdown("### **Derivatives Pricing & Risk Management — Indian Index (NIFTY / BANKNIFTY)**")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Black-Scholes Pricer",
    "Volatility Surface",
    "Exotic Options",
    "Risk Dashboard",
    "Portfolio Greeks",
    "Strategy Analysis",
    "IV Analysis"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Black-Scholes Pricer
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Black-Scholes Option Pricing")

    df_chain = st.session_state.get("df_chain")
    spot_data = st.session_state.get("spot_price")

    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.number_input("Spot Price (₹)", value=spot_data if spot_data else 22450.0)
        K = st.number_input("Strike Price (₹)", value=22500.0)

    if df_chain is not None:
        st.markdown("### Market Comparison")

        strikes = sorted(df_chain["strikePrice"].unique())
        selected_strike = st.selectbox("Select Market Strike", strikes)

        row = df_chain[df_chain["strikePrice"] == selected_strike].iloc[0]

        st.write(f"Market Call Price: ₹{row['CE_lastPrice']}")
        st.write(f"Market Put Price: ₹{row['PE_lastPrice']}")
    with col2:
        T = st.number_input("Time to Maturity (years)", value=0.08, min_value=0.003, max_value=5.0)
        r = st.number_input("Risk-Free Rate (%)", value=6.5, min_value=0.0) / 100
    with col3:
        sigma    = st.number_input("Volatility (%)", value=14.0, min_value=1.0, max_value=200.0) / 100
        opt_type = st.selectbox("Option Type", ["Call", "Put"])

    fn = bs_call if opt_type == "Call" else bs_put
    price, delta, gamma, vega, theta = fn(S, K, T, r, sigma)

    st.markdown("---")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Option Price", f"₹{price:.2f}")
    c2.metric("Delta", f"{delta:.4f}")
    c3.metric("Gamma", f"{gamma:.6f}")
    c4.metric("Vega", f"{vega:.4f}")
    c5.metric("Theta", f"{theta:.4f}")

    st.markdown("### Greeks Sensitivity Analysis")
    spot_range = np.linspace(S*0.7, S*1.3, 60)
    prices, deltas, gammas = [], [], []
    for s in spot_range:
        p,d,g,_,_ = fn(s,K,T,r,sigma)
        prices.append(p); deltas.append(d); gammas.append(g)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=prices, name='Option Price', line=dict(color='#00ff00', width=3)))
    fig.add_trace(go.Scatter(x=spot_range, y=deltas, name='Delta',        line=dict(color='#00d9ff', width=2)))
    fig.add_trace(go.Scatter(x=spot_range, y=gammas, name='Gamma',        line=dict(color='#ff00ff', width=2)))
    fig.add_vline(x=S, line_dash="dash", line_color="red", annotation_text="Current Spot")
    fig.update_layout(title="Option Price & Greeks vs Spot Price",
                      xaxis_title="Spot Price (₹)", yaxis_title="Value",
                      template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True, key="greeks_chart")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Volatility Surface (Dataset-driven)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Volatility Surface Builder")

    st.markdown("### Data Source")
    data_source = st.radio(
        "Choose data source",
        ["Built-in Sample Data (works immediately)",
         "Upload Kaggle CSV"],
        horizontal=True
    )

    df_chain = None
    spot_price = None

    if "Built-in" in data_source:
        df_chain, spot_price = generate_sample_data()
        st.success("Built-in NIFTY sample data loaded")

    else:
        st.markdown("""
**Download a free dataset:**
| Dataset | Link |
|---|---|
| Historical Nifty Options 2024 | Kaggle |
| NSE F&O Bhavcopy | NSE |
""")

        uploaded = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload Kaggle dataset"
        )

        if uploaded:
            try:
                raw = pd.read_csv(uploaded)

                st.info(f"Loaded {len(raw):,} rows, {len(raw.columns)} columns")

                with st.expander("Preview raw data"):
                    st.dataframe(raw.head(5), use_container_width=True)

                # Handle dataset
                df_chain, spot_from_data = transform_intraday_options(raw)

                # fallback
                if df_chain is None:
                    df_chain, spot_from_data = try_parse_kaggle(raw)

                col_spot1, col_spot2 = st.columns([2,1])
                with col_spot1:
                    default_spot = float(spot_from_data) if spot_from_data else 22450.0
                    spot_price = st.number_input(
                        "Underlying Spot Price (₹)",
                        value=default_spot,
                        min_value=1.0
                    )

                st.success(f"Parsed successfully — {len(df_chain):,} option rows")
                st.session_state["df_chain"] = df_chain
                st.session_state["spot_price"] = spot_price

            except Exception as e:
                st.error(f"Could not parse file: {e}")

    # ONLY RUN CHARTS IF DATA EXISTS
    if df_chain is not None and spot_price is not None:

        expiries = sorted(df_chain['expiryDate'].dropna().unique())
        selected_expiry = st.selectbox("Select Expiry", expiries)

        df_exp = df_chain[df_chain['expiryDate'] == selected_expiry].copy()

        # TTE
        try:
            exp_str = str(selected_expiry).split()[0]
            exp_dt = datetime.strptime(exp_str, '%d-%b-%Y')
            tte = max((exp_dt - datetime.now()).days / 365, 0.003)
        except:
            tte = 0.08

        df_exp = compute_iv_if_missing(df_exp.copy(), spot_price)

        atm_strike = df_exp.iloc[(df_exp['strikePrice']-spot_price).abs().argsort()[:1]]['strikePrice'].values[0]
        atm_row = df_exp[df_exp['strikePrice']==atm_strike].iloc[0]

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Spot Price", f"₹{spot_price:,.2f}")
        c2.metric("ATM Strike", f"₹{atm_strike:,.0f}")
        c3.metric("ATM IV (CE)", f"{atm_row['CE_impliedVolatility']:.1f}%")
        c4.metric("ATM IV (PE)", f"{atm_row['PE_impliedVolatility']:.1f}%")

        total_ce_oi = df_exp['CE_openInterest'].sum()
        total_pe_oi = df_exp['PE_openInterest'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

        c5.metric("PCR", f"{pcr:.2f}")

        # IV Smile
        st.markdown("### IV Smile")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_exp['strikePrice'],
            y=df_exp['CE_impliedVolatility'],
            name="Call IV"
        ))
        fig.add_trace(go.Scatter(
            x=df_exp['strikePrice'],
            y=df_exp['PE_impliedVolatility'],
            name="Put IV"
        ))

        st.plotly_chart(fig, use_container_width=True, key="iv_smile")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Exotic Options
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Exotic Options Pricing (Monte Carlo)")
    exotic_type = st.selectbox("Option Type", [
        "Barrier Option (Up-and-Out Call)",
        "Barrier Option (Down-and-Out Put)",
        "Asian Option (Arithmetic Average)",
    ])
    col1, col2, col3 = st.columns(3)
    with col1:
        Se = st.number_input("Spot Price (₹)",   value=22450.0, key='eS')
        Ke = st.number_input("Strike Price (₹)", value=22500.0, key='eK')
    with col2:
        Te = st.number_input("Time to Maturity (years)", value=0.08, key='eT')
        re = st.number_input("Risk-Free Rate",            value=0.065, key='er')
    with col3:
        se   = st.number_input("Volatility", value=0.14, key='es')
        nsim = st.selectbox("Simulations", [10000, 50000, 100000], index=1)

    if "Barrier" in exotic_type:
        barrier = st.number_input("Barrier Level (₹)",
                                   value=24000.0 if "Up" in exotic_type else 20500.0)

    if st.button("Price Exotic Option"):
        with st.spinner("Running Monte Carlo..."):
            if "Asian" in exotic_type:
                p, se2 = asian_mc(Se, Ke, Te, re, se, 'call', nsim)
                st.success(f"**Asian Option Price: ₹{p:.2f} ± ₹{se2:.2f}**")
            else:
                ot = 'call' if 'Call' in exotic_type else 'put'
                p, se2, paths = mc_option(Se, Ke, Te, re, se, ot, nsim, barrier)
                st.success(f"**Barrier Option Price: ₹{p:.2f} ± ₹{se2:.2f}**")
                fig = go.Figure()
                ts = np.linspace(0, Te, paths.shape[1])
                for path in paths[:100]:
                    fig.add_trace(go.Scatter(x=ts, y=path, mode='lines',
                                             line=dict(width=0.5), showlegend=False, opacity=0.3))
                fig.add_hline(y=barrier, line_dash="dash", line_color="red",
                              annotation_text=f"Barrier: ₹{barrier:,.0f}")
                fig.add_hline(y=Se, line_dash="dash", line_color="green",
                              annotation_text=f"Initial: ₹{Se:,.0f}")
                fig.update_layout(title="Sample Monte Carlo Paths",
                                  xaxis_title="Time (years)", yaxis_title="Index Level (₹)",
                                  template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True, key="mc_paths")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Risk Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Portfolio Risk Management")
    st.markdown("### Value at Risk (VaR) Calculator")
    col1, col2 = st.columns(2)
    with col1:
        pv = st.number_input("Portfolio Value (₹)", value=1_000_000.0, min_value=1000.0)
        hp = st.number_input("Holding Period (days)", value=1, min_value=1, max_value=252)
    with col2:
        cl  = st.selectbox("Confidence Level", [90, 95, 99], index=1)
        vol = st.number_input("Daily Volatility (%)", value=1.2, min_value=0.1) / 100

    if st.button("Calculate VaR"):
        ret   = np.random.normal(0, vol, (10000, hp))
        losses = pv - pv*(1+np.sum(ret,axis=1))
        var   = np.percentile(losses, cl)
        cvar  = np.mean(losses[losses>=var])
        c1,c2,c3 = st.columns(3)
        c1.metric(f"VaR ({cl}%)",  f"₹{var:,.0f}")
        c2.metric(f"CVaR ({cl}%)", f"₹{cvar:,.0f}")
        c3.metric("Max Loss",      f"₹{np.max(losses):,.0f}")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=losses, nbinsx=60, marker_color='#00d9ff'))
        fig.add_vline(x=var, line_dash="dash", line_color="red",
                      annotation_text=f"VaR ₹{var:,.0f}")
        fig.update_layout(title="Portfolio Loss Distribution", xaxis_title="Loss (₹)",
                          yaxis_title="Frequency", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True, key="var_chart")

    st.markdown("---")
    st.markdown("### Stress Testing Scenarios")
    scenarios = {
        "COVID Crash Mar 2020 (-38%)":        -0.38,
        "Demonetisation Nov 2016 (-6%)":      -0.06,
        "2008 Global Crisis (-60%)":          -0.60,
        "Yes Bank Crisis (-45%)":             -0.45,
        "Budget Shock (-4% single day)":      -0.04,
        "Bull Rally (+20%)":                   0.20,
        "Volatility Spike (+100% VIX)":        0.0,
    }
    srows = []
    for name, shock in scenarios.items():
        sv  = pv if shock == 0.0 else pv*(1+shock)
        pnl = 0 if shock == 0.0 else sv-pv
        srows.append({'Scenario': name, 'Shock': f"{shock*100:.0f}%" if shock != 0.0 else "Vol x2",
                      'Portfolio Value': f"₹{sv:,.0f}", 'P&L': f"₹{pnl:,.0f}"})
    st.dataframe(pd.DataFrame(srows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Portfolio Greeks
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Portfolio Greeks Aggregation")
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []

    with st.form("add_position"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pos_type   = st.selectbox("Type", ["Call", "Put"])
            pos_strike = st.number_input("Strike (₹)", value=22500.0)
        with col2:
            pos_spot     = st.number_input("Spot (₹)",  value=22450.0)
            pos_quantity = st.number_input("Lots (1 lot = 50)", value=1, min_value=-500, max_value=500)
        with col3:
            pos_expiry = st.number_input("Expiry (years)", value=0.08, min_value=0.003)
            pos_vol    = st.number_input("Vol (%)", value=14.0) / 100
        with col4:
            st.markdown("##")
            submitted = st.form_submit_button("Add Position")

    if submitted:
        st.session_state.portfolio.append({
            'type': pos_type, 'strike': pos_strike, 'spot': pos_spot,
            'quantity': pos_quantity, 'expiry': pos_expiry, 'vol': pos_vol
        })
        st.success("Position added!")

    if st.session_state.portfolio:
        total_delta = total_gamma = total_vega = total_theta = total_value = 0
        pdata = []
        LOT_SIZE = 50  # NIFTY lot size
        for i, pos in enumerate(st.session_state.portfolio):
            fn2 = bs_call if pos['type']=='Call' else bs_put
            price, delta, gamma, vega, theta = fn2(
                pos['spot'], pos['strike'], pos['expiry'], 0.065, pos['vol']
            )
            q = pos['quantity'] * LOT_SIZE
            total_value += price*q; total_delta += delta*q
            total_gamma += gamma*q; total_vega  += vega*q; total_theta += theta*q
            pdata.append({
                '#': i+1, 'Type': pos['type'],
                'Strike': f"₹{pos['strike']:,.0f}",
                'Lots': pos['quantity'],
                'Value': f"₹{price*q:,.0f}",
                'Delta': f"{delta*q:.2f}",
                'Gamma': f"{gamma*q:.4f}",
                'Vega':  f"{vega*q:.2f}",
                'Theta': f"{theta*q:.2f}",
            })

        st.dataframe(pd.DataFrame(pdata), use_container_width=True)
        st.markdown("### Portfolio-Level Greeks")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Value", f"₹{total_value:,.0f}")
        c2.metric("Net Delta",   f"{total_delta:.2f}")
        c3.metric("Net Gamma",   f"{total_gamma:.4f}")
        c4.metric("Net Vega",    f"{total_vega:.2f}")
        c5.metric("Net Theta",   f"{total_theta:.2f}")

        fig = go.Figure(data=[go.Bar(
            x=['Delta','Gamma','Vega','Theta'],
            y=[total_delta, total_gamma, total_vega, total_theta],
            marker_color=['#00ff00','#00d9ff','#ff00ff','#ffaa00']
        )])
        fig.update_layout(title="Portfolio Greeks Breakdown",
                          yaxis_title="Value", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True, key="portfolio_greeks")

        if st.button("Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Strategy Analysis 
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.header(" Strategy Analysis")

    df_chain = st.session_state.get("df_chain")
    spot_price = st.session_state.get("spot_price")

    if df_chain is None:
        st.warning("Upload data in Tab 2 first")
    else:
        strikes = sorted(df_chain["strikePrice"].unique())

        # 🔥 Helper: nearest strike
        def get_nearest_strike(strikes, target):
            return min(strikes, key=lambda x: abs(x - target))

        # Strategy selector
        strategy = st.selectbox(
            "Select Strategy",
            ["Straddle", "Strangle", "Call Spread", "Put Spread"]
        )

        K = st.selectbox("Select Base Strike", strikes)

        S_range = np.linspace(0.8 * spot_price, 1.2 * spot_price, 100)

        # ================= STRATEGY LOGIC =================

        if strategy == "Straddle":
            row = df_chain[df_chain["strikePrice"] == K].iloc[0]
            call_price = row["CE_lastPrice"]
            put_price = row["PE_lastPrice"]

            cost = call_price + put_price
            st.write(f"Straddle Cost: ₹{cost:.2f}")

            pnl = (
                np.maximum(S_range - K, 0) +
                np.maximum(K - S_range, 0)
                - cost
            )

        elif strategy == "Strangle":
            K_call = get_nearest_strike(strikes, K + 200)
            K_put  = get_nearest_strike(strikes, K - 200)

            row_call = df_chain[df_chain["strikePrice"] == K_call].iloc[0]
            row_put  = df_chain[df_chain["strikePrice"] == K_put].iloc[0]

            call_price = row_call["CE_lastPrice"]
            put_price  = row_put["PE_lastPrice"]

            cost = call_price + put_price
            st.write(f"Strangle Cost: ₹{cost:.2f}")
            st.caption(f"Using strikes: Call={K_call}, Put={K_put}")

            pnl = (
                np.maximum(S_range - K_call, 0) +
                np.maximum(K_put - S_range, 0)
                - cost
            )

        elif strategy == "Call Spread":
            K2 = get_nearest_strike(strikes, K + 200)

            row1 = df_chain[df_chain["strikePrice"] == K].iloc[0]
            row2 = df_chain[df_chain["strikePrice"] == K2].iloc[0]

            c1 = row1["CE_lastPrice"]
            c2 = row2["CE_lastPrice"]

            cost = c1 - c2
            st.write(f"Call Spread Cost: ₹{cost:.2f}")
            st.caption(f"Using strikes: Buy={K}, Sell={K2}")

            pnl = (
                np.maximum(S_range - K, 0) -
                np.maximum(S_range - K2, 0)
                - cost
            )

        elif strategy == "Put Spread":
            K2 = get_nearest_strike(strikes, K - 200)

            row1 = df_chain[df_chain["strikePrice"] == K].iloc[0]
            row2 = df_chain[df_chain["strikePrice"] == K2].iloc[0]

            p1 = row1["PE_lastPrice"]
            p2 = row2["PE_lastPrice"]

            cost = p1 - p2
            st.write(f"Put Spread Cost: ₹{cost:.2f}")
            st.caption(f"Using strikes: Buy={K}, Sell={K2}")

            pnl = (
                np.maximum(K - S_range, 0) -
                np.maximum(K2 - S_range, 0)
                - cost
            )

        # ================= PLOT =================

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=S_range,
            y=pnl,
            name=strategy,
            line=dict(width=3)
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="white")

        fig.update_layout(
            title=f"{strategy} Payoff",
            xaxis_title="Spot Price (₹)",
            yaxis_title="Profit / Loss (₹)",
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True, key="strategy_pnl")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — IV Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("IV Analysis")

    df_chain = st.session_state.get("df_chain")
    spot_price = st.session_state.get("spot_price")

    if df_chain is None:
        st.warning("Upload data first")
    else:
        df_chain = compute_iv_if_missing(df_chain.copy(), spot_price)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_chain["strikePrice"],
            y=df_chain["CE_impliedVolatility"],
            name="Call IV"
        ))
        fig.add_trace(go.Scatter(
            x=df_chain["strikePrice"],
            y=df_chain["PE_impliedVolatility"],
            name="Put IV"
        ))

        st.plotly_chart(fig, use_container_width=True, key="iv_analysis")

st.markdown("---")
st.markdown("<center style='color:#555'>Options Analytics Platform • Built-in data or Kaggle CSV upload</center>",
            unsafe_allow_html=True)