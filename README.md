#  Options Analytics Platform (NIFTY / BANKNIFTY)

An end-to-end options analytics platform built using Python and Streamlit, designed for analyzing Indian index derivatives.
Link to live demo : https://options-analytics-platform-nse-data-esv2mprcdf2srxurneracn.streamlit.app/

##  Features

- Black-Scholes pricing engine
- Volatility surface construction
- Implied volatility (IV) computation
- Strategy analysis (Straddle, Strangle, Spreads)
- Risk dashboard (VaR, distributions)
- Portfolio Greeks visualization
- Works with real intraday NIFTY options data

## Data

Uses historical intraday options data (Kaggle).  
Use : https://www.kaggle.com/datasets/senthilkumarvaithi/historical-nifty-options-2024-all-expiries.

Transforms raw time-series into option chain format.

##  Key Highlights

- Dynamic strike selection (robust to missing data)
- Real-market payoff simulation
- IV smile + surface visualization
- Modular multi-tab architecture

## Run locally 
```bash
pip install -r requirements.txt
streamlit run app.py
