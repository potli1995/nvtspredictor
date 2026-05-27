# NVTS Daily Predictor
A local Streamlit app focused on Navitas Semiconductor ( NVTS ).
What it does
• Pulls daily NVTS price data from Yahoo Finance through  yfinance 
• Trains an ensemble model:
• Random Forest
• Gradient Boosting
• Predicts probability that the next daily close is higher than the latest close
• Shows BUY / NEUTRAL / AVOID signal
• Shows backtest accuracy, support/resistance, and a recent intraday chart
Install
pip install -r requirements.txt
Run
streamlit run navitas_nvts_predictor.py
Important
This is not financial advice. It is an educational research tool. NVTS is volatile and can move sharply on news, market sentiment, semiconductor momentum, and earnings expectations.
