import os
from datetime import date, timedelta
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stat
import seaborn as sns
import statsmodels as sm
import streamlit as st
import yfinance as yf

import yahoo_fin.stock_info as si


DISTRIBUTIONS = {
    "Johnson SU":stat.johnsonsu,
    "Cauchy":stat.cauchy,
    "Exponential Power":stat.exponpow,
    "Folded Cauchy":stat.foldcauchy,
    "Folded Normal":stat.foldnorm,
    "Normal":stat.norm,
    "Student's T":stat.t
}


@st.cache
def data_grab(ticker):
    df = yf.download(
        stock,
        end = today,
        progress = False
    ).astype("float64")

    price_cols = ["Open", "High", 'Low', "Close", "Adj Close"]
    return_cols = ['pct ' + col for col in price_cols]
    df[return_cols] = np.log(df[price_cols]/df[price_cols].shift(1))
    df = df.dropna()

    fig = px.histogram(
        df['pct Adj Close'],
        labels= {
            'count':"",
            'value':"% Daily Return"
        },
        title='% Daily Returns Distribution'
    )
    return fig, df


@st.cache
def annualVolatility(df):
    return df.std()*np.sqrt(252)


@st.cache
def cagr(prices):
    days = (prices.index[-1] - prices.index[0]).days
    cagr = (prices.iloc[-1] / prices.iloc[0]) ** (252/days) - 1
    return cagr


#@st.cache(allow_output_mutation=True)
def cagr_sim(df, forecast_len, num_sims=1000):
    result = []
    last_close = df['Adj Close'].iloc[-1]
    
    fig = make_subplots(1, 2, shared_yaxes=True)
    for i in range(num_sims):
        daily_returns = np.random.normal(cagr_val/252, vol/np.sqrt(252), forecast_len) + 1
        price_list = [last_close]
        for x in daily_returns:
            price_list.append(price_list[-1] * x)
        fig.append_trace(go.Scatter(y=price_list, mode="lines", name=f'Simulation {i+1}'), 1, 1)
        result.append(price_list[-1])
    
    fig.append_trace(go.Histogram(y=result), 1, 2)
    return fig, np.array(result)


@st.cache()
def fit_boots(df, fit_distr=stat.johnsonsu, num_bootstraps=10000):
    bootstrap_dist = np.random.choice(df['pct Adj Close'], num_bootstraps, replace=True)
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Histogram(x=bootstrap_dist, name='Bootstrapped Returns'), secondary_y=False)
    
    johnson_mean = np.mean(bootstrap_dist)
    johnson_std = np.std(bootstrap_dist)
    
    params = fit_distr.fit(bootstrap_dist)
    dist = fit_distr(*params)
    x = np.linspace(min(bootstrap_dist), max(bootstrap_dist), num=1000)
    y = [dist.pdf(x1) for x1 in x]
    fig1.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f'Fitted Distribution'), secondary_y=True)

    return fig1, dist


@st.cache(allow_output_mutation=True)
def bootstrap_sim(df, forecast_len, dist, num_sims=1000):
    result = []
    last_close = df['Adj Close'].iloc[-1]
    
    fig2 = make_subplots(1, 2, shared_yaxes=True)
    for i in range(num_sims):
        daily_returns = np.array(dist.rvs(size=forecast_len)) + 1
        price_list = [last_close]
        for x in daily_returns:
            price_list.append(price_list[-1] * x)
        price_list = np.array(price_list)
        price_list[np.where(price_list < 0)] = 0
        fig2.append_trace(go.Scatter(y=price_list, mode="lines", name=f'Simulation {i+1}'), 1, 1)
        result.append(price_list[-1])
    fig2.append_trace(go.Histogram(y=result), 1, 2)
    result = np.array(result)
    return fig1, fig2, result


########################################### Title Page ########################################################

st.title("Monte Carlo Options Spread Calculator")

today = date.today().strftime("%Y-%m-%d")

stock = st.text_input("Stock Ticker:", "GOOGL").upper()

fig, df = data_grab(stock)
#st.plotly_chart(fig)
st.line_chart(df['Adj Close'])

financials = si.get_quote_table(stock)
cagr_val = cagr(df['Adj Close'])
vol = annualVolatility(df['pct Adj Close'])

financials["Compound Annual Growth"] = cagr_val
financials["Annual Volatility"] = vol
for key in financials.keys():
    financials[key] = [financials[key]]


st.header(f"{stock} Summary")
st.table(pd.DataFrame(financials).T.rename(columns={0:""}))


############################################# SIDEBAR #########################################################
# Options Spread container
with st.sidebar.beta_container():
    st.sidebar.markdown('*Options Spread*')
    spread = st.sidebar.selectbox("Which Spread are you longing?", ["Call", "Put", "Straddle/Strangle", "Butterfly/Condor"])

    if (spread == "Call") or (spread == "Put"):
        strike = round(st.sidebar.number_input("What's your strike price?", 0.0, value=df['Adj Close'].iloc[-1]), 2)
        premium = round(st.sidebar.number_input('What is the contract premium?', 0.0, value=1.0), 2)
    else:
        lower = round(st.sidebar.number_input("What's your lower breakeven point?", 0.0, value=np.quantile(df['Adj Close'], 0.5)), 2)
        upper = round(st.sidebar.number_input("What's your upper breakeven point?", 0.0, value=np.quantile(df['Adj Close'], 0.95)), 2)

# Monte Carlo Parameters
with st.sidebar.beta_container():
    st.sidebar.markdown('*Simulation Settings*')
    forecast_len = st.sidebar.number_input("How many trading days to forecast?", 1, value=10)
    num_sims = st.sidebar.number_input('How many simulations to run? (More simulations take more time to render, but provide cleaner results)', 10, value=1000)
    # Advanced Settings
    with st.sidebar.beta_expander("Advanced Simulation settings"):
        boots = st.number_input("How many resamples to redraw?", 10000,value=max(10000, len(df)*2))
        fit_distr = DISTRIBUTIONS[st.selectbox("Which distribution to use for the Bootstrap Simulation?", list(DISTRIBUTIONS.keys()))]

########################################## MONTE CARLO SIMS ###################################################

st.header("Monte Carlo Simulations")
cagr_fig, cagr_dist = cagr_sim(df, forecast_len, num_sims)

if (spread == "Call"):
    target_price = strike + premium
    cagr_fig.add_hrect(y0=target_price, y1=max(cagr_dist), opacity=0.2, fillcolor="green", line_width=0)
    cagr_fig.add_hrect(y0=min(cagr_dist), y1=target_price, opacity=0.2, fillcolor="red", line_width=0)
    st.plotly_chart(cagr_fig)
    st.write(f"Chance that price falls above {target_price}:", str(len(cagr_dist[np.where(cagr_dist >= target_price)]) / len(cagr_dist) * 100)+"%")

elif (spread == "Put"):
    target_price = strike - premium
    cagr_fig.add_hrect(y0=target_price, y1=max(cagr_dist), opacity=0.2, fillcolor="red", line_width=0)
    cagr_fig.add_hrect(y0=min(cagr_dist), y1=target_price, opacity=0.2, fillcolor="green", line_width=0)
    st.plotly_chart(cagr_fig)
    st.write(f"Chance that price falls below {target_price}:", str(len(cagr_dist[np.where(cagr_dist <= target_price)]) / len(cagr_dist) * 100)+"%")

elif (spread == "Straddle/Strangle"):
    cagr_fig.add_hrect(y0=upper, y1=max(cagr_dist), opacity=0.2, fillcolor="green", line_width=0)
    cagr_fig.add_hrect(y0=lower, y1=upper, opacity=0.2, fillcolor="red", line_width=0)
    cagr_fig.add_hrect(y0=min(cagr_dist), y1=lower, opacity=0.2, fillcolor="green", line_width=0)
    st.plotly_chart(cagr_fig)
    st.write(f"Chance that price falls outside ${lower} and ${upper}:", str(len(cagr_dist[np.where((cagr_dist <= lower) | (cagr_dist >= upper))]) / len(cagr_dist) * 100)+"%")
    
elif (spread == "Butterfly/Condor"):
    cagr_fig.add_hrect(y0=upper, y1=max(cagr_dist), opacity=0.2, fillcolor="red", line_width=0)
    cagr_fig.add_hrect(y0=lower, y1=upper, opacity=0.2, fillcolor="green", line_width=0)
    cagr_fig.add_hrect(y0=min(cagr_dist), y1=lower, opacity=0.2, fillcolor="red", line_width=0)
    st.plotly_chart(cagr_fig)
    st.write(f"Chance your spread is profitable: ", str(len(cagr_dist[np.where((cagr_dist <= upper) & (cagr_dist >= lower))]) / len(cagr_dist)))

st.header("Bootstrap Simulations")
fig1, dist = fit_boots(df, fit_distr, num_bootstraps=boots)
fig1, boot_fig, boot_dist = bootstrap_sim(df, forecast_len, dist, num_sims=num_sims)
st.plotly_chart(fig1)


if (spread == "Call"):
    target_price = strike + premium
    boot_fig.add_hrect(y0=target_price, y1=max(boot_dist), opacity=0.2, fillcolor="green", line_width=0)
    boot_fig.add_hrect(y0=min(boot_dist), y1=target_price, opacity=0.2, fillcolor="red", line_width=0)
    st.plotly_chart(boot_fig)
    st.write(f"Chance that price falls above {target_price}:", str(len(boot_dist[np.where(boot_dist >= target_price)]) / len(boot_dist) * 100)+"%")

elif (spread == "Put"):
    target_price = strike - premium
    boot_fig.add_hrect(y0=target_price, y1=max(boot_dist), opacity=0.2, fillcolor="red", line_width=0)
    boot_fig.add_hrect(y0=min(boot_dist), y1=target_price, opacity=0.2, fillcolor="green", line_width=0)
    st.plotly_chart(boot_fig)
    st.write(f"Chance that price falls below {target_price}:", str(len(boot_dist[np.where(boot_dist <= target_price)]) / len(boot_dist) * 100)+"%")

elif (spread == "Straddle/Strangle"):
    boot_fig.add_hrect(y0=upper, y1=max(boot_dist), opacity=0.2, fillcolor="green", line_width=0)
    boot_fig.add_hrect(y0=lower, y1=upper, opacity=0.2, fillcolor="red", line_width=0)
    boot_fig.add_hrect(y0=min(boot_dist), y1=lower, opacity=0.2, fillcolor="green", line_width=0)
    st.plotly_chart(boot_fig)
    st.write(f"Chance that price falls outside ${lower} and ${upper}:", str(len(boot_dist[np.where((boot_dist <= lower) | (boot_dist >= upper))]) / len(boot_dist) * 100)+"%")
    
elif (spread == "Butterfly/Condor"):
    boot_fig.add_hrect(y0=upper, y1=max(boot_dist), opacity=0.2, fillcolor="red", line_width=0)
    boot_fig.add_hrect(y0=lower, y1=upper, opacity=0.2, fillcolor="green", line_width=0)
    boot_fig.add_hrect(y0=min(boot_dist), y1=lower, opacity=0.2, fillcolor="red", line_width=0)
    st.plotly_chart(boot_fig)
    st.write(f"Chance your spread is profitable: ", str(len(boot_dist[np.where((boot_dist <= upper) & (boot_dist >= lower))]) / len(boot_dist)))

####################################################### Notices #####################################################################
st.text(" \n")
st.text(" \n")
st.text(" \n")
st.text(" \n")
st.text(" \n")
st.text(" \n")
st.text(" \n")
st.text(" \n")
st.text(" \n")
st.markdown("""
<style>
.small-font {
    font-size:10px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='small-font'>You should not treat any forecast provided by this page as a specific inducement to make a particular investment or follow a particular strategy. This page's predictions are based upon information that is considered reliable, but does not warrant its completeness or accuracy, and it should not be relied upon as such. This page is not under any obligation to update or correct any information provided.</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>All forecasts shown on this page are based on past performance of the stock. Past performance is not indicative of future results. This page cannot guarantee any specific outcome or profit. You should be aware of the real risk of loss in following any prediction shown. Forecasts may fluctuate in price or value. Investors may get back less than invested. This material does not take into account your particular investment objectives, financial situation or needs and is not intended as recommendations appropriate for you. You must make an independent decision regarding forecasts provided. Before acting on information, you should consider whether it is suitable for your particular circumstances and strongly consider seeking advice from your own financial or investment adviser.</p>", unsafe_allow_html=True)

