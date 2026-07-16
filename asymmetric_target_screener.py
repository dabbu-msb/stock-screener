"""
Asymmetric Analyst Target Screener
==================================
A single-file Streamlit app that screens the S&P 500 + NASDAQ 100 universe for
stocks with asymmetric analyst price-target profiles:

  1. LOW TARGET FLOOR   -> current price <= Analyst LOW target * (1 + low_margin)
                           (i.e., price is at/near/below the most pessimistic target)
  2. AVERAGE UPSIDE     -> Analyst MEAN target >= current price * (1 + avg_upside_min)
  3. HIGH UPSIDE        -> Analyst HIGH target >= current price * (1 + high_upside_min)

Data source: yfinance (free, no API key required).

Run:
    pip install streamlit yfinance pandas lxml requests
    streamlit run asymmetric_target_screener.py
"""

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Asymmetric Analyst Target Screener",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Asymmetric Analyst Target Screener")
st.caption(
    "Finds stocks trading at/near the analyst LOW target (limited modeled downside) "
    "with large consensus and high-target upside skew. Universe: S&P 500 + NASDAQ 100. "
    "Data: yfinance (delayed quotes)."
)

# ----------------------------------------------------------------------------
# Universe construction (S&P 500 + NASDAQ 100 from Wikipedia, with fallback)
# ----------------------------------------------------------------------------
FALLBACK_TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE",
    "ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG",
    "AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL",
    "ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC",
    "BA","BCR","BKNG","BWA","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR","BG","CDNS","CZR","CPT",
    "CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","CNC","CNX","CDAY",
    "CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX",
    "CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY",
    "CTVA","CSGP","COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DECK",
    "DE","DAL","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE",
    "DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM",
    "EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM",
    "FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV",
    "FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC",
    "GILD","GPN","GL","GDDY","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT",
    "HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW",
    "INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBAL",
    "JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KKR",
    "KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV","LKQ","LMT",
    "L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC",
    "MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH",
    "TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA",
    "NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY",
    "OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW","PH","PAYX","PAYC","PYPL",
    "PNR","PEP","PFE","PCG","PM","PSX","PNW","PXD","PNC","POOL","PPG","PPL","PFG","PG","PGR",
    "PLD","PRU","PEG","PTVE","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O",
    "REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB",
    "STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT",
    "STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL",
    "TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL",
    "TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN",
    "VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD",
    "WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WHR","WMB","WTW","WYNN","XEL","XYL","YUM",
    "ZBRA","ZBH","ZTS",
]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_universe() -> list[str]:
    """S&P 500 + NASDAQ 100 tickers scraped from Wikipedia (cached 24h)."""
    tickers: set[str] = set()
    try:
        sp500 = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        tickers.update(sp500["Symbol"].astype(str).str.strip().tolist())
    except Exception:
        pass
    try:
        ndx_tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for tbl in ndx_tables:
            cols = [str(c).lower() for c in tbl.columns]
            if "ticker" in cols or "symbol" in cols:
                col = tbl.columns[cols.index("ticker") if "ticker" in cols else cols.index("symbol")]
                tickers.update(tbl[col].astype(str).str.strip().tolist())
                break
    except Exception:
        pass

    if not tickers:
        return FALLBACK_TICKERS

    # Yahoo uses '-' instead of '.' for share classes (BRK.B -> BRK-B)
    return sorted({t.replace(".", "-") for t in tickers if t and t != "nan"})


# ----------------------------------------------------------------------------
# Per-ticker data fetch
# ----------------------------------------------------------------------------
def fetch_one(ticker: str) -> dict | None:
    """Fetch quote, analyst targets, and fundamentals for one ticker."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None
    if not info or not isinstance(info, dict):
        return None

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    low_t = info.get("targetLowPrice")
    mean_t = info.get("targetMeanPrice")
    high_t = info.get("targetHighPrice")

    # Analyst targets + price are mandatory for the screen
    if not all(isinstance(x, (int, float)) and x > 0 for x in (price, low_t, mean_t, high_t)):
        return None

    revenue = info.get("totalRevenue")
    net_income = info.get("netIncomeToCommon")
    net_margin = None
    if revenue and net_income is not None and revenue != 0:
        net_margin = (net_income / revenue) * 100.0

    peg = info.get("trailingPegRatio") or info.get("pegRatio")
    rev_growth = info.get("revenueGrowth")  # decimal, e.g. 0.23 = 23% YoY

    return {
        "Ticker": ticker,
        "Company": info.get("shortName") or info.get("longName") or ticker,
        "Sector": info.get("sector") or "—",
        "Market Cap": info.get("marketCap"),
        "Price": round(float(price), 2),
        "Low Target": round(float(low_t), 2),
        "Avg Target": round(float(mean_t), 2),
        "High Target": round(float(high_t), 2),
        "Price vs Low %": round((price / low_t - 1) * 100, 2),
        "Avg Upside %": round((mean_t / price - 1) * 100, 2),
        "High Upside %": round((high_t / price - 1) * 100, 2),
        "Trailing P/E": round(info["trailingPE"], 2) if isinstance(info.get("trailingPE"), (int, float)) else None,
        "PEG": round(peg, 2) if isinstance(peg, (int, float)) else None,
        "Revenue (TTM)": revenue,
        "Net Income (TTM)": net_income,
        "GAAP Net Margin %": round(net_margin, 2) if net_margin is not None else None,
        "Rev Growth YoY %": round(rev_growth * 100, 2) if isinstance(rev_growth, (int, float)) else None,
        "Avg Vol (3M)": info.get("averageVolume"),
        "# Analysts": info.get("numberOfAnalystOpinions"),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_universe_data(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Fetch all tickers concurrently (cached 30 min)."""
    rows = []
    progress = st.progress(0.0, text="Fetching analyst targets & fundamentals…")
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                rows.append(row)
            done += 1
            if done % 10 == 0 or done == len(tickers):
                progress.progress(done / len(tickers), text=f"Fetched {done}/{len(tickers)} tickers…")
    progress.empty()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Sidebar — interactive screening parameters
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Screening Criteria")

    low_margin = st.slider(
        "Low Target Margin (%)",
        min_value=0.0, max_value=25.0, value=5.0, step=0.5,
        help="Max % the current price may sit ABOVE the analyst LOW target. "
             "Price below the low target always passes.",
    )
    avg_upside_min = st.slider(
        "Average Upside Minimum (%)",
        min_value=0.0, max_value=100.0, value=15.0, step=1.0,
        help="Consensus (mean) target must be at least this % above current price.",
    )
    high_upside_min = st.slider(
        "High Upside Minimum (%)",
        min_value=0.0, max_value=200.0, value=30.0, step=1.0,
        help="High target must be at least this % above current price.",
    )

    st.divider()
    st.header("🧹 Optional Quality Filters")
    min_analysts = st.number_input("Min # of analyst opinions", 0, 60, 5)
    min_mcap_b = st.number_input("Min market cap ($B)", 0.0, 3000.0, 1.0, step=0.5)

    st.divider()
    run = st.button("🚀 Run Screener", type="primary", use_container_width=True)
    st.caption("First run fetches ~600 tickers and takes 2–5 minutes. "
               "Results are cached for 30 minutes; re-runs with new sliders are instant.")

# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
if run or "screener_ran" in st.session_state:
    st.session_state["screener_ran"] = True

    universe = get_universe()
    st.write(f"**Universe:** {len(universe)} tickers (S&P 500 ∪ NASDAQ 100)")

    t0 = time.time()
    df = fetch_universe_data(tuple(universe))
    if df.empty:
        st.error("No data returned — Yahoo may be rate-limiting. Wait a few minutes and retry.")
        st.stop()
    st.caption(f"Data ready for {len(df)} tickers with full analyst coverage "
               f"({time.time() - t0:.0f}s, cached).")

    # ---- Apply the asymmetric screen -------------------------------------
    mask = (
        (df["Price"] <= df["Low Target"] * (1 + low_margin / 100.0))
        & (df["Avg Target"] >= df["Price"] * (1 + avg_upside_min / 100.0))
        & (df["High Target"] >= df["Price"] * (1 + high_upside_min / 100.0))
        & (df["# Analysts"].fillna(0) >= min_analysts)
        & (df["Market Cap"].fillna(0) >= min_mcap_b * 1e9)
    )
    results = df[mask].copy()

    # Composite score: reward upside skew, penalize distance above the low target
    results["Skew Score"] = (
        results["Avg Upside %"] * 0.5
        + results["High Upside %"] * 0.3
        - results["Price vs Low %"].clip(lower=0) * 2.0
    ).round(1)
    results = results.sort_values("Skew Score", ascending=False).reset_index(drop=True)

    st.subheader(f"📋 Matches: {len(results)} stocks")
    if results.empty:
        st.info("No stocks match the current criteria. Try loosening the sliders.")
        st.stop()

    # ---- Display ----------------------------------------------------------
    display_cols = [
        "Ticker", "Company", "Sector", "Market Cap",
        "Price", "Low Target", "Avg Target", "High Target",
        "Price vs Low %", "Avg Upside %", "High Upside %", "Skew Score",
        "Trailing P/E", "PEG", "Revenue (TTM)", "Net Income (TTM)",
        "GAAP Net Margin %", "Rev Growth YoY %", "Avg Vol (3M)", "# Analysts",
    ]
    st.dataframe(
        results[display_cols],
        use_container_width=True,
        height=600,
        column_config={
            "Market Cap": st.column_config.NumberColumn(format="compact"),
            "Revenue (TTM)": st.column_config.NumberColumn(format="compact"),
            "Net Income (TTM)": st.column_config.NumberColumn(format="compact"),
            "Avg Vol (3M)": st.column_config.NumberColumn(format="compact"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Low Target": st.column_config.NumberColumn(format="$%.2f"),
            "Avg Target": st.column_config.NumberColumn(format="$%.2f"),
            "High Target": st.column_config.NumberColumn(format="$%.2f"),
        },
        hide_index=True,
    )
    st.caption("💡 Click any column header to sort (e.g., lowest PEG, highest GAAP margin).")

    # ---- Export -----------------------------------------------------------
    csv_buf = io.StringIO()
    results[display_cols].to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Export to CSV",
        data=csv_buf.getvalue(),
        file_name=f"asymmetric_screener_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    st.divider()
    st.caption(
        "⚠️ Analyst targets are opinions, not guarantees — a price sitting at the low target "
        "can still fall through it. This tool is informational and not investment advice."
    )
else:
    st.info("Set your criteria in the sidebar and click **Run Screener**.")
