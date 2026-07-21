"""
Asymmetric Analyst Target Screener
==================================
A single-file Streamlit app that screens the S&P 500 + NASDAQ 100 universe for
stocks with asymmetric analyst price-target profiles:

  1. LOW TARGET FLOOR   -> current price <= Analyst LOW target * (1 + low_margin)
                           (i.e., price is at/near/below the most pessimistic target)
  2. AVERAGE UPSIDE     -> Analyst MEAN target >= current price * (1 + avg_upside_min)
  3. HIGH UPSIDE        -> Analyst HIGH target >= current price * (1 + high_upside_min)

Data source: yfinance (free, no API key required). Universe is selectable:
built-in list or live index constituents pulled from the tracking ETF's
official holdings file (IVV, QQQ, DIA, IJH, IWM). Matches are enriched with
technical entry filters (RSI-14, price vs 200-SMA, % off 52w low), analyst
action freshness, and P/E relative to sector median.

Run:
    pip install streamlit yfinance pandas lxml requests openpyxl
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
    # ---- S&P 500 (user-provided) ----
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
    # ---- Berkshire (was missing) ----
    "BRK-B",
    # ---- NASDAQ 100 additions not in S&P 500 ----
    "ARM","APP","ASML","AZN","BIIB","CCEP","DDOG","MDB","MELI","MSTR","PDD","TEAM","ZS",
]

# Normalize any ticker with '.' to Yahoo's '-' convention (e.g., BRK.B -> BRK-B).
FALLBACK_TICKERS = sorted({t.replace(".", "-") for t in FALLBACK_TICKERS if t})


# ----------------------------------------------------------------------------
# Index constituents via ETF holdings files (official issuer downloads)
# ----------------------------------------------------------------------------
import re

import requests

_DL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

# Issuer holdings-download URLs. These are the documented download endpoints as
# of mid-2026; issuers occasionally reshuffle them, so every fetch has a
# fallback to the built-in list and surfaces the error in the UI.
INDEX_SOURCES: dict[str, tuple[str, str]] = {
    "S&P 500 — iShares IVV": (
        "csv",
        "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/"
        "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund",
    ),
    "NASDAQ-100 — Invesco QQQ": (
        "csv",
        "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0"
        "?audienceType=Investor&action=download&ticker=QQQ",
    ),
    "Dow 30 — SPDR DIA": (
        "xlsx",
        "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/"
        "etfs/us/holdings-daily-us-en-dia.xlsx",
    ),
    "S&P MidCap 400 — iShares IJH": (
        "csv",
        "https://www.ishares.com/us/products/239763/ishares-core-sp-mid-cap-etf/"
        "1467271812596.ajax?fileType=csv&fileName=IJH_holdings&dataType=fund",
    ),
    "Russell 2000 — iShares IWM": (
        "csv",
        "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
        "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund",
    ),
}

_TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")  # equity-like; excludes cash/futures codes


def _clean_ticker_series(series: pd.Series) -> list[str]:
    """Normalize a raw holdings ticker column to Yahoo-compatible equity tickers."""
    out = set()
    for raw in series.dropna().astype(str):
        t = raw.strip().upper().replace(".", "-").replace("/", "-")
        # Drop cash placeholders, futures, and junk (XTSLA, USD, ESH6, '--', numbers)
        if _TICKER_RE.match(t) and t not in {"USD", "CASH"}:
            out.add(t)
    return sorted(out)


def _parse_holdings_csv(text: str) -> list[str]:
    """Issuer CSVs have preamble rows before the real header — try each candidate."""
    lines = text.splitlines()
    # Every line that plausibly looks like a header row (has ticker/symbol AND
    # multiple commas — filters out single-cell preamble entries).
    candidates = [
        i for i, line in enumerate(lines[:40])
        if ("ticker" in line.lower() or "symbol" in line.lower())
        and line.count(",") >= 3
    ]
    if not candidates:
        raise ValueError("No plausible header row found in holdings CSV")

    last_err = None
    for header_idx in candidates:
        try:
            body = pd.read_csv(
                io.StringIO("\n".join(lines[header_idx:])),
                on_bad_lines="skip",
                dtype=str,
            )
        except Exception as e:
            last_err = e
            continue
        # Find a ticker column — skip preamble columns like "Fund Ticker"
        tick_col = None
        for c in body.columns:
            cl = str(c).strip().lower()
            if cl.startswith("fund"):
                continue
            if ("ticker" in cl) or (cl == "symbol") or cl.endswith(" symbol") or cl == "local symbol":
                tick_col = c
                break
        # Fallback: no ticker/symbol-named column, but one column's values LOOK
        # like tickers (short uppercase strings, mostly matching)
        if tick_col is None:
            for c in body.columns:
                vals = body[c].dropna().astype(str).str.strip().str.upper()
                if len(vals) == 0:
                    continue
                match_rate = vals.apply(lambda x: bool(_TICKER_RE.match(x.replace(".", "-")))).mean()
                if match_rate > 0.7 and len(vals) >= 20:
                    tick_col = c
                    break
        if tick_col is None:
            continue
        # If an asset-class column exists, keep only equities
        for c in body.columns:
            if "asset class" in str(c).lower() or "security type" in str(c).lower():
                body = body[body[c].astype(str).str.contains("equity|stock", case=False, na=False, regex=True)]
                break
        tickers = _clean_ticker_series(body[tick_col])
        if len(tickers) >= 20:
            return tickers

    raise ValueError(
        f"Header candidates tried but none yielded a valid ticker column"
        + (f" (last parse error: {last_err})" if last_err else "")
    )


def _parse_holdings_xlsx(content: bytes) -> list[str]:
    raw = pd.read_excel(io.BytesIO(content), header=None)
    header_idx = None
    for i in range(min(15, len(raw))):
        if any(str(v).strip().lower() == "ticker" for v in raw.iloc[i].tolist()):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("No 'Ticker' header row found in holdings XLSX")
    body = pd.read_excel(io.BytesIO(content), header=header_idx)
    tick_col = next(c for c in body.columns if str(c).strip().lower() == "ticker")
    return _clean_ticker_series(body[tick_col])


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_index_constituents(index_name: str) -> tuple[list[str], str | None]:
    """Pull an index's tickers from its tracking ETF's holdings file (cached 24h).

    Returns (tickers, error). On any failure returns ([], error_message) so the
    caller can fall back to the built-in list and show why.
    """
    kind, url = INDEX_SOURCES[index_name]
    try:
        resp = requests.get(url, headers=_DL_HEADERS, timeout=30)
        resp.raise_for_status()
        tickers = (
            _parse_holdings_csv(resp.text) if kind == "csv"
            else _parse_holdings_xlsx(resp.content)
        )
        if len(tickers) < 20:  # even Dow 30 should yield ~30
            raise ValueError(f"Only {len(tickers)} plausible tickers parsed — format likely changed")
        return tickers, None
    except Exception as e:
        return [], f"{index_name}: {type(e).__name__}: {e}"


# ----------------------------------------------------------------------------
# Per-ticker data fetch
# ----------------------------------------------------------------------------
import random
import requests

_YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_YF_HEADERS)
    return s


def fetch_one(ticker: str, session: requests.Session, max_retries: int = 3) -> dict | None:
    """Fetch quote, analyst targets, and fundamentals for one ticker.

    Uses a shared session with browser headers and retries with exponential
    backoff on transient failures (Yahoo rate limits datacenter IPs).
    """
    info = None
    for attempt in range(max_retries):
        try:
            # Small jitter so 4 workers don't hit Yahoo in lockstep
            time.sleep(random.uniform(0.15, 0.45))
            info = yf.Ticker(ticker, session=session).info
            if info and isinstance(info, dict) and info.get("symbol"):
                break
        except Exception:
            pass
        # Exponential backoff: 1s, 2s, 4s
        time.sleep(2 ** attempt)
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
        "Analyst Rating": str(info.get("recommendationKey", "")).replace("_", " ").title() or None,
        "Rec Mean": round(info["recommendationMean"], 2) if isinstance(info.get("recommendationMean"), (int, float)) else None,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_universe_data(tickers: tuple[str, ...], max_workers: int = 4) -> pd.DataFrame:
    """Fetch all tickers with limited concurrency (cached 30 min).

    Uses only 4 workers by default — Yahoo aggressively rate-limits
    datacenter IPs (Streamlit Cloud), so slow-and-steady beats parallel.
    Expect ~3–5 min for the S&P 500 + NASDAQ 100 universe.
    """
    rows = []
    attempted = 0
    session = _make_session()
    progress = st.progress(0.0, text="Fetching analyst targets & fundamentals…")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_one, t, session): t for t in tickers}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                rows.append(row)
            attempted += 1
            if attempted % 10 == 0 or attempted == len(tickers):
                progress.progress(
                    attempted / len(tickers),
                    text=f"Fetched {attempted}/{len(tickers)} — {len(rows)} with full data",
                )
    progress.empty()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Technical indicators (batched price history — cheap on request budget)
# ----------------------------------------------------------------------------
def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_technicals(tickers: tuple[str, ...]) -> pd.DataFrame:
    """RSI(14), price vs 200-SMA, and % off 52w low for a set of tickers.

    Uses yf.download in chunks — one request covers ~100 tickers, so this adds
    only a handful of requests even for large result sets.
    """
    out = []
    chunk_size = 100
    for i in range(0, len(tickers), chunk_size):
        chunk = list(tickers[i : i + chunk_size])
        try:
            data = yf.download(
                chunk, period="1y", interval="1d", group_by="ticker",
                auto_adjust=True, progress=False, threads=True,
            )
        except Exception:
            continue
        for t in chunk:
            try:
                close = (data[t]["Close"] if len(chunk) > 1 else data["Close"]).dropna()
            except Exception:
                continue
            if len(close) < 60:
                continue  # not enough history for meaningful indicators
            price = float(close.iloc[-1])
            sma200 = float(close.rolling(200, min_periods=60).mean().iloc[-1])
            lo, hi = float(close.min()), float(close.max())
            off_low = (price - lo) / (hi - lo) * 100 if hi > lo else None
            rsi = float(_rsi_wilder(close).iloc[-1])
            out.append({
                "Ticker": t,
                "RSI(14)": round(rsi, 1),
                "Px vs 200SMA %": round((price / sma200 - 1) * 100, 1) if sma200 else None,
                "% Off 52w Low": round(off_low, 1) if off_low is not None else None,
            })
    return pd.DataFrame(out, columns=["Ticker", "RSI(14)", "Px vs 200SMA %", "% Off 52w Low"])


# ----------------------------------------------------------------------------
# Analyst action freshness (per-ticker — fetched only for screen MATCHES)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_analyst_actions(tickers: tuple[str, ...]) -> dict[str, list[tuple[str, str]]]:
    """Per ticker: list of (date_iso, action) from Yahoo's upgrades/downgrades feed.

    Actions are Yahoo codes: 'up', 'down', 'init', 'main', 'reit'. Note: this
    feed has dates and firms but NOT the target prices — Yahoo doesn't expose
    per-analyst targets, so targets can't be recomputed from a recent window.
    """
    session = _make_session()
    out: dict[str, list[tuple[str, str]]] = {}
    for t in tickers:
        try:
            time.sleep(random.uniform(0.1, 0.3))
            ud = yf.Ticker(t, session=session).upgrades_downgrades
            if ud is None or len(ud) == 0:
                out[t] = []
                continue
            acts = []
            for idx, row in ud.iterrows():
                try:
                    acts.append((pd.Timestamp(idx).strftime("%Y-%m-%d"), str(row.get("Action", "")).lower()))
                except Exception:
                    continue
            out[t] = acts
        except Exception:
            out[t] = []
    return out


def summarize_actions(acts: list[tuple[str, str]], window_days: int) -> tuple[int | None, int, int]:
    """(days_since_last_action, count_in_window, net_up_minus_down_in_window)."""
    if not acts:
        return None, 0, 0
    today = pd.Timestamp.now().normalize()
    dates = [pd.Timestamp(d) for d, _ in acts]
    days_since = int((today - max(dates)).days)
    cutoff = today - pd.Timedelta(days=window_days)
    recent = [(d, a) for (d, a), ts in zip(acts, dates) if ts >= cutoff]
    n = len(recent)
    net = sum(1 for _, a in recent if a == "up") - sum(1 for _, a in recent if a == "down")
    return days_since, n, net


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
    st.header("🌐 Universe")
    universe_choice = st.selectbox(
        "Index",
        ["Built-in list (S&P 500 + NDX)"] + list(INDEX_SOURCES.keys()),
        index=0,
        help="Built-in list: fast, no network call, may lag index rebalances. "
             "ETF options pull live constituents from the issuer's holdings file "
             "(cached 24h) and fall back to the built-in list on failure.",
    )
    if "Russell 2000" in universe_choice:
        st.warning(
            "Russell 2000 ≈ 2,000 tickers — at throttle-safe speed the first "
            "fetch can take 20+ minutes and is more likely to trip Yahoo's "
            "rate limits on cloud hosting."
        )

    st.divider()
    st.header("📅 Analyst Freshness (matches only)")
    action_window = st.number_input(
        "Action window (days)", 5, 365, 30,
        help="Look-back window for counting analyst actions (upgrades, downgrades, "
             "initiations, reiterations) on stocks that pass the screen.",
    )
    require_recent = st.checkbox(
        "Require ≥1 analyst action within window", value=False,
        help="Filters out matches whose consensus may be stale.",
    )

    st.divider()
    st.header("📈 Technical Entry Filters")
    apply_tech = st.checkbox("Apply technical filters", value=True)
    rsi_range = st.slider(
        "RSI(14) range", 0, 100, (30, 55),
        help="30–55 targets 'pulled back but not in free-fall'.",
    )
    min_sma_pct = st.slider(
        "Min price vs 200-day SMA (%)", -50, 50, -10,
        help="-10 allows price up to 10% below the 200-SMA; 0 requires price above it.",
    )
    min_off_low = st.slider(
        "Min % off 52-week low", 0, 100, 12,
        help="Position within the 52w range: 0 = at the low, 100 = at the high. "
             "A floor here filters stocks that haven't stabilized off the bottom.",
    )
    hide_tech_fails = st.checkbox(
        "Hide stocks failing technical filters", value=False,
        help="Off: failing rows stay visible, marked ❌ in the Tech Pass column, "
             "so you can judge what the filter excludes.",
    )

    st.divider()
    run = st.button("🚀 Run Screener", type="primary", use_container_width=True)
    st.caption("First run fetches the universe and takes 2–5 minutes (longer for "
               "large indices). Results are cached for 30 minutes; re-runs with "
               "new sliders are instant.")

# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
if run or "screener_ran" in st.session_state:
    st.session_state["screener_ran"] = True

    if universe_choice == "Built-in list (S&P 500 + NDX)":
        universe = FALLBACK_TICKERS
        universe_label = "Built-in S&P 500 ∪ NASDAQ 100"
    else:
        universe, uerr = fetch_index_constituents(universe_choice)
        universe_label = universe_choice
        if uerr:
            st.warning(f"Holdings download failed — {uerr}")
            st.error("Falling back to the built-in list.")
            universe = FALLBACK_TICKERS
            universe_label = "Built-in S&P 500 ∪ NASDAQ 100 (fallback)"

    st.write(f"**Universe:** {len(universe)} tickers ({universe_label})")
    if st.button("🔄 Clear Cache & Re-run"):
        fetch_index_constituents.clear()
        fetch_universe_data.clear()
        fetch_technicals.clear()
        fetch_analyst_actions.clear()
        st.rerun()

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

    # ---- P/E vs Sector (median of positive P/Es across the fetched universe) --
    pe_pos = df[df["Trailing P/E"].notna() & (df["Trailing P/E"] > 0)]
    sector_median_pe = pe_pos.groupby("Sector")["Trailing P/E"].median()

    def _pe_vs_sector(row):
        pe, sec = row["Trailing P/E"], row["Sector"]
        med = sector_median_pe.get(sec)
        if pd.notna(pe) and pe > 0 and med and med > 0:
            return round((pe / med - 1) * 100, 1)
        return None

    results["P/E vs Sector %"] = results.apply(_pe_vs_sector, axis=1)

    # ---- Technical indicators (batched, matches only) ----------------------
    with st.spinner("Computing technical indicators…"):
        tech = fetch_technicals(tuple(results["Ticker"]))
    results = results.merge(tech, on="Ticker", how="left")

    tech_pass = (
        results["RSI(14)"].between(rsi_range[0], rsi_range[1])
        & (results["Px vs 200SMA %"] >= min_sma_pct)
        & (results["% Off 52w Low"] >= min_off_low)
    ).fillna(False)
    results["Tech Pass"] = tech_pass.map({True: "✅", False: "❌"})
    if apply_tech and hide_tech_fails:
        results = results[tech_pass.values].reset_index(drop=True)
        if results.empty:
            st.info("All fundamental matches were removed by the technical filters. "
                    "Loosen them or untick 'Hide stocks failing technical filters'.")
            st.stop()

    # ---- Analyst action freshness (per-ticker, matches only) ---------------
    with st.spinner("Fetching analyst action history for matches…"):
        actions = fetch_analyst_actions(tuple(results["Ticker"]))
    fresh = results["Ticker"].map(
        lambda t: summarize_actions(actions.get(t, []), int(action_window))
    )
    results["Days Since Action"] = [f[0] for f in fresh]
    results[f"Actions ({int(action_window)}d)"] = [f[1] for f in fresh]
    results[f"Net Revisions ({int(action_window)}d)"] = [f[2] for f in fresh]
    if require_recent:
        keep = results[f"Actions ({int(action_window)}d)"] >= 1
        results = results[keep.values].reset_index(drop=True)
        if results.empty:
            st.info("No matches have an analyst action within the window. "
                    "Widen the window or untick the freshness requirement.")
            st.stop()

    # ---- Rules-based Signal label ------------------------------------------
    # A mechanical summary of the columns above — NOT advice or a prediction.
    # Scoring is intentionally simple and fully visible in the legend expander.
    act_col = f"Actions ({int(action_window)}d)"
    net_col = f"Net Revisions ({int(action_window)}d)"

    def _signal(row) -> tuple[str, str, str]:
        score = 0
        parts: list[str] = []
        # Entry timing
        if row.get("Tech Pass") == "✅":
            score += 2; parts.append("+2 Tech✅")
        rsi, sma = row.get("RSI(14)"), row.get("Px vs 200SMA %")
        if pd.notna(rsi) and rsi < 25:
            score -= 1; parts.append("−1 RSI<25")
        if pd.notna(sma) and sma < -20:
            score -= 2; parts.append("−2 <200SMA")
        # Analyst momentum & freshness
        net, acts, days = row.get(net_col), row.get(act_col), row.get("Days Since Action")
        if pd.notna(net):
            if net > 0:
                score += 1; parts.append(f"+1 Rev+{int(net)}")
            elif net < 0:
                score -= 2; parts.append(f"−2 Rev{int(net)}")
        if pd.notna(acts) and acts >= 1:
            score += 1; parts.append(f"+1 {int(acts)}act")
        if days is None or (pd.notna(days) and days > 90):
            score -= 1; parts.append("−1 stale")
        # Valuation extras
        if pd.notna(row.get("Price vs Low %")) and row["Price vs Low %"] <= 0:
            score += 1; parts.append("+1 ≤Low")
        peg = row.get("PEG")
        if pd.notna(peg) and 0 < peg <= 2:
            score += 1; parts.append("+1 PEG≤2")
        pes = row.get("P/E vs Sector %")
        if pd.notna(pes) and pes < 0:
            score += 1; parts.append("+1 P/E<sect")

        if score >= 5:
            label = "🟢 Strong Buy"
        elif score >= 3:
            label = "🟢 Buy"
        elif score >= 1:
            label = "🟡 Wait"
        else:
            label = "🔴 Avoid"
        short_score = f"{score:+d}"
        breakdown = " · ".join(parts) if parts else "no rules triggered"
        return label, short_score, breakdown

    _sig = results.apply(_signal, axis=1)
    results["Signal"] = [s[0] for s in _sig]
    results["Score"] = [s[1] for s in _sig]
    results["Signal Breakdown"] = [s[2] for s in _sig]

    with st.expander("ℹ️ How the Signal label is computed (rules, not advice)"):
        st.markdown(
            """
Every match already passed your fundamental screen; the Signal only ranks *entry conditions*
using a simple point system (max ≈ 7):

| Points | Rule |
|---|---|
| +2 | Passes all technical filters (Tech Pass ✅) |
| −1 | RSI(14) below 25 — capitulation zone, still being sold |
| −2 | Price more than 20% below 200-day SMA — entrenched downtrend |
| +1 / −2 | Net revisions in window positive / negative |
| +1 | ≥1 analyst action within the window |
| −1 | No analyst action in 90+ days (stale consensus) |
| +1 | Price at or below the analyst LOW target |
| +1 | PEG between 0 and 2 |
| +1 | Trailing P/E below sector median |

**Score ≥5 → 🟢 Strong Buy · ≥3 → 🟢 Buy · ≥1 → 🟡 Wait · else 🔴 Avoid**

These are mechanical labels from the rules above — heuristics, not predictions, and not
investment advice. Compare them against the *Analyst Rating* column (the Street's actual
consensus) — disagreements between the two are often the most interesting rows.
            """
        )
    # ---- Display ----------------------------------------------------------
    display_cols = [
        "Ticker", "Company", "Sector", "Signal", "Score", "Analyst Rating", "Market Cap",
        "Price", "Low Target", "Avg Target", "High Target",
        "Price vs Low %", "Avg Upside %", "High Upside %", "Skew Score",
        "Tech Pass", "RSI(14)", "Px vs 200SMA %", "% Off 52w Low",
        "Days Since Action", f"Actions ({int(action_window)}d)",
        f"Net Revisions ({int(action_window)}d)",
        "Trailing P/E", "P/E vs Sector %", "PEG",
        "Revenue (TTM)", "Net Income (TTM)",
        "GAAP Net Margin %", "Rev Growth YoY %", "Avg Vol (3M)", "# Analysts",
    ]
    view = results[display_cols].copy()

    st.caption("💡 Ticker/Company/Sector stay pinned when you scroll right; headers stay "
               "visible when you scroll down. Click any column header to sort.")
    event = st.dataframe(
        view,
        use_container_width=True,
        height=600,
        on_select="rerun",
        selection_mode="single-row",
        key="results_table",
        column_config={
            "Ticker": st.column_config.TextColumn(pinned=True),
            "Company": st.column_config.TextColumn(pinned=True),
            "Sector": st.column_config.TextColumn(pinned=True),
            "Signal": st.column_config.TextColumn(
                pinned=True,
                help=(
                    "Rules-based entry label (mechanical, not advice). Points:\n"
                    "+2 Tech ✅ | −1 RSI<25 | −2 price >20% below 200-SMA\n"
                    "+1 Rev>0 | −2 Rev<0 | +1 ≥1 action | −1 no action 90d+\n"
                    "+1 ≤ Low target | +1 PEG 0–2 | +1 P/E below sector median\n"
                    "≥5 Strong Buy · ≥3 Buy · ≥1 Wait · else Avoid.\n"
                    "See 'Score' for the total, or click a row for the full breakdown."
                ),
            ),
            "Signal Breakdown": st.column_config.TextColumn(
                # Not shown in the table (see display_cols); kept here as
                # defensive config in case it ever gets added back.
                width="large",
            ),
            "Score": st.column_config.TextColumn(
                width="small",
                help="Sum of the rule points for this stock. See the Signal column's "
                     "tooltip for the point system, or open the detail view for the "
                     "full breakdown of which rules fired.",
            ),
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

    # ---- Full-page single-stock detail view --------------------------------
    # Two ways to open it: tick a row's checkbox in the table, or pick from the
    # dropdown below (works on every Streamlit version / device).
    detail_choice = st.selectbox(
        "🔍 Open full-page detail for a stock",
        ["— select a ticker —"] + view["Ticker"].tolist(),
        index=0,
    )

    sel = None
    sel_ticker = None
    if event.selection.rows:
        sel_ticker = view.iloc[event.selection.rows[0]]["Ticker"]
    elif detail_choice != "— select a ticker —":
        sel_ticker = detail_choice
    if sel_ticker is not None:
        sel = results[results["Ticker"] == sel_ticker].iloc[0]

    # ---- Full-page single-stock detail view --------------------------------
    def _fmt_value(col: str, v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        if col in ("Market Cap", "Revenue (TTM)", "Net Income (TTM)", "Avg Vol (3M)"):
            n = float(v)
            for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
                if abs(n) >= div:
                    return f"${n / div:.2f}{suf}" if col != "Avg Vol (3M)" else f"{n / div:.2f}{suf}"
            return f"{n:,.0f}"
        if col in ("Price", "Low Target", "Avg Target", "High Target"):
            return f"${v:,.2f}"
        if col.endswith("%") or "%" in col:
            return f"{v}%"
        return str(v)

    if sel is not None:
        st.divider()
        st.header(f"🔍 {sel['Ticker']} — {sel['Company']}")
        st.caption(f"{sel['Sector']} · all screener columns for this stock in one view")

        sections = {
            "🧭 Verdict": ["Signal", "Score", "Signal Breakdown", "Analyst Rating", "Rec Mean", "Skew Score"],
            "🎯 Price & Analyst Targets": [
                "Price", "Low Target", "Avg Target", "High Target",
                "Price vs Low %", "Avg Upside %", "High Upside %", "Skew Score",
            ],
            "📈 Technicals": ["Tech Pass", "RSI(14)", "Px vs 200SMA %", "% Off 52w Low"],
            "📅 Analyst Activity": [
                "Days Since Action", f"Actions ({int(action_window)}d)",
                f"Net Revisions ({int(action_window)}d)", "# Analysts",
            ],
            "🏦 Fundamentals": [
                "Market Cap", "Trailing P/E", "P/E vs Sector %", "PEG",
                "Revenue (TTM)", "Net Income (TTM)", "GAAP Net Margin %",
                "Rev Growth YoY %", "Avg Vol (3M)",
            ],
        }
        for title, cols in sections.items():
            st.subheader(title)
            grid = st.columns(4)
            for i, col in enumerate(cols):
                grid[i % 4].metric(col, _fmt_value(col, sel.get(col)))
        st.caption("Close: untick the row's checkbox, or reset the dropdown to '— select a ticker —'.")

    # ---- Export (respects active column filters) ---------------------------
    csv_buf = io.StringIO()
    view.to_csv(csv_buf, index=False)
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
