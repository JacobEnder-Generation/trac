#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from PyPDF2 import PdfMerger

FACTOR=1.284; TOL=0.064; MAX_N=15
NAVY="#001F5B"; CORRIDOR_LINE="#5FA0FF"; CORRIDOR_FILL="#A9C8FF"; PRICE_COLOR="#000000"
GREEN="#0E7C0E"; RED="#B00020"

# Floor tier colors
# GOLD BANDS HIDDEN: Using same colors as regular corridors
# To re-enable: Change GOLD_FLOOR_FILL back to "#FFD700" and GOLD_FLOOR_LINE to "#DAA520"
GOLD_FLOOR_FILL = "#A9C8FF"      # Hidden: using regular corridor fill
GOLD_FLOOR_LINE = "#5FA0FF"      # Hidden: using regular corridor line
SILVER_FLOOR_FILL = "#C0C0C0"    # Silver fill
SILVER_FLOOR_LINE = "#808080"    # Silver line (darker)
BRONZE_FLOOR_FILL = "#CD7F32"    # Bronze fill
BRONZE_FLOOR_LINE = "#8B4513"    # Bronze line (darker)


@dataclass(frozen=True)
class BacktestParameters:
    band_tolerance: float = TOL
    buy_signal_days: int = 2
    ceiling_sell_days: int = 5
    stop_sell_days: int = 6
    backtest_years: Optional[int] = None

    def normalized(self) -> "BacktestParameters":
        tol = float(min(max(self.band_tolerance, 0.005), 0.2))
        buy_days = max(1, int(self.buy_signal_days))
        ceil_days = max(1, int(self.ceiling_sell_days))
        stop_days = max(1, int(self.stop_sell_days))
        yrs = int(self.backtest_years) if self.backtest_years else None
        return BacktestParameters(
            band_tolerance=tol,
            buy_signal_days=buy_days,
            ceiling_sell_days=ceil_days,
            stop_sell_days=stop_days,
            backtest_years=yrs,
        )


@dataclass
class BacktestResult:
    ticker: str
    rule: str
    dates: pd.DatetimeIndex
    close: pd.Series
    bvps: pd.Series
    buys: List[tuple]
    cycles: pd.DataFrame
    bench: Dict[str, Any]
    strat_rf: Dict[str, Any]
    fmv: Optional[pd.Series] = None
    warnings: List[str] = field(default_factory=list)
    params: BacktestParameters = field(default_factory=BacktestParameters)

def clean_number(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float,np.number)): return float(x)
    s=str(x).strip().replace("\u2212","-").replace(",","").replace(" ","")
    if re.match(r"^\(.*\)$", s): s="-"+s[1:-1]
    try: return float(s)
    except: return np.nan

def parse_date_maybe(x):
    if pd.isna(x): return pd.NaT
    if isinstance(x,(int,float,np.integer,np.floating)):
        val=float(x)
        d1=pd.to_datetime(val, unit="D", origin="1899-12-30", errors="coerce")
        d2=pd.to_datetime(val, unit="D", origin="1904-01-01", errors="coerce")
        ref=pd.Timestamp("2000-01-01")
        def ok(d): return pd.notna(d) and pd.Timestamp("1980-01-01")<=d<=pd.Timestamp("2100-12-31")
        if ok(d1) and not ok(d2): return d1
        if ok(d2) and not ok(d1): return d2
        if ok(d1) and ok(d2): return d1 if abs((d1-ref).days)<abs((d2-ref).days) else d2
        return d1 if pd.notna(d1) else d2
    try: return pd.to_datetime(x, errors="raise")
    except: return pd.NaT

def to_quarter_end(d):
    if pd.isna(d): return pd.NaT
    d=pd.Timestamp(d); q=((d.month-1)//3+1)*3
    nm=pd.Timestamp(year=d.year + (1 if q==12 else 0), month=(1 if q==12 else q+1), day=1)
    return nm - pd.Timedelta(days=1)

def geometric_ticks(vmin, vmax, count=7):
    if vmin<=0: vmin=1e-6
    r=np.linspace(0,1,count); return list(vmin*(vmax/vmin)**r)

def detect_groups(df):
    cols=list(df.columns); groups=[]; i=0
    while i<len(cols):
        remain=len(cols)-i
        if remain>=10: groups.append((i,10)); i+=10
        elif remain>=9: groups.append((i,9)); i+=9
        else: break
    return groups

def load_group(df, start, count):
    cols=list(df.columns)[start:start+count]
    flat_col=cols[9] if count>=10 else None
    rec=pd.DataFrame({
        "Date": df[cols[0]].apply(parse_date_maybe),
        "Close": df[cols[4]].apply(clean_number),
        "BV_Date": df[cols[5]].apply(parse_date_maybe),
        "BVPS": df[cols[6]].apply(clean_number),
        "EstBV_Date": df[cols[7]].apply(parse_date_maybe),
        "EstBVPS": df[cols[8]].apply(clean_number),
    })
    rec["FlatBVPS"]=df[flat_col].apply(clean_number) if flat_col else np.nan
    rec=rec.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return rec

def build_bvps(rec):
    dates=pd.DatetimeIndex(rec["Date"])
    q_actual=rec.dropna(subset=["BV_Date","BVPS"])[["BV_Date","BVPS"]].copy()
    q_est   =rec.dropna(subset=["EstBV_Date","EstBVPS"])[["EstBV_Date","EstBVPS"]].copy()
    q_actual["Q"]=q_actual["BV_Date"].apply(to_quarter_end)
    q_est["Q"]=q_est["EstBV_Date"].apply(to_quarter_end)
    if len(q_actual): q_actual=q_actual.sort_values("BV_Date").groupby("Q",as_index=False).tail(1)
    if len(q_est):    q_est   =q_est.sort_values("EstBV_Date").groupby("Q",as_index=False).tail(1)
    q_all=pd.merge(q_actual[["Q","BVPS"]], q_est[["Q","EstBVPS"]], on="Q", how="outer")
    q_all["BVPS_final"]=np.where(q_all["EstBVPS"].notna(), q_all["EstBVPS"], q_all["BVPS"])
    q_all=q_all.dropna(subset=["Q","BVPS_final"]).sort_values("Q")
    if len(q_all)>=3:
        base=pd.Series(q_all["BVPS_final"].values, index=pd.DatetimeIndex(q_all["Q"])).sort_index()
        union_idx=dates.union(base.index).unique().sort_values()
        bvps=base.reindex(union_idx).interpolate(method="time").ffill().bfill()
        bvps=bvps.reindex(dates).ffill().bfill().astype(float)
    else:
        if rec["FlatBVPS"].notna().any(): const=float(rec["FlatBVPS"].dropna().iloc[0])
        else: const = rec["EstBVPS"].dropna().iloc[-1] if rec["EstBVPS"].notna().any() else rec["BVPS"].dropna().iloc[-1]
        bvps=pd.Series(const, index=dates).astype(float)
    return bvps

def calculate_fmv(dates, close, bvps, cycles_df):
    """
    Calculate Fair Market Value (FMV) as LOWESS smoothed gold band ceiling.
    This represents a smoothed fair value estimate based on the most frequently hit ceiling band.
    
    Returns:
        pd.Series: FMV values indexed by dates, or None if calculation fails
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        
        # Get dynamic gold band classification
        gold_band_series = classify_floor_tiers_dynamic(dates, cycles_df, bvps)
        
        # Calculate the actual price of the gold band for each date
        gold_ceiling_prices = []
        for i, d in enumerate(dates):
            gold_band = gold_band_series.iloc[i]
            if pd.notna(gold_band):
                bvps_value = float(bvps.loc[d])
                gold_price = bvps_value * (FACTOR ** gold_band)
                gold_ceiling_prices.append(gold_price)
            else:
                gold_ceiling_prices.append(np.nan)
        
        # Create series and interpolate NaN values
        gold_ceiling_series = pd.Series(gold_ceiling_prices, index=dates)
        gold_ceiling_interpolated = gold_ceiling_series.interpolate(method='linear')
        
        # Apply LOWESS smoothing if we have valid data
        if gold_ceiling_interpolated.notna().any():
            valid_mask = gold_ceiling_interpolated.notna()
            y_values = gold_ceiling_interpolated[valid_mask].values
            x_values = np.arange(len(dates))[valid_mask]
            
            # Apply LOWESS (frac=0.2 means use 20% of data for each local regression)
            smoothed = lowess(y_values, x_values, frac=0.20, return_sorted=False)
            
            # Create full series with smoothed values
            fmv_series = pd.Series(index=dates, dtype=float)
            fmv_series.iloc[valid_mask] = smoothed
            
            return fmv_series
        else:
            return None
    except Exception as e:
        print(f"    Warning: FMV calculation failed: {e}")
        return None

def detect_bob_with_prior_inband(dates, close, bvps, params: BacktestParameters, fmv=None):
    """
    Detect buy-on-breakout signals with prior in-band requirement.
    
    NEW: If fmv is provided, only trigger buys when price < FMV (fair value filter)
    """
    tol = params.band_tolerance
    required_days = params.buy_signal_days
    ratio=(close/bvps).astype(float)
    k=np.floor(np.log(ratio)/np.log(FACTOR)); k=pd.Series(k, index=dates).ffill().bfill()
    inband=pd.Series(False, index=dates); m_at=pd.Series(np.nan, index=dates)
    for d in dates:
        m=k.loc[d]; center=float(bvps.loc[d]*(FACTOR**m)); lo=center*(1-tol); hi=center*(1+tol)
        if lo<=close.loc[d]<=hi: inband.loc[d]=True; m_at.loc[d]=m
    idx=list(dates); buys=[]; i=0
    while i<len(idx):
        d=idx[i]
        if inband.loc[d]:
            base_band=m_at.loc[d]; j=i+1; streak=0
            while j<len(idx):
                dj=idx[j]; upper=float(bvps.loc[dj]*(FACTOR**base_band)*(1+tol))
                if close.loc[dj]>=upper:
                    streak+=1
                    if streak>=required_days: 
                        # NEW: Check FMV filter - only buy if price < FMV
                        if fmv is not None and pd.notna(fmv.loc[dj]):
                            if close.loc[dj] < fmv.loc[dj]:
                                buys.append((dj, base_band, d))
                            # else: Skip this buy signal (price >= FMV, overvalued)
                        else:
                            # No FMV filter, use original logic
                            buys.append((dj, base_band, d))
                        break
                else: break
                j+=1
        i+=1
    return sorted(buys, key=lambda x:x[0])

def classify_floor_tiers_dynamic(dates, cycles_df, bvps):
    """
    Calculate which band should be gold using 2-year periods starting from first date.
    Each period's gold band is based on ceiling sells within that same period.
    Uses the actual CeilingBand stored during the exit logic.
    
    Returns:
        pd.Series indexed by date, values are the gold band number (or None if no gold)
    """
    if cycles_df.empty:
        return pd.Series(index=dates, data=None)
    
    # Filter to ceiling exits only
    ceiling_exits = cycles_df[cycles_df['ExitType'] == 'ceiling-5down'].copy()
    
    if ceiling_exits.empty:
        return pd.Series(index=dates, data=None)
    
    ceiling_exits['SellDate'] = pd.to_datetime(ceiling_exits['SellDate'])
    
    # Use the CeilingBand column that was stored during exit logic
    # Filter out rows where CeilingBand is None/NaN
    ceiling_exits = ceiling_exits[ceiling_exits['CeilingBand'].notna()].copy()
    
    if ceiling_exits.empty:
        return pd.Series(index=dates, data=None)
    
    # Create 2-year periods starting from first date
    first_date = dates[0]
    last_date = dates[-1]
    
    period_starts = []
    current_start = first_date
    while current_start <= last_date:
        period_starts.append(current_start)
        current_start = current_start + pd.Timedelta(days=365.25 * 2)  # 2 years
    
    # For each date, determine which period it belongs to and calculate gold band
    gold_bands = []
    for current_date in dates:
        # Find which period this date belongs to
        period_start = None
        period_end = None
        for i, ps in enumerate(period_starts):
            pe = ps + pd.Timedelta(days=365.25 * 2)  # 2 years
            if ps <= current_date < pe:
                period_start = ps
                period_end = pe
                break
        
        if period_start is None:
            gold_bands.append(None)
            continue
        
        # Get ceiling sells within this period
        period_ceilings = ceiling_exits[
            (ceiling_exits['SellDate'] >= period_start) & 
            (ceiling_exits['SellDate'] < period_end)
        ]
        
        if period_ceilings.empty:
            gold_bands.append(None)
        else:
            # Count ceiling sells per band using the stored CeilingBand column
            band_counts = period_ceilings['CeilingBand'].value_counts()
            gold_band = band_counts.index[0]
            gold_bands.append(gold_band)
    
    return pd.Series(gold_bands, index=dates)

def sell_target_date(dates, close, bvps, buy_date, base_band, params: BacktestParameters):
    idx=list(dates); i=idx.index(pd.Timestamp(buy_date))
    tol = params.band_tolerance
    for j in range(i, len(idx)):
        d=idx[j]; center_next=float(bvps.loc[d]*(FACTOR**(base_band+1))); lower_next=center_next*(1-tol)
        if close.loc[d]>=lower_next: return d
    return None

def sell_stop_date(dates, close, bvps, buy_date, base_band, params: BacktestParameters):
    """
    Stop loss exit with band-check:
      - Count N consecutive days below entry band lower edge (configurable)
      - On the trigger day, check if price is IN a band (within blue shaded corridor)
      - If IN band: Cancel stop loss (potential re-entry point)
      - If BETWEEN bands (white space): Execute stop loss
    """
    idx = list(dates)
    i = idx.index(pd.Timestamp(buy_date))
    streak = 0
    tol = params.band_tolerance
    stop_days = params.stop_sell_days
    
    for j in range(i + 1, len(idx)):
        d = idx[j]
        bv = float(bvps.loc[d])
        px = float(close.loc[d])
        center_entry = bv * (FACTOR ** base_band)
        lower_entry = center_entry * (1 - tol)
        
        if px < lower_entry:
            streak += 1
            if streak >= stop_days:  # Trigger day - check if in band before stopping out
                # Check if price is IN any band (blue shaded area)
                ratio = px / bv
                k = np.log(ratio) / np.log(FACTOR)
                nearest_band = int(np.floor(k + 0.5))  # Round to nearest band
                
                # Check if price is IN that band (within ±TOL of the band center)
                band_center = bv * (FACTOR ** nearest_band)
                band_lower = band_center * (1 - tol)
                band_upper = band_center * (1 + tol)
                
                if band_lower <= px <= band_upper:
                    # Price IS in a band (blue shaded area) - CANCEL stop loss
                    streak = 0
                    continue
                else:
                    # Price is between bands (white space) - EXECUTE stop loss
                    return d
        else:
            streak = 0
    
    return None

def sell_ceiling_band_date(dates, close, bvps, buy_date, base_band, params: BacktestParameters):
    """
    Dynamic ceiling exit with band-check:
      - After an anchor buy, as soon as price ENTERS any ceiling corridor (>= lower edge),
        we start watching that ceiling.
      - If price later reaches an even HIGHER ceiling corridor, we PROMOTE the active ceiling
        to that higher one.
      - SELL on the 5th consecutive close strictly BELOW the LOWER edge of the
        CURRENT HIGHEST ceiling reached since the buy.
      - NEW: At trigger time (day 5), check if price is IN a band (within the blue shaded corridor).
        If YES (in-band), CANCEL the sell - it's at a potential entry point.
        If NO (between bands, in white space), EXECUTE the sell.
    
    Returns:
        tuple: (sell_date, ceiling_band) or (None, None) if no sell
    """
    idx = list(dates)
    i = idx.index(pd.Timestamp(buy_date))

    # Start by monitoring the next ceiling above the entry band
    current_band = int(base_band) + 1
    streak = 0
    seen_any_ceiling = False
    tol = params.band_tolerance
    ceiling_days = params.ceiling_sell_days

    for j in range(i + 1, len(idx)):
        d = idx[j]
        bv = float(bvps.loc[d])
        px = float(close.loc[d])

        # Promote to the highest ceiling that has been entered (>= lower edge)
        # Find largest n >= current_band such that px >= lower_edge(n)
        n = current_band
        promoted = False
        while True:
            next_n = n + 1
            lower_n = bv * (FACTOR ** n) * (1 - tol)
            lower_next = bv * (FACTOR ** next_n) * (1 - tol) if next_n <= MAX_N else None

            if px >= lower_n:
                seen_any_ceiling = True
            # If we can promote to a higher ceiling that's also entered, do it
            if next_n <= MAX_N and px >= lower_next:
                n = next_n
                promoted = True
                continue
            break
        if promoted or n != current_band:
            current_band = n
            # when ceiling moves up, reset streak because reference threshold moved up
            streak = 0

        # Active lower edge for current highest ceiling
        lower_edge = bv * (FACTOR ** current_band) * (1 - tol)

        if seen_any_ceiling:
            if px < lower_edge:
                streak += 1
                if streak >= ceiling_days:   # Trigger day - check if price is IN any band
                    # NEW LOGIC: Check if price is within ANY band (in the blue shaded area)
                    # Calculate which band number the price is closest to
                    ratio = px / bv
                    k = np.log(ratio) / np.log(FACTOR)
                    nearest_band = int(np.floor(k + 0.5))  # Round to nearest band
                    
                    # Check if price is IN that band (within ±TOL of the band center)
                    band_center = bv * (FACTOR ** nearest_band)
                    band_lower = band_center * (1 - tol)
                    band_upper = band_center * (1 + tol)
                    
                    if band_lower <= px <= band_upper:
                        # Price IS in a band (blue shaded area) - CANCEL sell
                        streak = 0
                        continue
                    else:
                        # Price is between bands (white space) - EXECUTE sell
                        # Return both the sell date AND the ceiling band
                        return (d, current_band)
            else:
                streak = 0

    return (None, None)

def build_cycles(dates, close, bvps, buys, rule, params: BacktestParameters):
    cycles=[]; i=0
    while i<len(buys):
        bdate, bband, _=buys[i]
        # Convert base_band to int to avoid TypeError in range operations
        bband = int(bband)
        ceiling_band = None  # Track ceiling band for ceiling exits
        
        if rule=="rule1":
            sd=sell_target_date(dates, close, bvps, bdate, bband, params); which="target"
        elif rule=="rule1_stop":
            tgt=sell_target_date(dates, close, bvps, bdate, bband, params)
            stp=sell_stop_date(dates, close, bvps, bdate, bband, params)
            cand=[d for d in [tgt,stp] if d is not None]; sd=min(cand) if cand else None
            which="target" if sd==tgt else ("stop" if sd==stp else None)
        else:
            stp=sell_stop_date(dates, close, bvps, bdate, bband, params)
            ceil_result=sell_ceiling_band_date(dates, close, bvps, bdate, bband, params)
            # Unpack tuple: (sell_date, ceiling_band)
            ceil_date, ceiling_band = ceil_result if ceil_result else (None, None)
            
            cand=[d for d in [stp,ceil_date] if d is not None]; sd=min(cand) if cand else None
            which="stop" if sd==stp else ("ceiling-5down" if sd==ceil_date else None)
        
        cycles.append({"BuyDate":bdate,"BuyPx":float(close.loc[bdate]),"BaseBand":float(bband),
                       "SellDate":sd,"SellPx":float(close.loc[sd]) if sd else np.nan,"ExitType":which,
                       "CeilingBand":ceiling_band})
        if sd is None: break
        i+=1
        while i<len(buys) and pd.Timestamp(buys[i][0])<=sd: i+=1
    if not cycles:
        # Return empty DataFrame with proper columns if no cycles detected
        return pd.DataFrame(columns=["BuyDate","BuyPx","BaseBand","SellDate","SellPx","ExitType","CeilingBand","ReturnPct"])
    df=pd.DataFrame(cycles).sort_values("BuyDate").reset_index(drop=True)
    if not df.empty: df["ReturnPct"]=((df["SellPx"]/df["BuyPx"])-1.0)*100.0
    return df

def buyhold_benchmark(dates, close):
    start_date=pd.Timestamp(dates[0]); end_date=pd.Timestamp(dates[-1])
    start=float(close.loc[start_date]); end=float(close.loc[end_date])
    total=(end/start)-1.0
    days=max((end_date-start_date).days, 1)
    ann=( (end/start)**(365.25/days) ) - 1.0
    return {"StartDate": start_date, "EndDate": end_date, "StartPx": start, "EndPx": end, "TotalReturn": total, "Annualized": ann, "Days": days}

def strategy_with_rf(dates, close, cycles_df, rf=0.03, span='cycles'):
    """
    3% RF when not in the stock.
    span='cycles'  -> first completed buy to last sell (comparable to prior strategy metric).
    span='window'  -> whole file window (first -> last price date).
    """
    idx = list(pd.DatetimeIndex(dates))
    if len(idx) < 2: return {"HasData": False}

    cycles = cycles_df.dropna(subset=["SellDate"]).copy()
    if cycles.empty: return {"HasData": False}

    if span == 'cycles':
        start = pd.Timestamp(cycles["BuyDate"].iloc[0])
        end   = pd.Timestamp(cycles["SellDate"].iloc[-1])
        equity = 1.0
        last_dt = start
        for _, r in cycles.iterrows():
            b = pd.Timestamp(r["BuyDate"]); s = pd.Timestamp(r["SellDate"])
            if b > last_dt:
                d_gap = (b - last_dt).days
                equity *= (1.0 + rf) ** (d_gap / 365.25)
            equity *= float(r["SellPx"] / r["BuyPx"])
            last_dt = s
        total = equity - 1.0
        days = max((end - start).days, 1)
        ann = ( (1.0 + total) ** (365.25 / days) ) - 1.0
        return {"HasData": True, "StartDate": start, "EndDate": end,
                "TotalReturn": total, "Annualized": ann, "Days": days,
                "NumCycles": int(len(cycles))}
    else:
        start = idx[0]; end = idx[-1]
        equity = 1.0
        first_buy = pd.Timestamp(cycles["BuyDate"].iloc[0])
        if start < first_buy:
            d = (first_buy - start).days
            equity *= (1.0 + rf) ** (d / 365.25)
        last_dt = first_buy
        for _, r in cycles.iterrows():
            b = pd.Timestamp(r["BuyDate"]); s = pd.Timestamp(r["SellDate"])
            if b > last_dt:
                d_gap = (b - last_dt).days
                equity *= (1.0 + rf) ** (d_gap / 365.25)
            equity *= float(r["SellPx"] / r["BuyPx"])
            last_dt = s
        if end > last_dt:
            d = (end - last_dt).days
            equity *= (1.0 + rf) ** (d / 365.25)
        total = equity - 1.0
        days = max((end - start).days, 1)
        ann = ( (1.0 + total) ** (365.25 / days) ) - 1.0
        return {"HasData": True, "StartDate": start, "EndDate": end,
                "TotalReturn": total, "Annualized": ann, "Days": days,
                "NumCycles": int(len(cycles))}

def equity_curve_with_rf(dates, close, cycles_df, rf=0.03):
    """Full-window equity curves starting at 100 for both:
       - Passive buy&hold
       - BOB strategy with 3% RF when not invested (between completed cycles and any open leg to data end)
    """
    idx = pd.DatetimeIndex(dates)
    px = pd.Series(close.values, index=idx).astype(float)

    # Passive curve
    passive = 100.0 * (px / px.iloc[0])

    # In-stock mask: for each cycle mark buy -> sell, and if sell is NaN, extend to the last date
    in_stock = pd.Series(False, index=idx)
    if cycles_df is not None and not cycles_df.empty:
        for _, r in cycles_df.sort_values("BuyDate").iterrows():
            b = pd.Timestamp(r["BuyDate"])
            s = pd.Timestamp(r["SellDate"]) if pd.notna(r["SellDate"]) else idx[-1]
            # clamp to available index just in case
            b = max(b, idx[0]); s = min(s, idx[-1])
            in_stock.loc[b:s] = True

    # Strategy with 3% RF when not invested
    strat = pd.Series(np.nan, index=idx, dtype=float)
    strat.iloc[0] = 100.0
    for i in range(1, len(idx)):
        d0, d1 = idx[i-1], idx[i]
        days = max((d1 - d0).days, 1)
        if in_stock.iloc[i]:
            growth = px.loc[d1] / px.loc[d0]
        else:
            growth = (1.0 + rf) ** (days / 365.25)
        strat.iloc[i] = strat.iloc[i-1] * growth

    return passive, strat

def render_pdf(title, dates, close, bvps, buys, cycles_df, out_pdf, out_png, bench, strat_rf, params: BacktestParameters):
    bands=[0]+list(range(1,MAX_N+1))+list(range(-1,-MAX_N-1,-1))
    band_lines={n:(bvps*(FACTOR**n)).astype(float) for n in bands}
    tol = params.band_tolerance
    corridors={n:((line*(1-tol)).astype(float),(line*(1+tol)).astype(float)) for n,line in band_lines.items()}
    ymin_plot=max(np.nanmin(close.values),1e-6)*0.8; ymax_plot=np.nanmax(close.values)*1.2
    ticks=geometric_ticks(ymin_plot, ymax_plot, 7)

    anchor_dates=pd.to_datetime(cycles_df["BuyDate"]) if not cycles_df.empty else pd.to_datetime([])
    all_buy_dates=pd.to_datetime([b[0] for b in buys])
    non_anchor=all_buy_dates[~all_buy_dates.isin(anchor_dates)]

    with PdfPages(out_pdf) as pdf:
        # Main chart + Equity curve subplot (stacked)
        fig=plt.figure(figsize=(11,8.5), facecolor="white")
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.99)
        fig.subplots_adjust(top=0.93)
        gs = GridSpec(4, 1, height_ratios=[3,0.1,1,0.2], hspace=0.15)  # top signals, spacer, equity, spacer

        # Top: signals chart
        ax=fig.add_subplot(gs[0])
        
        # Get dynamic gold band classification (time-varying)
        gold_band_series = classify_floor_tiers_dynamic(dates, cycles_df, bvps)
        
        # Render corridors - split by gold band changes
        for n in sorted(corridors.keys()):
            lo, hi = corridors[n]
            
            # Check if this band is ever gold
            is_gold_dates = gold_band_series == n
            
            if is_gold_dates.any():
                # Render gold segments
                ax.fill_between(dates, lo.values, hi.values, 
                               where=is_gold_dates.values,
                               color=GOLD_FLOOR_FILL, alpha=0.25, linewidth=0,
                               interpolate=True)
                # Render blue segments (where not gold)
                ax.fill_between(dates, lo.values, hi.values,
                               where=~is_gold_dates.values,
                               color=CORRIDOR_FILL, alpha=0.25, linewidth=0,
                               interpolate=True)
            else:
                # This band is never gold, render all blue
                ax.fill_between(dates, lo.values, hi.values, 
                               color=CORRIDOR_FILL, alpha=0.25, linewidth=0)
        
        # Render corridor lines with dynamic colors
        for n in sorted(corridors.keys()):
            lo, hi = corridors[n]
            is_gold_dates = gold_band_series == n
            
            if is_gold_dates.any():
                # Split into segments
                # For simplicity, we'll draw lines that may overlap - matplotlib handles this reasonably
                for idx, (d, is_gold) in enumerate(zip(dates, is_gold_dates)):
                    if idx == 0:
                        continue
                    prev_date = dates[idx - 1]
                    color = GOLD_FLOOR_LINE if is_gold else CORRIDOR_LINE
                    ax.plot([prev_date, d], [lo.iloc[idx-1], lo.iloc[idx]], 
                           linestyle=(0,(4,3)), color=color, linewidth=0.8)
                    ax.plot([prev_date, d], [hi.iloc[idx-1], hi.iloc[idx]], 
                           linestyle=(0,(4,3)), color=color, linewidth=0.8)
            else:
                # Never gold, draw normally
                ax.plot(dates, lo.values, linestyle=(0,(4,3)), color=CORRIDOR_LINE, linewidth=0.8)
                ax.plot(dates, hi.values, linestyle=(0,(4,3)), color=CORRIDOR_LINE, linewidth=0.8)
        for n in sorted(band_lines.keys()):
            line=band_lines[n]
            ax.plot(dates, line.values, linestyle=":", color=NAVY, linewidth=1.0)
        ax.plot(dates, close.values, color=PRICE_COLOR, linewidth=1.8)
        
        # Add smoothed gold band ceiling line (LOWESS smoothing)
        # Calculate the actual price of the gold band for each date
        gold_ceiling_prices = []
        for i, d in enumerate(dates):
            gold_band = gold_band_series.iloc[i]
            if pd.notna(gold_band):
                # Calculate center price of the gold band
                bvps_value = float(bvps.loc[d])
                gold_price = bvps_value * (FACTOR ** gold_band)
                gold_ceiling_prices.append(gold_price)
            else:
                gold_ceiling_prices.append(np.nan)
        
        # Create series and interpolate NaN values first
        gold_ceiling_series = pd.Series(gold_ceiling_prices, index=dates)
        gold_ceiling_interpolated = gold_ceiling_series.interpolate(method='linear')
        
        # Apply LOWESS smoothing if we have valid data
        if gold_ceiling_interpolated.notna().any():
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # Prepare data for LOWESS (needs numeric x-axis)
            valid_mask = gold_ceiling_interpolated.notna()
            y_values = gold_ceiling_interpolated[valid_mask].values
            x_values = np.arange(len(dates))[valid_mask]
            
            # Apply LOWESS (frac controls smoothness: smaller = less smooth, larger = more smooth)
            # frac=0.2 means use 20% of data for each local regression
            smoothed = lowess(y_values, x_values, frac=0.20, return_sorted=False)
            
            # Create full series with smoothed values
            gold_ceiling_smooth = pd.Series(index=dates, dtype=float)
            gold_ceiling_smooth.iloc[valid_mask] = smoothed
            
            ax.plot(dates, gold_ceiling_smooth.values, color='#9932CC', linewidth=2.5, 
                   alpha=0.8, label='FMV - Buy Filter (LOWESS)', zorder=9)

        if len(non_anchor):
            ax.scatter(non_anchor, close.loc[non_anchor], marker="^", s=60, facecolors="none", edgecolors=GREEN, linewidths=1.2, zorder=6, label="BOB Buy")
        if len(anchor_dates):
            ax.scatter(anchor_dates, close.loc[anchor_dates], marker="^", s=70, color=GREEN, zorder=7, label="Anchor Buy")

        sells=pd.to_datetime(cycles_df["SellDate"].dropna()) if not cycles_df.empty else pd.to_datetime([])
        if len(sells):
            ax.scatter(sells, close.loc[sells], marker="v", s=65, color=RED, zorder=8, label="Sell")
            for _,r in cycles_df.dropna(subset=["SellDate"]).iterrows():
                d=pd.Timestamp(r["SellDate"]); px=close.loc[d]; ret=(r["SellPx"]/r["BuyPx"]-1.0)
                color=GREEN if ret>0 else RED
                tag=r["ExitType"]; tag="C" if tag=="ceiling-5down" else ("S" if tag=="stop" else ("T" if tag=="target" else ""))
                ax.annotate(f"{ret*100:.1f}% {tag}", (d,px), xytext=(5,10), textcoords="offset points",
                            fontsize=10, color=color, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.6, alpha=0.95))

        bh_text=f"Buy&Hold: {bench['TotalReturn']*100:+.1f}% | Ann {bench['Annualized']*100:+.1f}%"
        if strat_rf.get('HasData', True):  # Check if we have strategy data
            st_text=f"BOB+RF: {strat_rf['TotalReturn']*100:+.1f}% | Ann {strat_rf['Annualized']*100:+.1f}%"
            info_text = bh_text+"\n"+st_text
        else:
            st_text = "BOB+RF: No completed trades"
            info_text = bh_text+"\n"+st_text
        ax.text(0.99,0.02, info_text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color="#222", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", lw=0.6, alpha=0.95))

        ax.set_yscale("log"); ax.set_ylim([ticks[0], ticks[-1]]); ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: "${:,.0f}".format(y)))

        ax.grid(False); ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white", edgecolor="#CCC")

        # Bottom: cumulative equity
        ax2 = fig.add_subplot(gs[2], sharex=ax)
        passive, strat_curve = equity_curve_with_rf(pd.DatetimeIndex(dates), pd.Series(close.values, index=dates), cycles_df, rf=0.03)
        ax2.plot(dates, passive.values, linewidth=1.5, label="Passive (Buy & Hold)")
        ax2.plot(dates, strat_curve.values, linewidth=1.8, label="BOB Strategy (with 3% RF)")

        ax2.set_ylabel("Equity (Start = 100)")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white", edgecolor="#CCC")

        pdf.savefig(fig, dpi=300, facecolor="white")
        fig.savefig(out_png, dpi=300, facecolor="white")
        plt.close(fig)


# === Portfolio aggregation (equal-weight across concurrently LONG tickers) =========================
def _build_investment_mask(idx, cycles_df):
    mask = pd.Series(False, index=idx)
    if cycles_df is None or cycles_df.empty:
        return mask
    for _, r in cycles_df.iterrows():
        b = pd.Timestamp(r["BuyDate"]) if pd.notna(r["BuyDate"]) else None
        s = pd.Timestamp(r["SellDate"]) if pd.notna(r["SellDate"]) else None
        if b is None:
            continue
        if s is None:
            s = idx[-1]
        # clamp to bounds
        b = max(b, idx[0]); s = min(s, idx[-1])
        mask.loc[b:s] = True
    return mask

def _portfolio_equity_and_weights(prices_dict, cycles_dict, buys_dict, fmv_dict=None, rf=0.03, weighting='equal'):
    """
    Portfolio with configurable weighting:
    - 'equal': Equal weight across all positions at all times
    - 'undervalued': Weight proportional to discount from FMV ((FMV-Price)/FMV)
    """
    # Align on intersection of dates
    all_idx = None
    for s in prices_dict.values():
        all_idx = s.index if all_idx is None else all_idx.intersection(s.index)
    idx = pd.DatetimeIndex(sorted(all_idx.unique()))
    P = {k: v.reindex(idx).astype(float) for k, v in prices_dict.items()}
    
    # CRITICAL FIX: Reindex FMV to match portfolio dates
    F = {}
    if fmv_dict is not None:
        for k, fmv_series in fmv_dict.items():
            if fmv_series is not None:
                F[k] = fmv_series.reindex(idx).ffill().bfill().astype(float)
            else:
                F[k] = pd.Series(np.nan, index=idx, dtype=float)
    
    # Build investment masks (True when we have a position)
    masks = {k: _build_investment_mask(idx, cycles_dict.get(k)) for k in prices_dict.keys()}
    
    # Calculate weights based on method
    W = pd.DataFrame(0.0, index=idx, columns=list(prices_dict.keys()))
    
    if weighting == 'equal':
        # Simple equal weighting
        active = pd.Series(0.0, index=idx)
        for m in masks.values(): 
            active = active + m.astype(float)
        
        for k, m in masks.items():
            W[k] = m.astype(float) / active.replace(0, pd.NA)
        W = W.fillna(0.0)
        
    elif weighting == 'undervalued':
        # Undervaluation-based weighting using FMV discount
        if fmv_dict is None or len(F) == 0:
            raise ValueError("FMV dict required for undervalued weighting")
        
        # For each date, calculate discount to FMV for each active stock
        for i, date in enumerate(idx):
            active_stocks = [k for k in prices_dict.keys() if masks[k].iloc[i]]
            
            if not active_stocks:
                continue  # No active positions, weights remain 0
            
            # Check which active stocks have valid FMV
            valid_fmv_stocks = []
            for k in active_stocks:
                if k in F and pd.notna(F[k].iloc[i]) and F[k].iloc[i] > 0:
                    price = P[k].iloc[i]
                    if pd.notna(price) and price > 0:
                        valid_fmv_stocks.append(k)
            
            # If all active stocks have valid FMV, use undervalued weighting
            if len(valid_fmv_stocks) == len(active_stocks):
                discounts = {}
                for k in active_stocks:
                    price = P[k].iloc[i]
                    fmv = F[k].iloc[i]
                    # Calculate discount: (FMV - Price) / FMV
                    discount = max(0, (fmv - price) / fmv)
                    discounts[k] = discount
                
                total_discount = sum(discounts.values())
                if total_discount > 0:
                    # Allocate based on discount proportions
                    for k, discount in discounts.items():
                        W.at[date, k] = discount / total_discount
                else:
                    # All discounts are 0 or negative - fall back to equal weight
                    for k in active_stocks:
                        W.at[date, k] = 1.0 / len(active_stocks)
            else:
                # Some stocks missing valid FMV - fall back to equal weighting for this date
                for k in active_stocks:
                    W.at[date, k] = 1.0 / len(active_stocks)
    
    # APPLY POSITION SIZE CAP: Maximum 6% per position
    MAX_POSITION_SIZE = 0.06
    for i, date in enumerate(idx):
        row_weights = W.iloc[i]
        active_weights = row_weights[row_weights > 0]
        
        if len(active_weights) > 0:
            # Apply cap
            capped_weights = active_weights.clip(upper=MAX_POSITION_SIZE)
            
            # If sum of capped weights < 1.0, we hold the rest in cash
            # No renormalization - we want to hold cash if under-invested
            total_capped = capped_weights.sum()
            
            if total_capped < 1.0:
                # Under-invested due to caps - leave rest in cash
                for k in capped_weights.index:
                    W.at[date, k] = capped_weights[k]
            else:
                # If somehow still over 1.0 (shouldn't happen with 6% cap and many stocks)
                # normalize down to 1.0
                for k in capped_weights.index:
                    W.at[date, k] = capped_weights[k] / total_capped
    
    cash_w = 1.0 - W.sum(axis=1)
    
    # Debug output for undervalued weighting
    if weighting == 'undervalued':
        print(f"\n=== UNDERVALUED WEIGHTING DEBUG ===")
        print(f"Total dates: {len(idx)}")
        
        # Count how many dates use undervalued vs equal weighting
        undervalued_dates = 0
        equal_fallback_dates = 0
        
        # Track weight variation over time
        weight_variations = []
        
        # Track position cap impact
        dates_with_caps = 0
        max_positions_capped = []
        
        for i in range(len(idx)):
            if W.iloc[i].sum() > 0:  # Has active positions
                weights = W.iloc[i].values
                active_weights = weights[weights > 0]
                
                # Check if any position hit the 6% cap
                if any(w >= 0.059 for w in active_weights):  # 0.059 to account for rounding
                    dates_with_caps += 1
                    max_positions_capped.append(max(active_weights))
                
                if len(active_weights) > 1:
                    # Check if weights are equal (within small tolerance)
                    expected_equal = 1.0 / len(active_weights)
                    if all(abs(w - expected_equal) < 0.001 for w in active_weights):
                        equal_fallback_dates += 1
                    else:
                        undervalued_dates += 1
                        # Calculate standard deviation of active weights
                        weight_std = np.std(active_weights)
                        weight_variations.append(weight_std)
        
        print(f"Dates using UNDERVALUED weighting: {undervalued_dates}")
        print(f"Dates using EQUAL fallback weighting: {equal_fallback_dates}")
        print(f"Percentage using undervalued: {undervalued_dates/(undervalued_dates+equal_fallback_dates)*100:.1f}%")
        
        print(f"\n=== POSITION CAP (6%) IMPACT ===")
        print(f"Dates where at least one position hit 6% cap: {dates_with_caps}")
        if max_positions_capped:
            print(f"Average max position on capped dates: {np.mean(max_positions_capped)*100:.2f}%")
        
        # Show average cash held
        cash_held = 1.0 - W.sum(axis=1)
        avg_cash = cash_held[cash_held > 0.001].mean() if (cash_held > 0.001).any() else 0
        print(f"Average cash held (when >0): {avg_cash*100:.2f}%")
        
        if weight_variations:
            print(f"\nWeight variation statistics:")
            print(f"  Mean std dev of weights: {np.mean(weight_variations):.4f}")
            print(f"  Median std dev of weights: {np.median(weight_variations):.4f}")
            print(f"  Min std dev: {np.min(weight_variations):.4f}")
            print(f"  Max std dev: {np.max(weight_variations):.4f}")
        
        # Show weights at MULTIPLE points in time (beginning, middle, end)
        active_dates_idx = [i for i in range(len(idx)) if W.iloc[i].sum() > 0]
        if active_dates_idx:
            sample_points = [
                ('EARLY', active_dates_idx[min(10, len(active_dates_idx)-1)]),
                ('MIDDLE', active_dates_idx[len(active_dates_idx)//2]),
                ('LATE', active_dates_idx[-min(10, len(active_dates_idx))])
            ]
            
            print(f"\nSample weights at different time periods:")
            for label, i in sample_points:
                date = idx[i]
                total_invested = W.iloc[i].sum()
                cash_pct = (1.0 - total_invested) * 100
                print(f"\n  [{label}] Date: {date.date()} | Invested: {total_invested*100:.1f}% | Cash: {cash_pct:.1f}%")
                
                # Show top 5 positions
                date_weights = W.iloc[i][W.iloc[i] > 0].sort_values(ascending=False).head(5)
                for k, weight in date_weights.items():
                    price = P[k].iloc[i]
                    fmv = F[k].iloc[i] if k in F else np.nan
                    discount = ((fmv - price) / fmv * 100) if pd.notna(fmv) and fmv > 0 else np.nan
                    capped = " [CAPPED]" if weight >= 0.059 else ""
                    print(f"    {k}: weight={weight*100:.1f}% | price=${price:.2f} | FMV=${fmv:.2f} | discount={discount:.1f}%{capped}")
        print(f"=================================\n")
    
    # Calculate equity curve WITH NATURAL DRIFT
    # Key change: Only rebalance when there are NEW BUYS, otherwise let positions drift
    equity = pd.Series(100.0, index=idx, dtype=float)
    
    # Track ACTUAL weights (which drift with prices)
    ActualW = pd.DataFrame(0.0, index=idx, columns=list(prices_dict.keys()))
    ActualW.iloc[0] = W.iloc[0]  # Start with target weights on day 1
    
    # Build buy signal lookup for fast checking
    # buys_dict contains lists of (date, price) tuples
    buy_dates = {}
    for ticker, buys in buys_dict.items():
        if buys is not None and len(buys) > 0:
            # buys is a list of (date, price) tuples - extract dates
            buy_dates[ticker] = set(pd.to_datetime([b[0] for b in buys]))
    
    # DEBUG: Sample a few equity calculations if undervalued
    if weighting == 'undervalued':
        debug_calcs = []
    
    for i in range(1, len(idx)):
        d0, d1 = idx[i-1], idx[i]
        days = max((d1 - d0).days, 1)
        
        # Check if there are ANY buy signals on d1
        has_buys = any(d1 in buy_dates.get(ticker, set()) for ticker in prices_dict.keys())
        
        # DECISION POINT: Rebalance or drift?
        if has_buys:
            # NEW BUYS: Use target weights from W (rebalance to add new positions)
            weights_to_use = W.iloc[i-1]
            rebalanced = True
        else:
            # NO BUYS: Let positions drift naturally
            weights_to_use = ActualW.iloc[i-1]
            rebalanced = False
        
        # Calculate growth using appropriate weights
        growth = 0.0
        
        # DEBUG: Track first few calculations
        if weighting == 'undervalued' and len(debug_calcs) < 5 and weights_to_use.sum() > 0:
            calc_detail = {'date': d0, 'rebalanced': rebalanced, 'stocks': {}}
        
        for k in P.keys():
            if P[k].isna().loc[[d0,d1]].any():  # skip if NaN
                continue
            weight = weights_to_use[k]
            if weight > 0:
                stock_return = P[k].at[d1] / P[k].at[d0]
                weighted_return = weight * stock_return
                growth += weighted_return
                
                # Calculate new drifted weight for this stock
                # New weight = old weight × (stock return / portfolio growth)
                # We'll update after we know total growth
                
                # DEBUG: Capture details
                if weighting == 'undervalued' and len(debug_calcs) < 5 and weights_to_use.sum() > 0:
                    calc_detail['stocks'][k] = {
                        'weight': weight,
                        'return': stock_return,
                        'contribution': weighted_return
                    }
        
        cash_return = (1.0 - weights_to_use.sum()) * ((1.0 + rf) ** (days / 365.25))
        growth += cash_return
        
        # DEBUG: Save calculation detail
        if weighting == 'undervalued' and len(debug_calcs) < 5 and weights_to_use.sum() > 0 and 'stocks' in calc_detail and calc_detail['stocks']:
            calc_detail['cash_contribution'] = cash_return
            calc_detail['total_growth'] = growth
            debug_calcs.append(calc_detail)
        
        # Update equity
        equity.at[d1] = equity.at[d0] * growth
        
        # Update ActualW for next iteration (natural drift)
        if has_buys:
            # Rebalanced: actual weights = target weights
            ActualW.iloc[i] = W.iloc[i]
        else:
            # Drifted: calculate new weights based on price movements
            for k in P.keys():
                if weights_to_use[k] > 0 and not P[k].isna().loc[[d0,d1]].any():
                    stock_return = P[k].at[d1] / P[k].at[d0]
                    # New weight = old weight × stock return / portfolio growth
                    ActualW.at[d1, k] = weights_to_use[k] * stock_return / growth
                else:
                    ActualW.at[d1, k] = 0.0
    
    # DEBUG: Print equity calculation samples
    if weighting == 'undervalued' and debug_calcs:
        print(f"\n=== EQUITY CALCULATION DEBUG (WITH DRIFT) ===")
        for calc in debug_calcs:
            rebal_str = " [REBALANCED]" if calc['rebalanced'] else " [DRIFTED]"
            print(f"\nDate: {calc['date'].date()}{rebal_str}")
            for stock, details in calc['stocks'].items():
                print(f"  {stock}: weight={details['weight']:.3f} × return={details['return']:.4f} = {details['contribution']:.4f}")
            print(f"  Cash contribution: {calc['cash_contribution']:.4f}")
            print(f"  Total growth factor: {calc['total_growth']:.4f}")
        print(f"=================================\n")
    
    # CRITICAL: Print first and last few equity values to verify they're different
    if weighting in ['equal', 'undervalued']:
        print(f"\n=== EQUITY VALUES ({weighting.upper()}) ===")
        print(f"First 5 equity values:")
        for i in range(min(5, len(equity))):
            print(f"  {idx[i].date()}: ${equity.iloc[i]:.6f}")
        print(f"Last 5 equity values:")
        for i in range(max(0, len(equity)-5), len(equity)):
            print(f"  {idx[i].date()}: ${equity.iloc[i]:.6f}")
        print(f"Final value: ${equity.iloc[-1]:.2f}")
        print(f"=================================\n")
    
    return idx, equity, ActualW  # Return ActualW instead of W for more accurate reporting

def _calculate_calendar_returns(idx, equity, bm_idx=None, bm_curve=None):
    """Calculate calendar year returns for portfolio and benchmark"""
    # Create a series for easier resampling
    eq_series = pd.Series(equity.values, index=idx)
    
    # Get annual returns
    years = sorted(eq_series.index.year.unique())
    calendar_data = []
    
    for year in years:
        year_data = eq_series[eq_series.index.year == year]
        if len(year_data) < 2:
            continue
        
        start_val = year_data.iloc[0]
        end_val = year_data.iloc[-1]
        port_ret = (end_val / start_val - 1.0) * 100
        
        # Calculate benchmark return for the same year if available
        bm_ret = None
        if bm_idx is not None and bm_curve is not None:
            bm_series = pd.Series(bm_curve, index=bm_idx)
            bm_year_data = bm_series[bm_series.index.year == year]
            if len(bm_year_data) >= 2:
                bm_start = bm_year_data.iloc[0]
                bm_end = bm_year_data.iloc[-1]
                bm_ret = (bm_end / bm_start - 1.0) * 100
        
        calendar_data.append({
            'Year': str(year),
            'Portfolio': port_ret,
            'Benchmark': bm_ret
        })
    
    return calendar_data

def _render_portfolio_summary(base_title, idx, equity, W, out_pdf, bm_idx=None, bm_curve=None, bm_ann=None, bm_end=None):
    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(12,6))
        fig.suptitle("Core_BOB_Strategy", fontsize=14, y=0.97)
        from matplotlib.gridspec import GridSpec as _GS
        gs = _GS(4, 1, height_ratios=[3,0.1,1,0.2], hspace=0.15)
        # Equity (no benchmark)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(idx, equity.values, linewidth=1.8, label="Portfolio Equity (Start=100)")
        if bm_curve is not None and bm_idx is not None:
            ax1.plot(bm_idx, bm_curve, linewidth=1.2, label="Benchmark (Start=100)")
        ax1.set_yscale("log")
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
        ax1.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        ax1.legend(loc="upper left", fontsize=8)
        
        # --- Add Calendar Returns Table in upper left ---
        calendar_data = _calculate_calendar_returns(idx, equity, bm_idx, bm_curve)
        if calendar_data:
            # Create table data
            table_data = [['Year', 'Portfolio', 'Benchmark']]
            for row in calendar_data:
                port_str = f"{row['Portfolio']:+.1f}%"
                bm_str = f"{row['Benchmark']:+.1f}%" if row['Benchmark'] is not None else "—"
                table_data.append([row['Year'], port_str, bm_str])
            
            # Create a new axes for the table positioned in the upper left
            # Position: [left, bottom, width, height] in figure coordinates
            ax_table = fig.add_axes([0.12, 0.65, 0.20, 0.25])
            ax_table.axis('off')
            
            # Create the table
            table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                                  colWidths=[0.3, 0.35, 0.35])
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.3)
            
            # Style the table
            for i, row in enumerate(table_data):
                for j in range(3):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#4472C4')
                        cell.set_text_props(weight='bold', color='white', fontsize=7)
                        cell.set_edgecolor('white')
                    else:
                        cell.set_facecolor('white' if i % 2 == 0 else '#F2F2F2')
                        cell.set_edgecolor('#CCCCCC')
                        # Color code the returns
                        if j == 1:  # Portfolio column
                            try:
                                val = float(row[j].replace('%', '').replace('+', ''))
                                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                                cell.set_text_props(color=color, weight='bold', fontsize=7)
                            except:
                                pass
                        elif j == 2 and row[j] != "—":  # Benchmark column
                            try:
                                val = float(row[j].replace('%', '').replace('+', ''))
                                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                                cell.set_text_props(color=color, fontsize=7)
                            except:
                                pass
        # --- Add end value and annualized return (start is 100) ---
        start_val = float(equity.iloc[0])
        end_val   = float(equity.iloc[-1])
        days      = max((idx[-1] - idx[0]).days, 1)
        ann       = ( (end_val / start_val) ** (365.25 / days) ) - 1.0
        ax1.text(
            0.99, 0.02,
            (
                f"End: ${end_val:,.0f}\nAnn: {ann*100:.2f}%" +
                (f"\nBM Ann: {bm_ann*100:.2f}%\nBM End: ${bm_end:,.0f}" if bm_ann is not None and bm_end is not None else "")
            ),
            transform=ax1.transAxes, ha="right", va="bottom",
            fontsize=9, color="#222",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", lw=0.6, alpha=0.95)
        )
        # Weights stacked
        ax2 = fig.add_subplot(gs[2], sharex=ax1)
        names = list(W.columns)
        if names:
            ax2.stackplot(idx, [W[c].values for c in names], labels=names)
            ax2.legend(loc="upper left", fontsize=8, ncol=2)
        ax2.set_ylim(0,1)
        ax2.set_ylabel("Weight")
        ax2.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        pdf.savefig(fig); plt.close(fig)


def _render_aggregated_statistics(base_title, cycles_dict, out_pdf):
    """Create aggregated trading statistics across all tickers"""
    # Combine all cycles from all tickers
    all_cycles = []
    for ticker, cycles_df in cycles_dict.items():
        if cycles_df is not None and not cycles_df.empty:
            temp_df = cycles_df.copy()
            temp_df['Ticker'] = ticker
            all_cycles.append(temp_df)
    
    if not all_cycles:
        return  # No data to aggregate
    
    combined = pd.concat(all_cycles, ignore_index=True)
    completed = combined.dropna(subset=['SellDate']).copy()
    
    if len(completed) == 0:
        return  # No completed trades
    
    with PdfPages(out_pdf) as pdf:
        # Calculate statistics
        returns = completed['ReturnPct'] / 100.0  # Convert to decimal
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        
        # Calculate hold periods
        completed['HoldDays'] = (pd.to_datetime(completed['SellDate']) - 
                                pd.to_datetime(completed['BuyDate'])).dt.days
        
        # Calculate annualized returns per trade
        annualized_returns = []
        for _, row in completed.iterrows():
            ret = row['ReturnPct'] / 100.0
            days = row['HoldDays']
            if days > 0:
                ann_ret = ((1 + ret) ** (365.25 / days)) - 1
                annualized_returns.append(ann_ret)
        
        annualized_returns = pd.Series(annualized_returns) if annualized_returns else pd.Series([0])
        
        # Count tickers
        num_tickers = completed['Ticker'].nunique()
        
        stats = {
            'Number of Tickers': num_tickers,
            'Total Trades': len(completed),
            'Winning Trades': len(wins),
            'Losing Trades': len(losses),
            'Win Rate': len(wins) / len(completed) * 100 if len(completed) > 0 else 0,
            '': '',  # Spacer
            'Avg Return per Trade': returns.mean() * 100,
            'Avg Annualized Return per Trade': annualized_returns.mean() * 100,
            'Median Annualized Return per Trade': annualized_returns.median() * 100,
            'Avg Winning Trade': wins.mean() * 100 if len(wins) > 0 else 0,
            'Avg Losing Trade': losses.mean() * 100 if len(losses) > 0 else 0,
            'Best Trade': returns.max() * 100,
            'Worst Trade': returns.min() * 100,
            ' ': '',  # Spacer
            'Win/Loss Ratio': abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0,
            'Profit Factor': abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0,
            'Expectancy per Trade': returns.mean() * 100,
            '  ': '',  # Spacer
            'Avg Hold Period (days)': completed['HoldDays'].mean(),
            'Median Hold Period (days)': completed['HoldDays'].median(),
            'Min Hold Period (days)': completed['HoldDays'].min(),
            'Max Hold Period (days)': completed['HoldDays'].max(),
        }
        
        # Exit type breakdown
        exit_types = completed['ExitType'].value_counts()
        stats['    '] = ''  # Spacer
        stats['Exit Breakdown:'] = ''
        for exit_type, count in exit_types.items():
            label_map = {
                'ceiling-5down': 'Ceiling Exits',
                'ceiling': 'Ceiling Exits',
                'stop': 'Stop Loss Exits',
                'target': 'Target Exits'
            }
            label = label_map.get(exit_type, exit_type)
            stats[f'  {label}'] = f"{count} ({count/len(completed)*100:.1f}%)"
        
        # Create statistics table figure
        fig_stats = plt.figure(figsize=(11, 8.5), facecolor="white")
        fig_stats.suptitle(f"Aggregated Trading Statistics - Across All Tickers", 
                          fontsize=14, fontweight='bold', y=0.98)
        
        ax_stats = fig_stats.add_subplot(111)
        ax_stats.axis('tight')
        ax_stats.axis('off')
        
        # Add margins to fit everything
        fig_stats.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.9)
        
        # Prepare table data
        table_data = []
        for key, value in stats.items():
            if key.strip() == '':  # Spacer row
                table_data.append(['', ''])
            elif isinstance(value, str):
                if value == '':  # Section header
                    table_data.append([key, ''])
                else:
                    table_data.append([key, value])
            elif 'days' in key.lower() or 'Tickers' in key or 'Trades' in key:
                table_data.append([key, f"{int(value) if value == int(value) else value:.1f}"])
            elif 'rate' in key.lower() or key in ['Win/Loss Ratio', 'Profit Factor']:
                table_data.append([key, f"{value:.2f}"])
            elif 'Exit' in key or 'Breakdown' in key:
                table_data.append([key, ''])
            else:
                table_data.append([key, f"{value:.2f}%"])
        
        # Create table
        table = ax_stats.table(cellText=table_data,
                              colLabels=['Metric', 'Value'],
                              cellLoc='left',
                              loc='center',
                              colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)  # Reduced from 1.8 to 1.5
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                elif i % 2 == 0:  # Alternate row colors
                    cell.set_facecolor('#F2F2F2')
                
                # Highlight section headers
                if i > 0 and j == 0:
                    text = table_data[i-1][0]
                    if text.strip() and table_data[i-1][1] == '':
                        cell.set_text_props(weight='bold')
                        cell.set_facecolor('#D9E1F2')
                
                # Color code returns
                if i > 0 and j == 1 and table_data[i-1][0] in [
                    'Avg Return per Trade', 'Avg Annualized Return per Trade', 'Median Annualized Return per Trade',
                    'Avg Winning Trade', 'Avg Losing Trade',
                    'Best Trade', 'Worst Trade', 'Expectancy per Trade'
                ]:
                    try:
                        val_str = table_data[i-1][1].replace('%', '')
                        val = float(val_str)
                        if val > 0:
                            cell.set_text_props(color='green', weight='bold')
                        elif val < 0:
                            cell.set_text_props(color='red', weight='bold')
                    except:
                        pass
        
        pdf.savefig(fig_stats, dpi=300, facecolor="white")
        plt.close(fig_stats)


def _render_daily_report(base_name, idx, equity, W, cycles_dict, prices_dict, out_pdf, bm_idx=None, bm_curve=None, bm_ann=None, bm_end=None):
    """
    Generate a compact daily report PDF with 4 pages:
    1. Action Report (what changed vs yesterday)
    2. Current Portfolio Holdings
    3. Portfolio vs Benchmark Summary
    4. Aggregated Trading Statistics
    """
    with PdfPages(out_pdf) as pdf:
        current_date = idx[-1]
        
        # Page 1: Action Report
        if len(W) < 2:
            fig = plt.figure(figsize=(11, 8.5), facecolor="white")
            fig.text(0.5, 0.5, "Action Report\n\nInsufficient data for comparison\n(Need at least 2 days of data)", 
                    ha='center', va='center', fontsize=14, color='#666')
            fig.suptitle("Action Report: Changes vs. Yesterday", fontsize=16, fontweight='bold', y=0.95)
            pdf.savefig(fig, dpi=300, facecolor="white")
            plt.close(fig)
        else:
            yesterday_weights = W.iloc[-2]
            today_weights = W.iloc[-1]
            yesterday_date = W.index[-2]
            today_date = W.index[-1]
            
            yesterday_tickers = set([t for t in yesterday_weights.index if yesterday_weights[t] > 0])
            today_tickers = set([t for t in today_weights.index if today_weights[t] > 0])
            
            # Calculate basic sets for display
            fully_exited = yesterday_tickers - today_tickers  # Tickers completely removed
            newly_entered = today_tickers - yesterday_tickers  # Tickers newly added
            maintained = yesterday_tickers & today_tickers  # Tickers in both
            
            # Build sell data - check ALL tickers in cycles_dict for sells between dates
            sell_data = []
            for ticker in cycles_dict.keys():
                cycles_df = cycles_dict.get(ticker)
                if cycles_df is None or cycles_df.empty:
                    continue
                completed = cycles_df.dropna(subset=['SellDate'])
                if completed.empty:
                    continue
                
                # Find all sells that occurred in the window (after yesterday, up to and including today)
                for _, trade in completed.iterrows():
                    sell_date = pd.to_datetime(trade['SellDate'])
                    if yesterday_date < sell_date <= today_date:
                        buy_date = pd.to_datetime(trade['BuyDate'])
                        days_held = (sell_date - buy_date).days
                        sell_data.append({
                            'Ticker': ticker,
                            'Sell Date': sell_date.strftime('%Y-%m-%d'),
                            'Sell Price': float(trade['SellPx']),
                            'Return': float(trade['ReturnPct']),
                            'Exit Type': trade['ExitType'],
                            'Days Held': days_held
                        })
            
            # Build buy data - check ALL tickers in cycles_dict for buys between dates
            buy_data = []
            for ticker in cycles_dict.keys():
                cycles_df = cycles_dict.get(ticker)
                if cycles_df is None or cycles_df.empty:
                    continue
                
                # Find all buys that occurred in the window (after yesterday, up to and including today)
                for _, trade in cycles_df.iterrows():
                    buy_date = pd.to_datetime(trade['BuyDate'])
                    if yesterday_date < buy_date <= today_date:
                        buy_data.append({
                            'Ticker': ticker,
                            'Buy Date': buy_date.strftime('%Y-%m-%d'),
                            'Buy Price': float(trade['BuyPx']),
                            'Base Band': int(trade['BaseBand'])
                        })
            
            # Calculate actual holds (maintained positions with no activity)
            active_tickers = set([s['Ticker'] for s in sell_data] + [b['Ticker'] for b in buy_data])
            holds = maintained - active_tickers
            
            fig = plt.figure(figsize=(11, 8.5), facecolor="white")
            fig.suptitle("Action Report: Changes vs. Yesterday", fontsize=16, fontweight='bold', y=0.95)
            
            summary_text = (
                f"Yesterday: {yesterday_date.strftime('%Y-%m-%d')}  →  Today: {current_date.strftime('%Y-%m-%d')}\n"
                f"SELLS: {len(sell_data)}  |  BUYS: {len(buy_data)}  |  HOLDS: {len(holds)}"
            )
            fig.text(0.5, 0.90, summary_text, ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='#F0F0F0', edgecolor='#888', linewidth=1))
            
            if sell_data:
                ax_sell = fig.add_axes([0.08, 0.52, 0.84, 0.32])
                ax_sell.axis('off')
                ax_sell.text(0.5, 1.05, "SELLS (Exit Signals)", ha='center', va='bottom', 
                            fontsize=12, fontweight='bold', color='#B00020', transform=ax_sell.transAxes)
                
                sell_table_data = []
                for s in sell_data:
                    exit_type_map = {'ceiling-5down': 'Ceiling', 'ceiling': 'Ceiling', 'stop': 'Stop Loss', 'target': 'Target'}
                    exit_label = exit_type_map.get(s['Exit Type'], s['Exit Type'])
                    sell_table_data.append([
                        s['Ticker'], s['Sell Date'], f"${s['Sell Price']:.2f}",
                        f"{s['Return']:+.2f}%", exit_label, str(s['Days Held'])
                    ])
                
                sell_table = ax_sell.table(
                    cellText=sell_table_data,
                    colLabels=['Ticker', 'Sell Date', 'Sell Price', 'Return', 'Exit Type', 'Days Held'],
                    cellLoc='center', loc='center', colWidths=[0.15, 0.18, 0.15, 0.15, 0.17, 0.15]
                )
                sell_table.auto_set_font_size(False)
                sell_table.set_fontsize(8)
                sell_table.scale(1, 1.5)
                
                for i in range(len(sell_table_data) + 1):
                    for j in range(6):
                        cell = sell_table[(i, j)]
                        if i == 0:
                            cell.set_facecolor('#B00020')
                            cell.set_text_props(weight='bold', color='white', fontsize=9)
                        else:
                            cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
                            if j == 3:
                                try:
                                    val = float(sell_table_data[i-1][3].replace('%', '').replace('+', ''))
                                    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                                    cell.set_text_props(color=color, weight='bold')
                                except:
                                    pass
            else:
                fig.text(0.5, 0.68, "No sells", ha='center', va='center', fontsize=11, color='#666', style='italic')
            
            if buy_data:
                ax_buy = fig.add_axes([0.08, 0.12, 0.84, 0.32])
                ax_buy.axis('off')
                ax_buy.text(0.5, 1.05, "BUYS (New Signals)", ha='center', va='bottom', 
                           fontsize=12, fontweight='bold', color='#0E7C0E', transform=ax_buy.transAxes)
                
                buy_table_data = []
                for b in buy_data:
                    buy_table_data.append([b['Ticker'], b['Buy Date'], f"${b['Buy Price']:.2f}", str(b['Base Band'])])
                
                buy_table = ax_buy.table(
                    cellText=buy_table_data,
                    colLabels=['Ticker', 'Buy Date', 'Buy Price', 'Base Band'],
                    cellLoc='center', loc='center', colWidths=[0.25, 0.25, 0.25, 0.25]
                )
                buy_table.auto_set_font_size(False)
                buy_table.set_fontsize(8)
                buy_table.scale(1, 1.5)
                
                for i in range(len(buy_table_data) + 1):
                    for j in range(4):
                        cell = buy_table[(i, j)]
                        if i == 0:
                            cell.set_facecolor('#0E7C0E')
                            cell.set_text_props(weight='bold', color='white', fontsize=9)
                        else:
                            cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
            else:
                fig.text(0.5, 0.28, "No buys", ha='center', va='center', fontsize=11, color='#666', style='italic')
            
            pdf.savefig(fig, dpi=300, facecolor="white")
            plt.close(fig)
        
        # Page 2+: Current Portfolio Holdings (WITH PAGINATION)
        last_weights = W.iloc[-1]
        held_tickers = [ticker for ticker in last_weights.index if last_weights[ticker] > 0]
        
        if not held_tickers:
            fig = plt.figure(figsize=(11, 8.5), facecolor="white")
            fig.text(0.5, 0.5, "No Current Holdings\n\nPortfolio is 100% cash", 
                    ha='center', va='center', fontsize=16, color='#666')
            fig.suptitle("Current Portfolio Holdings", fontsize=16, fontweight='bold', y=0.95)
            pdf.savefig(fig, dpi=300, facecolor="white")
            plt.close(fig)
        else:
            holdings_data = []
            for ticker in held_tickers:
                cycles_df = cycles_dict.get(ticker)
                if cycles_df is None or cycles_df.empty:
                    continue
                open_cycles = cycles_df[cycles_df['SellDate'].isna()]
                if open_cycles.empty:
                    continue
                latest_open = open_cycles.iloc[-1]
                buy_date = pd.to_datetime(latest_open['BuyDate'])
                buy_price = float(latest_open['BuyPx'])
                current_price = float(prices_dict[ticker].iloc[-1])
                unrealized_return = (current_price / buy_price - 1.0) * 100
                days_held = (current_date - buy_date).days
                weight = last_weights[ticker] * 100
                completed_trades = cycles_df.dropna(subset=['SellDate'])
                avg_historical_return = completed_trades['ReturnPct'].mean() if len(completed_trades) > 0 else None
                
                holdings_data.append({
                    'Ticker': ticker, 'Weight': weight, 'Entry Date': buy_date.strftime('%Y-%m-%d'),
                    'Entry Price': buy_price, 'Current Price': current_price,
                    'Unrealized Return': unrealized_return, 'Days Held': days_held,
                    'Avg Historical Return': avg_historical_return
                })
            
            holdings_data.sort(key=lambda x: x['Weight'], reverse=True)
            
            # Calculate summary statistics (used on all pages)
            num_holdings = len(holdings_data)
            total_weight = sum(h['Weight'] for h in holdings_data)
            cash_weight = 100 - total_weight
            avg_return = sum(h['Unrealized Return'] for h in holdings_data) / num_holdings if num_holdings > 0 else 0
            
            # Pagination: split holdings into pages (15 holdings per page fits well)
            HOLDINGS_PER_PAGE = 15
            num_pages = (len(holdings_data) + HOLDINGS_PER_PAGE - 1) // HOLDINGS_PER_PAGE  # Ceiling division
            
            for page_num in range(num_pages):
                start_idx = page_num * HOLDINGS_PER_PAGE
                end_idx = min(start_idx + HOLDINGS_PER_PAGE, len(holdings_data))
                page_holdings = holdings_data[start_idx:end_idx]
                
                fig = plt.figure(figsize=(11, 8.5), facecolor="white")
                
                # Title with page number if multiple pages
                if num_pages > 1:
                    title = f"Current Portfolio Holdings (Page {page_num + 1} of {num_pages})"
                else:
                    title = "Current Portfolio Holdings"
                fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
                
                # Summary box (appears on all pages)
                summary_text = (
                    f"Portfolio Date: {current_date.strftime('%Y-%m-%d')}\n"
                    f"Number of Holdings: {num_holdings}\n"
                    f"Invested: {total_weight:.1f}%  |  Cash: {cash_weight:.1f}%\n"
                    f"Avg Unrealized Return: {avg_return:+.2f}%"
                )
                fig.text(0.5, 0.90, summary_text, ha='center', va='top', fontsize=9, 
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='#F0F0F0', edgecolor='#888', linewidth=1))
                
                # Table for this page's holdings
                ax = fig.add_axes([0.08, 0.08, 0.84, 0.78])
                ax.axis('off')
                
                table_data = []
                for h in page_holdings:
                    avg_hist_str = f"{h['Avg Historical Return']:+.2f}%" if h['Avg Historical Return'] is not None else "N/A"
                    table_data.append([
                        h['Ticker'], f"{h['Weight']:.1f}%", h['Entry Date'],
                        f"${h['Entry Price']:.2f}", f"${h['Current Price']:.2f}",
                        f"{h['Unrealized Return']:+.2f}%", str(h['Days Held']), avg_hist_str
                    ])
                
                col_labels = ['Ticker', 'Weight', 'Entry Date', 'Entry Price', 'Current Price', 
                              'Unrealized\nReturn', 'Days\nHeld', 'Avg Hist.\nReturn']
                
                table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center',
                                colWidths=[0.11, 0.09, 0.13, 0.12, 0.12, 0.13, 0.09, 0.11])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
                for i in range(len(table_data) + 1):
                    for j in range(8):
                        cell = table[(i, j)]
                        if i == 0:
                            cell.set_facecolor('#4472C4')
                            cell.set_text_props(weight='bold', color='white', fontsize=10)
                        else:
                            cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
                            if j == 5 or j == 7:
                                try:
                                    val_str = table_data[i-1][j].replace('%', '').replace('+', '')
                                    val = float(val_str)
                                    if val > 0:
                                        cell.set_text_props(color='green', weight='bold')
                                    elif val < 0:
                                        cell.set_text_props(color='red', weight='bold')
                                except:
                                    pass
                
                pdf.savefig(fig, dpi=300, facecolor="white")
                plt.close(fig)
        
        # Page 3: Portfolio vs Benchmark Summary
        fig = plt.figure(figsize=(11, 8.5), facecolor="white")
        fig.suptitle("Portfolio vs. Benchmark Summary", fontsize=16, fontweight='bold', y=0.95)
        
        start_val = float(equity.iloc[0])
        end_val = float(equity.iloc[-1])
        days = max((idx[-1] - idx[0]).days, 1)
        port_total = (end_val / start_val - 1.0) * 100
        port_ann = ((end_val / start_val) ** (365.25 / days) - 1.0) * 100
        
        summary_lines = [
            f"Period: {idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')} ({days} days)",
            "", "Portfolio:", f"  Start Value: $100.00", f"  End Value: ${end_val:.2f}",
            f"  Total Return: {port_total:+.2f}%", f"  Annualized Return: {port_ann:+.2f}%",
        ]
        
        if bm_ann is not None and bm_end is not None:
            bm_total = (bm_end / 100.0 - 1.0) * 100
            summary_lines.extend([
                "", "Benchmark:", f"  Start Value: $100.00", f"  End Value: ${bm_end:.2f}",
                f"  Total Return: {bm_total:+.2f}%", f"  Annualized Return: {bm_ann*100:+.2f}%",
                "", "Outperformance:", f"  Total: {port_total - bm_total:+.2f}%",
                f"  Annualized: {port_ann - bm_ann*100:+.2f}%",
            ])
        
        summary_text = "\n".join(summary_lines)
        fig.text(0.5, 0.72, summary_text, ha='center', va='top', fontsize=10, family='monospace',
                bbox=dict(boxstyle="round,pad=0.6", facecolor='#F0F0F0', edgecolor='#888', linewidth=1))
        
        ax = fig.add_axes([0.15, 0.12, 0.7, 0.45])
        ax.plot(idx, equity.values, linewidth=2, label="Portfolio", color='#4472C4')
        if bm_curve is not None and bm_idx is not None:
            ax.plot(bm_idx, bm_curve, linewidth=1.5, label="Benchmark", color='#888', linestyle='--')
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(loc="upper left", fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Value (Start = $100)", fontsize=10)
        
        pdf.savefig(fig, dpi=300, facecolor="white")
        plt.close(fig)
        
        # Page 4: Aggregated Trading Statistics
        all_cycles = []
        for ticker, cycles_df in cycles_dict.items():
            if cycles_df is not None and not cycles_df.empty:
                temp_df = cycles_df.copy()
                temp_df['Ticker'] = ticker
                all_cycles.append(temp_df)
        
        if all_cycles:
            combined = pd.concat(all_cycles, ignore_index=True)
            completed = combined.dropna(subset=['SellDate']).copy()
            
            if len(completed) > 0:
                fig_stats = plt.figure(figsize=(11, 8.5), facecolor="white")
                fig_stats.suptitle("Historical Trading Statistics", fontsize=16, fontweight='bold', y=0.95)
                
                ax_stats = fig_stats.add_axes([0.15, 0.15, 0.7, 0.7])
                ax_stats.axis('off')
                
                returns = completed['ReturnPct'] / 100.0
                wins = returns[returns > 0]
                losses = returns[returns <= 0]
                completed['HoldDays'] = (pd.to_datetime(completed['SellDate']) - 
                                        pd.to_datetime(completed['BuyDate'])).dt.days
                
                annualized_returns = []
                for _, row in completed.iterrows():
                    ret = row['ReturnPct'] / 100.0
                    days = row['HoldDays']
                    if days > 0:
                        ann_ret = ((1 + ret) ** (365.25 / days)) - 1
                        annualized_returns.append(ann_ret)
                
                annualized_returns = pd.Series(annualized_returns) if annualized_returns else pd.Series([0])
                num_tickers = completed['Ticker'].nunique()
                
                stats = {
                    'Number of Tickers': f"{num_tickers}",
                    'Total Trades': f"{len(completed)}",
                    'Winning Trades': f"{len(wins)}",
                    'Losing Trades': f"{len(losses)}",
                    'Win Rate': f"{len(wins) / len(completed) * 100:.2f}%",
                    '': '', 'Avg Return per Trade': f"{returns.mean() * 100:.2f}%",
                    'Avg Annualized Return': f"{annualized_returns.mean() * 100:.2f}%",
                    'Median Annualized Return': f"{annualized_returns.median() * 100:.2f}%",
                    'Avg Winning Trade': f"{wins.mean() * 100:.2f}%" if len(wins) > 0 else "0.00%",
                    'Avg Losing Trade': f"{losses.mean() * 100:.2f}%" if len(losses) > 0 else "0.00%",
                    'Best Trade': f"{returns.max() * 100:.2f}%",
                    'Worst Trade': f"{returns.min() * 100:.2f}%",
                    ' ': '',
                    'Win/Loss Ratio': f"{abs(wins.mean() / losses.mean()):.2f}" if len(losses) > 0 and losses.mean() != 0 else "N/A",
                    'Profit Factor': f"{abs(wins.sum() / losses.sum()):.2f}" if len(losses) > 0 and losses.sum() != 0 else "N/A",
                    'Expectancy per Trade': f"{returns.mean() * 100:.2f}%",
                    '  ': '', 'Avg Hold Period': f"{completed['HoldDays'].mean():.0f} days",
                    'Median Hold Period': f"{completed['HoldDays'].median():.0f} days",
                }
                
                exit_types = completed['ExitType'].value_counts()
                stats['   '] = ''
                stats['Exit Breakdown:'] = ''
                for exit_type, count in exit_types.items():
                    label_map = {'ceiling-5down': 'Ceiling Exits', 'ceiling': 'Ceiling Exits', 
                                'stop': 'Stop Loss Exits', 'target': 'Target Exits'}
                    label = label_map.get(exit_type, exit_type)
                    stats[f'  {label}'] = f"{count} ({count/len(completed)*100:.1f}%)"
                
                table_data = []
                for key, value in stats.items():
                    if key.strip() == '' or 'Exit' in key or 'Breakdown' in key:
                        table_data.append([key, ''])
                    else:
                        table_data.append([key, value])
                
                table = ax_stats.table(cellText=table_data, colLabels=['Metric', 'Value'],
                                      cellLoc='left', loc='center', colWidths=[0.6, 0.4])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
                for i in range(len(table_data) + 1):
                    for j in range(2):
                        cell = table[(i, j)]
                        if i == 0:
                            cell.set_facecolor('#4472C4')
                            cell.set_text_props(weight='bold', color='white')
                        elif i % 2 == 0:
                            cell.set_facecolor('#F2F2F2')
                        
                        if i > 0 and j == 0:
                            text = table_data[i-1][0]
                            if text.strip() and table_data[i-1][1] == '':
                                cell.set_text_props(weight='bold')
                                cell.set_facecolor('#D9E1F2')
                        
                        if i > 0 and j == 1:
                            try:
                                if '%' in table_data[i-1][1]:
                                    val_str = table_data[i-1][1].replace('%', '')
                                    val = float(val_str)
                                    if val > 0:
                                        cell.set_text_props(color='green', weight='bold')
                                    elif val < 0:
                                        cell.set_text_props(color='red', weight='bold')
                            except:
                                pass
                
                pdf.savefig(fig_stats, dpi=300, facecolor="white")
                plt.close(fig_stats)


def compute_backtest(chart_title: str, rec: pd.DataFrame, rule: str, params: Optional[BacktestParameters] = None) -> BacktestResult:
    """Run the core backtest logic without producing any files."""
    normalized_params = (params or BacktestParameters()).normalized()
    working = rec.copy()
    working["Date"] = pd.to_datetime(working["Date"])
    if normalized_params.backtest_years and not working["Date"].empty:
        latest_date = working["Date"].max()
        if pd.notna(latest_date):
            cutoff = latest_date - pd.DateOffset(years=normalized_params.backtest_years)
            trimmed = working[working["Date"] >= cutoff]
            if not trimmed.empty:
                working = trimmed.copy()
    title = str(chart_title)
    dates = pd.DatetimeIndex(working["Date"]).sort_values()
    close = pd.Series(working["Close"].values, index=dates).astype(float)
    bvps = build_bvps(working)

    buys_prelim = detect_bob_with_prior_inband(dates, close, bvps, normalized_params, fmv=None)
    cycles_prelim = build_cycles(dates, close, bvps, buys_prelim, rule, normalized_params)
    fmv = calculate_fmv(dates, close, bvps, cycles_prelim)
    buys = detect_bob_with_prior_inband(dates, close, bvps, normalized_params, fmv=fmv)

    cycles_df = build_cycles(dates, close, bvps, buys, rule, normalized_params)
    bench = buyhold_benchmark(dates, close)
    strat_rf = strategy_with_rf(dates, close, cycles_df, rf=0.03, span="cycles")

    warnings: List[str] = []
    if len(buys) == 0:
        warnings.append("WARNING: No buy signals detected")
        if len(dates) > 0:
            warnings.append(f"Date range: {dates[0]} to {dates[-1]}")
        if not close.empty:
            warnings.append(f"Price range: {close.min():.2f} to {close.max():.2f}")
        if bvps.notna().any():
            warnings.append(f"BVPS range: {bvps.min():.2f} to {bvps.max():.2f}")

    return BacktestResult(
        ticker=title,
        rule=rule,
        dates=dates,
        close=close,
        bvps=bvps,
        buys=buys,
        cycles=cycles_df,
        bench=bench,
        strat_rf=strat_rf,
        fmv=fmv,
        warnings=warnings,
        params=normalized_params,
    )


def run_one(rec, rule, chart_title, outdir, pdf_basename=None, params: Optional[BacktestParameters] = None):
    result = compute_backtest(chart_title, rec, rule, params=params)

    print(
        f"  [{result.ticker}] Loaded {len(result.dates)} dates, "
        f"detected {len(result.buys)} buy signals (FMV-filtered)"
    )
    for msg in result.warnings:
        print(f"    {msg}")

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    out_pdf = outdir / (f"{pdf_basename}.pdf" if pdf_basename else f"{result.ticker}_{rule}.pdf")
    out_png = outdir / f"{result.ticker}_{rule}_PAGE.png"
    out_csv = outdir / f"{result.ticker}_{rule}_cycles.csv"

    display_rule = "Core_BOB_Strategy" if rule == "revised_exit" else rule
    render_pdf(
        f"{result.ticker}  •  {display_rule}",
        result.dates,
        result.close,
        result.bvps,
        result.buys,
        result.cycles,
        out_pdf,
        out_png,
        result.bench,
        result.strat_rf,
        result.params,
    )

    result.cycles.to_csv(out_csv, index=False)

    with open(outdir / f"{result.ticker}_{rule}_benchmark.csv", "w", newline="") as f:
        import csv

        w = csv.writer(f)
        w.writerow(["BuyHold_Start", "BuyHold_End", "Total", "Annualized", "Days"])
        w.writerow(
            [
                result.bench["StartDate"].date(),
                result.bench["EndDate"].date(),
                f"{result.bench['TotalReturn']:.6f}",
                f"{result.bench['Annualized']:.6f}",
                result.bench["Days"],
            ]
        )

    with open(outdir / f"{result.ticker}_{rule}_strategy_rf.csv", "w", newline="") as f:
        import csv

        w = csv.writer(f)
        w.writerow(["HasData", "Start", "End", "Total", "Annualized", "Days", "NumCycles", "RF", "Span"])
        if result.strat_rf.get("HasData", True):
            w.writerow(
                [
                    True,
                    result.strat_rf["StartDate"].date(),
                    result.strat_rf["EndDate"].date(),
                    f"{result.strat_rf['TotalReturn']:.6f}",
                    f"{result.strat_rf['Annualized']:.6f}",
                    result.strat_rf["Days"],
                    result.strat_rf["NumCycles"],
                    0.03,
                    "cycles",
                ]
            )
        else:
            w.writerow([False, "", "", "", "", "", 0, 0.03, "cycles"])

    return result, out_pdf, out_png, out_csv



def main():
    import pandas as pd
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--rule", default="revised_exit", choices=["rule1","rule1_stop","revised_exit"])
    ap.add_argument("--weighting", default="equal", choices=["equal", "undervalued"], 
                    help="Portfolio weighting method: 'equal' or 'undervalued' (based on FMV discount)")
    ap.add_argument("--group-index", type=int, default=None)
    args=ap.parse_args()

    xls=Path(args.input); df=pd.read_excel(xls)  # read all columns; keep as-is
    base_name = Path(args.input).stem
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    cols = list(df.columns); ncols = len(cols)
    print(f"Total columns in Excel: {ncols}")
    print(f"Expected: Cols 0-1 (Portfolio), Cols 2-3 (Benchmark), then 9-col stock blocks starting at col 4")

    # Iterate 9-column blocks starting at column index 4 (E..M, N..V, ...)
    pdf_paths: List[str] = []
    results_by_title: Dict[str, BacktestResult] = {}
    start_indices = list(range(4, ncols, 9))
    gi = 0
    for start in start_indices:
        end = start + 9
        if end > ncols: break
        if args.group_index is not None and gi != args.group_index:
            gi += 1; continue
        print(f"\nProcessing group {gi}: columns {start}-{end-1}")
        rec = load_group(df, start, 9)
        if rec is None or rec.empty:
            print(f"  WARNING: Group {gi} loaded empty data")
            gi += 1; continue
        title = str(cols[start])
        print(f"  Title: {title}")
        pdf_tag = f"{base_name}_{stamp}_{gi}"
        result, out_pdf, _, _ = run_one(rec, args.rule, title, args.outdir, pdf_basename=pdf_tag)
        results_by_title[title] = result
        pdf_paths.append(str(out_pdf))
        gi += 1

    # Build portfolio summary inputs from computed results
    prices_dict: Dict[str, pd.Series] = {}
    cycles_dict: Dict[str, pd.DataFrame] = {}
    buys_dict: Dict[str, List[tuple]] = {}
    fmv_dict: Dict[str, Optional[pd.Series]] = {}

    for title_tmp, result in results_by_title.items():
        prices_dict[title_tmp] = result.close
        cycles_dict[title_tmp] = result.cycles
        buys_dict[title_tmp] = result.buys
        fmv_dict[title_tmp] = result.fmv

    port_pdf = None
    stats_pdf = None
    daily_pdf = None
    if prices_dict:
        print(f"\n!!! CRITICAL DEBUG: args.weighting = '{args.weighting}' !!!\n")
        idxP, eqP, WP = _portfolio_equity_and_weights(prices_dict, cycles_dict, buys_dict, 
                                                       fmv_dict=fmv_dict, rf=0.03, 
                                                       weighting=args.weighting)
        bm_dates = pd.Series(df.iloc[:,2]).apply(parse_date_maybe)
        bm_px = pd.Series(df.iloc[:,3]).apply(clean_number)
        bm = pd.Series(bm_px.values, index=pd.DatetimeIndex(bm_dates)).dropna()
        bm_aligned = bm.reindex(idxP).ffill().dropna()
        if not bm_aligned.empty:
            bm_curve = 100.0 * (bm_aligned / bm_aligned.iloc[0])
            bm_days = max((bm_aligned.index[-1] - bm_aligned.index[0]).days, 1)
            bm_ann = ( (bm_aligned.iloc[-1] / bm_aligned.iloc[0]) ** (365.25 / bm_days) ) - 1.0
            bm_end = float(bm_curve.iloc[-1])
        else:
            bm_curve = None; bm_ann = None; bm_end = None
        
        # Generate NEW Daily Report PDF (compact 4-page report)
        daily_pdf = Path(args.outdir) / f"{base_name}_{stamp}_DAILY_REPORT.pdf"
        _render_daily_report(base_name, idxP, eqP, WP, cycles_dict, prices_dict, daily_pdf, 
                            bm_idx=(bm_curve.index if bm_curve is not None else None), 
                            bm_curve=(bm_curve.values if bm_curve is not None else None), 
                            bm_ann=bm_ann, bm_end=bm_end)
        
        # Generate existing Portfolio PDF (UNCHANGED)
        port_pdf = Path(args.outdir) / f"{base_name}_{stamp}_PORTFOLIO.pdf"
        _render_portfolio_summary(base_name, idxP, eqP, WP, port_pdf, bm_idx=(bm_curve.index if bm_curve is not None else None), bm_curve=(bm_curve.values if bm_curve is not None else None), bm_ann=bm_ann, bm_end=bm_end)
        
        # Create aggregated statistics summary
        stats_pdf = Path(args.outdir) / f"{base_name}_{stamp}_STATS.pdf"
        _render_aggregated_statistics(base_name, cycles_dict, stats_pdf)
        
        # NEW: Export daily portfolio holdings snapshot (what we held each day)
        daily_holdings = []
        for i, date in enumerate(idxP):
            date_str = date.strftime('%Y-%m-%d')
            weights = WP.iloc[i]
            
            # Track held positions this day
            for ticker in weights.index:
                weight = weights[ticker]
                if weight > 0.001:  # Only include positions > 0.1%
                    # Get current price
                    if ticker in prices_dict:
                        current_price = float(prices_dict[ticker].iloc[i])
                        
                        # Find entry price for this position (from cycles)
                        entry_price = None
                        entry_date = None
                        unrealized_return = None
                        
                        if ticker in cycles_dict:
                            cycles = cycles_dict[ticker]
                            open_cycles = cycles[cycles['SellDate'].isna()]
                            if not open_cycles.empty:
                                latest = open_cycles.iloc[-1]
                                entry_price = float(latest['BuyPx'])
                                entry_date = pd.to_datetime(latest['BuyDate']).strftime('%Y-%m-%d')
                                unrealized_return = (current_price / entry_price - 1.0) * 100
                        
                        daily_holdings.append({
                            'Date': date_str,
                            'Ticker': ticker,
                            'Weight_%': f"{weight * 100:.2f}",
                            'Entry_Date': entry_date if entry_date else '',
                            'Entry_Price': f"{entry_price:.2f}" if entry_price else '',
                            'Current_Price': f"{current_price:.2f}",
                            'Unrealized_Return_%': f"{unrealized_return:+.2f}" if unrealized_return is not None else ''
                        })
            
            # Add cash position
            cash_weight = 1.0 - weights.sum()
            if cash_weight > 0.001:
                daily_holdings.append({
                    'Date': date_str,
                    'Ticker': 'CASH',
                    'Weight_%': f"{cash_weight * 100:.2f}",
                    'Entry_Date': '',
                    'Entry_Price': '',
                    'Current_Price': '',
                    'Unrealized_Return_%': ''
                })
        
        if daily_holdings:
            holdings_df = pd.DataFrame(daily_holdings)
            holdings_csv = Path(args.outdir) / f"{base_name}_{stamp}_DAILY_HOLDINGS.csv"
            holdings_df.to_csv(holdings_csv, index=False)
            print(f"Exported {len(holdings_df)} daily position records to: {holdings_csv.name}")
        
        # NEW: Export daily equity curve CSV
        equity_data = []
        for i in range(len(idxP)):
            date = idxP[i]
            value = eqP.iloc[i]
            
            # Calculate daily return
            if i == 0:
                daily_return = 0.0
            else:
                daily_return = (eqP.iloc[i] / eqP.iloc[i-1] - 1.0) * 100
            
            # Calculate cumulative return
            cumulative_return = (value / 100.0 - 1.0) * 100
            
            equity_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Portfolio_Value': f"{value:.2f}",
                'Daily_Return_%': f"{daily_return:+.4f}",
                'Cumulative_Return_%': f"{cumulative_return:+.2f}"
            })
        
        equity_df = pd.DataFrame(equity_data)
        equity_csv = Path(args.outdir) / f"{base_name}_{stamp}_DAILY_EQUITY.csv"
        equity_df.to_csv(equity_csv, index=False)
        print(f"Exported {len(equity_df)} days of equity data to: {equity_csv.name}")
        
        # Export master CSV with ALL completed trades
        all_trades = []
        for ticker, cycles_df in cycles_dict.items():
            if cycles_df is not None and not cycles_df.empty:
                ticker_trades = cycles_df.copy()
                ticker_trades.insert(0, 'Ticker', ticker)
                all_trades.append(ticker_trades)
        
        if all_trades:
            master_df = pd.concat(all_trades, ignore_index=True)
            master_df = master_df.sort_values('BuyDate').reset_index(drop=True)
            
            # Add HoldDays column
            master_df['HoldDays'] = (
                pd.to_datetime(master_df['SellDate']) - pd.to_datetime(master_df['BuyDate'])
            ).dt.days
            
            col_order = ['Ticker', 'BuyDate', 'BuyPx', 'BaseBand', 'SellDate', 'SellPx', 
                        'ExitType', 'CeilingBand', 'ReturnPct', 'HoldDays']
            master_df = master_df[[c for c in col_order if c in master_df.columns]]
            
            master_csv = Path(args.outdir) / f"{base_name}_{stamp}_ALL_TRADES.csv"
            master_df.to_csv(master_csv, index=False)
            print(f"Exported {len(master_df)} completed trades to: {master_csv.name}")
            
            # Export summary stats by ticker
            summary_by_ticker = []
            for ticker in master_df['Ticker'].unique():
                ticker_trades = master_df[master_df['Ticker'] == ticker]
                completed = ticker_trades.dropna(subset=['SellDate'])
                if len(completed) > 0:
                    wins = completed[completed['ReturnPct'] > 0]
                    summary_by_ticker.append({
                        'Ticker': ticker,
                        'Total_Trades': len(completed),
                        'Wins': len(wins),
                        'Losses': len(completed) - len(wins),
                        'Win_Rate_%': len(wins) / len(completed) * 100,
                        'Avg_Return_%': completed['ReturnPct'].mean(),
                        'Best_Trade_%': completed['ReturnPct'].max(),
                        'Worst_Trade_%': completed['ReturnPct'].min(),
                        'Avg_Hold_Days': completed['HoldDays'].mean()
                    })
            
            if summary_by_ticker:
                summary_df = pd.DataFrame(summary_by_ticker)
                summary_df = summary_df.sort_values('Avg_Return_%', ascending=False)
                summary_csv = Path(args.outdir) / f"{base_name}_{stamp}_SUMMARY_BY_TICKER.csv"
                summary_df.to_csv(summary_csv, index=False)
                print(f"Exported ticker summary to: {summary_csv.name}\n")

    # Merge per-ticker PDFs (and append the portfolio summary and aggregated stats)
    if pdf_paths:
        combined = Path(args.outdir) / f"{base_name}_{stamp}_MULTI.pdf"

        merger = PdfMerger()
        for p in pdf_paths:
            merger.append(p)
        if 'port_pdf' in locals() and (port_pdf is not None):
            merger.append(str(port_pdf))
        if 'stats_pdf' in locals() and (stats_pdf is not None):
            merger.append(str(stats_pdf))
  
        # WRITE ONCE, then CLOSE ONCE
        with open(combined, "wb") as f:
            merger.write(f)
        merger.close()

        # Optional: remove individual per-ticker PDFs; keep only the MULTI
        for p in pdf_paths:
            try:
                 Path(p).unlink(missing_ok=True)
            except Exception:
                pass


if __name__=="__main__":
    main()
