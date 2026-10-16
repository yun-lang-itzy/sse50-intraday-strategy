# -*- coding: utf-8 -*-
"""Aggregate the raw Level-2 files into the dashboard datasets (JSON, one row object per record).

Outputs: stocks, minute, market_minute, flow_minute, vap, depth, order_size, lifetime, lifetime_stats.
"""
import numpy as np
import pandas as pd

from common import (ADD, BUY, DEL, NAMES, SELL, SIZE_BINS, SIZE_LABELS, SW1, continuous, dump, hm_label, load,
                    minute_bucket, parse_args, to_ms)

args = parse_args(__doc__)
e = load(args, 'exe')
s = load(args, 'snp')
o = load(args, 'ord')

# ---------- trades: drop after-hours fixed-price trades (after 15:00:00.000)
e = e[e.Time <= 150000000].copy()
e['amt'] = e.Price.astype(float) * e.Volume
e['hm'] = minute_bucket(e.Time)
aggr = e.BSFlag.isin([BUY, SELL])                      # call-auction trades carry no aggressor side
e['buy'] = np.where(e.BSFlag == BUY, e.amt, 0.)
e['sell'] = np.where(e.BSFlag == SELL, e.amt, 0.)

# Aggressor ("taker") orders: sum each aggressor order's fills, bucket by its total amount
tk = e[aggr].copy()
tk['oid'] = np.where(tk.BSFlag == BUY, tk.BidOrder, tk.AskOrder)
tk['order_amt'] = tk.groupby(['Symbol', 'BSFlag', 'oid']).amt.transform('sum')
tk['cls'] = pd.cut(tk.order_amt, SIZE_BINS, labels=SIZE_LABELS, right=False)
tk['signed'] = np.where(tk.BSFlag == BUY, tk.amt, -tk.amt)
net_cls = tk.pivot_table(index='Symbol', columns='cls', values='signed', aggfunc='sum', observed=False).fillna(0)

# ---------- snapshots (continuous session, two-sided book)
pre = s.groupby('Symbol').PreClose.first()
sc = s[continuous(s.Time) & (s.BidPrice1 > 0) & (s.AskPrice1 > 0)].copy()
mid = (sc.BidPrice1 + sc.AskPrice1) / 2
sc['spr'] = (sc.AskPrice1 - sc.BidPrice1) / mid * 1e4
bv5 = sc[[f'BidVolume{i}' for i in range(1, 6)]].sum(axis=1)
av5 = sc[[f'AskVolume{i}' for i in range(1, 6)]].sum(axis=1)
sc['imb'] = (bv5 - av5) / (bv5 + av5)
sc['depth'] = (bv5 * sc.BidPrice1 + av5 * sc.AskPrice1) / 2
sc['hm'] = minute_bucket(sc.Time)

# ---------- orders: SSE publishes only the resting part of an order (A) and its cancels (D)
oa = o[o.OrderKind == ADD]
od = o[o.OrderKind == DEL]
o2 = o[o.OrderKind.isin([ADD, DEL])].copy()
o2['hm'] = minute_bucket(o2.Time)
o2['add'] = (o2.OrderKind == ADD).astype(int)
o2['cxl'] = (o2.OrderKind == DEL).astype(int)

# ---------- minute (per stock)
g = e.groupby(['Symbol', 'hm'])
m = pd.DataFrame({'px': g.Price.last(), 'vol': g.Volume.sum(), 'amt': g.amt.sum(),
                  'buy': g.buy.sum(), 'sell': g.sell.sum(), 'n': g.size()})
m = (m.join(sc.groupby(['Symbol', 'hm']).agg(spr=('spr', 'mean'), imb=('imb', 'mean')), how='outer')
      .join(o2.groupby(['Symbol', 'hm']).agg(adds=('add', 'sum'), cxls=('cxl', 'sum')), how='outer')
      .reset_index())
m = m[m.hm >= 925].sort_values(['Symbol', 'hm'])
m['px'] = m.groupby('Symbol').px.ffill()
for c in ['vol', 'amt', 'buy', 'sell', 'n', 'adds', 'cxls']:
    m[c] = m[c].fillna(0)
m['vwap'] = m.groupby('Symbol').amt.cumsum() / m.groupby('Symbol').vol.cumsum()
m['ret'] = m.px / m.Symbol.map(pre) - 1
opt = lambda v, d: None if pd.isna(v) else round(float(v), d)
dump([dict(code=str(r.Symbol), t=hm_label(r.hm), price=round(float(r.px), 3), vwap=round(float(r.vwap), 3),
           ret_pct=round(float(r.ret) * 100, 3), volume=int(r.vol), amount=round(float(r.amt)),
           buy_amt=round(float(r.buy)), sell_amt=round(float(r.sell)), trades=int(r.n),
           spread_bps=opt(r.spr, 2), imbalance=opt(r.imb, 3), orders=int(r.adds), cancels=int(r.cxls))
      for r in m.itertuples()], args, 'minute.json')

# ---------- market minute (50 stocks combined) + breadth
mm = m.groupby('hm').agg(amount=('amt', 'sum'), trades=('n', 'sum'), orders=('adds', 'sum'), cancels=('cxls', 'sum'),
                         buy=('buy', 'sum'), sell=('sell', 'sum'), ew=('ret', 'mean'), spr=('spr', 'mean'),
                         up=('ret', lambda x: int((x > 0).sum())), down=('ret', lambda x: int((x < 0).sum())),
                         flat=('ret', lambda x: int((x == 0).sum()))).reset_index()
dump([dict(t=hm_label(r.hm), amount=round(r.amount), trades=int(r.trades), orders=int(r.orders), cancels=int(r.cancels),
           net_buy=round(r.buy - r.sell), ew_ret_pct=round(r.ew * 100, 3), spread_bps=opt(r.spr, 2),
           up_count=int(r.up), down_count=int(r.down), flat_count=int(r.flat))
      for r in mm.itertuples()], args, 'market_minute.json')

# ---------- size-bucket flow per minute (continuous session), with running totals
tk['t'] = [hm_label(h) for h in minute_bucket(tk.Time)]
p = tk.pivot_table(index='t', columns='cls', values='signed', aggfunc='sum', observed=False).fillna(0).sort_index()
cum = p.cumsum()
dump([dict(t=t, **{f'net_{k}_wan': round(p.loc[t, k] / 1e4) for k in SIZE_LABELS[::-1]},
           **{f'cum_{k}_yi': round(cum.loc[t, k] / 1e8, 3) for k in SIZE_LABELS[::-1]}) for t in p.index],
     args, 'flow_minute.json')

# ---------- per-stock daily summary
last = s.sort_values('Time').groupby('Symbol').last()
stocks = []
for sym in sorted(e.Symbol.unique()):
    L, ex = last.loc[sym], e[e.Symbol == sym]
    c, so, a, d = ex[ex.BSFlag.isin([BUY, SELL])], sc[sc.Symbol == sym], oa[oa.Symbol == sym], od[od.Symbol == sym]
    close, pc = float(ex.Price.iloc[-1]), float(pre[sym])
    r = dict(code=str(sym), name=NAMES.get(sym, str(sym)), industry=SW1.get(sym, ''),
             board='科创板' if str(sym).startswith('688') else '主板',
             pre_close=round(pc, 3), open=round(float(L.Open), 3), high=round(float(L.High), 3),
             low=round(float(L.Low), 3), close=round(close, 3), ret_pct=round((close / pc - 1) * 100, 3),
             range_pct=round((float(L.High) - float(L.Low)) / pc * 100, 3),
             amount=round(float(ex.amt.sum())), volume=int(ex.Volume.sum()), trades=int(len(ex)),
             avg_trade_amt=round(float(ex.amt.mean())), buy_amt=round(float(c.buy.sum())),
             sell_amt=round(float(c.sell.sum())), net_buy=round(float(c.buy.sum() - c.sell.sum())),
             **{f'net_{k}': round(float(net_cls.loc[sym, k])) for k in SIZE_LABELS[::-1]},
             orders=int(len(a)), cancels=int(len(d)),
             cancel_vol_ratio=round(float(d.Volume.sum() / max(a.Volume.sum(), 1)), 4),
             spread_bps=round(float(so.spr.mean()), 2),
             spread_ticks=round(float(((so.AskPrice1 - so.BidPrice1) / 0.01).mean()), 2),
             imbalance=round(float(so.imb.mean()), 3), depth_top5=round(float(so.depth.mean())))
    for k in ['amount', 'net_buy', 'net_xlarge', 'net_large', 'net_medium', 'net_small']:
        r[k + '_yi'] = round(r[k] / 1e8, 2)
    r['cancel_pct'] = round(r['cancel_vol_ratio'] * 100, 1)
    stocks.append(r)
dump(stocks, args, 'stocks.json')

# ---------- traded amount by price level (30 bins per stock)
vap = []
for sym, ex in e.groupby('Symbol'):
    lo, hi = ex.Price.min(), ex.Price.max()
    w = (hi - lo) / 30 or 0.01
    b = np.clip(((ex.Price - lo) / w).astype(int), 0, 29)
    for k, r in ex.assign(b=b).groupby('b').agg(buy=('buy', 'sum'), sell=('sell', 'sum'), amt=('amt', 'sum')).iterrows():
        vap.append(dict(code=str(sym), price_lo=round(float(lo + k * w), 3), price_hi=round(float(lo + (k + 1) * w), 3),
                        amount=round(r.amt), buy_amt=round(r.buy), sell_amt=round(r.sell)))
dump(vap, args, 'vap.json')

# ---------- all-day average depth by level (lots)
depth = []
for sym, gg in sc.groupby('Symbol'):
    for i in range(1, 11):
        for side, V in [('bid', 'BidVolume'), ('ask', 'AskVolume')]:
            depth.append(dict(code=str(sym), side=side, level=i, avg_lots=round(float(gg[f'{V}{i}'].mean()) / 100, 1),
                              snapshots=int(len(gg))))
dump(depth, args, 'depth.json')

# ---------- size distribution of new (resting) orders
bins = [0, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, np.inf]
labs = ['1手', '2手', '3–5手', '6–10手', '11–20手', '21–50手', '51–100手', '101–500手', '>500手']
h = (oa.assign(c=pd.cut(oa.Volume, bins, labels=labs), side=np.where(oa.FunctionCode == BUY, 'buy', 'sell'))
       .groupby(['c', 'side'], observed=False).agg(count=('Volume', 'size'), shares=('Volume', 'sum')).reset_index())
dump([dict(size_bucket=str(r.c), order=i // 2, side=r.side, orders=int(r.count), shares=int(r.shares))
      for i, r in enumerate(h.itertuples())], args, 'order_size.json')

# ---------- resting-order lifetime: match each cancel to its add by original order number
a_ = oa[['Symbol', 'OrderOriNo', 'Time']]
d_ = od[['Symbol', 'OrderOriNo', 'Time']]
j = d_.merge(a_, on=['Symbol', 'OrderOriNo'], suffixes=('_d', '_a'))
j = j[j.Time_a >= 93000000]
lt = (to_ms(j.Time_d) - to_ms(j.Time_a)).astype(float)
lt = np.where((j.Time_a.values < 113000000) & (j.Time_d.values >= 130000000), lt - 90 * 60000, lt) / 1000  # lunch
j = j.assign(lt=lt, board=np.where(j.Symbol.astype(str).str.startswith('688'), '科创板', '主板'))
grid = [0.5, 1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600, 7200]
cdf, stats = [], []
for b, gg in j.groupby('board'):
    x = gg['lt']
    cdf += [dict(board=b, seconds=v, cum_share_pct=round(float((x <= v).mean() * 100), 2)) for v in grid]
    stats.append(dict(board=b, cancels=int(len(gg)), median_s=round(float(x.median()), 1),
                      within_10s_pct=round(float((x <= 10).mean() * 100), 1),
                      within_1s_pct=round(float((x <= 1).mean() * 100), 1)))
dump(cdf, args, 'lifetime.json')
dump(stats, args, 'lifetime_stats.json')
