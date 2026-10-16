# -*- coding: utf-8 -*-
"""Microstructure factors on 3-second snapshots vs future mid-price change.

Split within the day: train 09:30–11:30, validation 13:00–14:00, holdout 14:00–14:57.
IC = Pearson correlation computed per stock, then averaged with equal weights.
Outputs: factor_ic, composite_ic, composite_quintile.
"""
import numpy as np
import pandas as pd

from common import ADD, BUY, SELL, continuous, dump, load, parse_args, to_ms

args = parse_args(__doc__)
AM0, AM1, PM0, PM1 = to_ms([93000000, 113000000, 130000000, 145700000])
TRAIN_END, VALID_END = to_ms([113000000, 140000000])
# Seconds; snapshots arrive every 3 s. Longer horizons are left out: the 57-minute holdout
# holds only a dozen non-overlapping windows per stock at 180–300 s.
HORIZONS = [3, 6, 12, 30, 60]
FACTORS = ['买卖一档量差', '五档盘口量差', '十档加权量差', '微价偏离', '近5秒主动买卖压力', '近30秒主动买卖压力',
           '近5秒订单流不平衡', '近5秒撤单压力', '近5秒新增挂单压力', '近5秒价格动量', '近5秒成交量', '买卖价差']

s = load(args, 'snp')
e = load(args, 'exe', ['Symbol', 'BSFlag', 'Volume', 'Time'])
o = load(args, 'ord', ['Symbol', 'OrderKind', 'FunctionCode', 'Volume', 'Time'])
s = s[continuous(s.Time) & (s.BidPrice1 > 0) & (s.AskPrice1 > 0)].copy()
s['T'] = to_ms(s.Time)
e = e[e.BSFlag.isin([BUY, SELL])].copy()
e['T'] = to_ms(e.Time)
o = o[o.OrderKind.isin([ADD, 68])].copy()
o['T'] = to_ms(o.Time)


def csum(x):
    return np.concatenate([[0.], np.cumsum(x)])


def window(times, cum, T, w):
    """Sum over (T - w, T] using a cumulative sum aligned to sorted event times."""
    return cum[np.searchsorted(times, T, 'right')] - cum[np.searchsorted(times, T - w, 'right')]


def ratio(num, den):
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.)


frames = []
for sym, g in s.groupby('Symbol'):
    g = g.sort_values('T')
    T = g['T'].values.astype(np.int64)
    bp, ap = g.BidPrice1.values.astype(float), g.AskPrice1.values.astype(float)
    bv = np.stack([g[f'BidVolume{i}'].values for i in range(1, 11)], 1)
    av = np.stack([g[f'AskVolume{i}'].values for i in range(1, 11)], 1)
    mid = (bp + ap) / 2
    f = {}
    f[0] = ratio(bv[:, 0] - av[:, 0], bv[:, 0] + av[:, 0])
    f[1] = ratio(bv[:, :5].sum(1) - av[:, :5].sum(1), bv[:, :5].sum(1) + av[:, :5].sum(1))
    wt = 1 / np.arange(1, 11)
    f[2] = ratio((bv * wt).sum(1) - (av * wt).sum(1), (bv * wt).sum(1) + (av * wt).sum(1))
    micro = (av[:, 0] * bp + bv[:, 0] * ap) / (bv[:, 0] + av[:, 0])
    f[3] = (micro - mid) / mid * 1e4

    ge = e[e.Symbol == sym].sort_values('T')
    te = ge['T'].values
    signed = csum(np.where(ge.BSFlag == BUY, 1., -1.) * ge.Volume.values)
    total = csum(ge.Volume.values.astype(float))
    f[4] = ratio(window(te, signed, T, 5000), window(te, total, T, 5000))
    f[5] = ratio(window(te, signed, T, 30000), window(te, total, T, 30000))
    f[10] = np.log1p(window(te, total, T, 5000))

    go = o[o.Symbol == sym].sort_values('T')
    to = go['T'].values
    v = go.Volume.values.astype(float)
    is_add, is_buy = (go.OrderKind == ADD).values, (go.FunctionCode == BUY).values
    W = {k: window(to, csum(v * msk), T, 5000) for k, msk in
         {'ab': is_add & is_buy, 'as': is_add & ~is_buy, 'db': ~is_add & is_buy, 'ds': ~is_add & ~is_buy}.items()}
    f[6] = ratio((W['ab'] - W['db']) - (W['as'] - W['ds']), W['ab'] + W['as'] + W['db'] + W['ds'])
    f[7] = ratio(W['ds'] - W['db'], W['db'] + W['ds'])     # ask cancels count as buying pressure
    f[8] = ratio(W['ab'] - W['as'], W['ab'] + W['as'])

    j = np.searchsorted(T, T - 5000, 'right') - 1
    ok = (j >= 0) & (T - 5000 >= np.where(T < AM1, AM0, PM0))
    f[9] = np.where(ok, (mid / mid[np.maximum(j, 0)] - 1) * 1e4, np.nan)
    f[11] = (ap - bp) / mid * 1e4

    df = pd.DataFrame({FACTORS[k]: f[k] for k in range(12)})
    df['sym'], df['T'] = sym, T
    session_end = np.where(T < AM1, AM1, PM1)
    for h in HORIZONS:
        jj = np.searchsorted(T, T + h * 1000, 'right') - 1
        df[f'y{h}'] = np.where(T + h * 1000 <= session_end, (mid[jj] / mid - 1) * 1e4, np.nan)
    frames.append(df)

X = pd.concat(frames, ignore_index=True)
X['seg'] = np.where(X['T'] < TRAIN_END, 'train', np.where(X['T'] < VALID_END, 'valid', 'holdout'))
print('samples', X.seg.value_counts().to_dict())


def ic(d, a, b):
    r = []
    for _, g in d.groupby('sym'):
        x, y = g[a].values, g[b].values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 30 and x[m].std() > 0 and y[m].std() > 0:
            r.append(np.corrcoef(x[m], y[m])[0, 1])
    return float(np.mean(r))


dump([dict(factor=fn, factor_order=i, horizon_s=h,
           **{f'ic_{sg}': round(ic(X[X.seg == sg], fn, f'y{h}'), 3) for sg in ['train', 'valid', 'holdout']})
      for i, fn in enumerate(FACTORS) for h in HORIZONS], args, 'factor_ic.json')

# Ridge composite: z-score each factor per stock with train-segment stats, fit on train,
# pick the penalty on validation IC, report holdout IC.
Z = X.copy()
for fn in FACTORS:
    st = Z[Z.seg == 'train'].groupby('sym')[fn].agg(['mean', 'std'])
    Z[fn] = ((Z[fn] - Z.sym.map(st['mean'])) / Z.sym.map(st['std']).replace(0, np.nan)).clip(-5, 5).fillna(0)
F = Z[FACTORS].values.astype(np.float64)
composite, w30 = [], None
with np.errstate(all='ignore'):        # Accelerate BLAS on macOS can emit spurious matmul warnings
    for h in HORIZONS:
        tr = (Z.seg == 'train').values & Z[f'y{h}'].notna().values
        A, y = F[tr], Z.loc[tr, f'y{h}'].values
        y = y - y.mean()
        best = None
        for lam in [1, 10, 100, 1000, 1e4, 1e5]:
            w = np.linalg.solve(A.T @ A + lam * len(A) / 1e4 * np.eye(len(FACTORS)), A.T @ y)
            Z['score'] = F @ w
            v = ic(Z[Z.seg == 'valid'], 'score', f'y{h}')
            if best is None or v > best[0]:
                best = (v, lam, w)
        v, lam, w = best
        Z['score'] = F @ w
        composite.append(dict(horizon_s=h, model='12因子组合（岭回归）', ic_valid=round(v, 3),
                              ic_holdout=round(ic(Z[Z.seg == 'holdout'], 'score', f'y{h}'), 3), ridge_lambda=lam))
        if h == 30:
            w30 = w
dump(composite, args, 'composite_ic.json')

# Quintiles of the 30 s composite within each stock -> mean future 30 s mid change
Z['score'] = F @ w30
rows = []
for sg, name in [('valid', '验证段'), ('holdout', '尾段留出')]:
    d = Z[(Z.seg == sg) & Z['y30'].notna()].copy()
    d['q'] = d.groupby('sym')['score'].transform(lambda x: pd.qcut(x.rank(method='first'), 5, labels=False) + 1)
    rows += [dict(segment=name, quintile=int(k), ret_bps=round(float(v), 3)) for k, v in d.groupby('q')['y30'].mean().items()]
dump(rows, args, 'composite_quintile.json')
