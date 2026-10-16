# -*- coding: utf-8 -*-
"""Print the headline figures quoted in README.md from the JSON results (no raw data needed)."""
import json
import os
import sys

import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else 'output'
rd = lambda n: pd.DataFrame(json.load(open(os.path.join(out, n + '.json'), encoding='utf-8')))
st, mm, fl = rd('stocks'), rd('market_minute').set_index('t'), rd('flow_minute').set_index('t')
ic, comp, q, life = rd('factor_ic'), rd('composite_ic'), rd('composite_quintile'), rd('lifetime_stats')


def spearman(a, b):
    return round(a.rank().corr(b.rank()), 2)


print('== 市场')
print(f"成交额 {st.amount.sum() / 1e8:.1f} 亿元；上涨 {(st.ret_pct > 0).sum()} / 下跌 {(st.ret_pct < 0).sum()}；"
      f"等权涨跌 {st.ret_pct.mean():.2f}%")
print('等权路径', mm.ew_ret_pct.loc[['09:30', '10:30', '11:29', '14:00', '15:00']].round(2).to_dict(),
      '最低', mm.ew_ret_pct.idxmin(), round(mm.ew_ret_pct.min(), 2))
print('市场宽度(涨/跌)', {t: (int(mm.up_count[t]), int(mm.down_count[t])) for t in ['09:30', '10:30', '13:00', '14:56']})
print(f"主动净买入 上午 {mm.net_buy[mm.index < '12:00'].sum() / 1e8:.1f} 亿，下午 {mm.net_buy[mm.index > '12:00'].sum() / 1e8:.1f} 亿")
print('超大单累计(亿)', fl.cum_xlarge_yi.loc[['10:00', '10:30', '13:00', '14:00', '14:56']].to_dict())
print('分档净买入(亿)', {k: round(float(st[f'net_{k}'].sum()) / 1e8, 2) for k in ['xlarge', 'large', 'medium', 'small']})
tot = mm.amount.sum()
seg = lambda a, b: round(mm.amount[(mm.index >= a) & (mm.index <= b)].sum() / tot * 100, 1)
print(f"成交占比 开盘竞价 {mm.amount['09:25'] / tot * 100:.1f}% 前30分钟 {seg('09:30', '09:59')}% "
      f"尾盘30分钟 {seg('14:27', '14:56')}% 收盘竞价 {mm.amount['15:00'] / tot * 100:.1f}%")

print('== 行业 / 个股')
ind = st.groupby('industry').agg(n=('code', 'size'), ret=('ret_pct', 'mean'), amt=('amount_yi', 'sum'),
                                 net=('net_buy_yi', 'sum')).sort_values('amt', ascending=False).round(2)
print(ind.head(6).to_string())
print('前5成交', st.nlargest(5, 'amount').name.tolist(), round(st.nlargest(5, 'amount').amount.sum() / st.amount.sum() * 100, 1), '%')
star = st.board == '科创板'
print(f"科创板 成交占比 {st[star].amount.sum() / st.amount.sum() * 100:.1f}% 平均涨跌 {st[star].ret_pct.mean():.2f}% | 主板 {st[~star].ret_pct.mean():.2f}%")
print('价差≤1.1个最小价位', int((st.spread_ticks <= 1.1).sum()), '/', len(st))
print('盘口不平衡 vs 涨跌 秩相关', spearman(st.imbalance, st.ret_pct))

print('== 委托')
print(life.to_string(index=False))

print('== 因子（尾段留出 IC，3–60 秒）')
p = ic[ic.horizon_s <= 60].pivot(index='factor_order', columns='horizon_s', values='ic_holdout')
p.index = ic.drop_duplicates('factor_order').set_index('factor_order').factor.loc[p.index]
print(p.to_string())
print(comp[comp.horizon_s <= 60][['horizon_s', 'ic_valid', 'ic_holdout']].to_string(index=False))
g = q.set_index(['segment', 'quintile']).ret_bps
print('Q5-Q1 (bp)', {s: round(float(g[s][5] - g[s][1]), 2) for s in ['验证段', '尾段留出']})
