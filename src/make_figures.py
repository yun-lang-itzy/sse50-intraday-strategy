# -*- coding: utf-8 -*-
"""Draw the submission figures from the repo's output/*.json (no raw data needed).

Usage: python src/make_figures.py [output dir] [figure dir]   (defaults: output figures)
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else 'output'
FIG = sys.argv[2] if len(sys.argv) > 2 else 'figures'
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': ['Hiragino Sans GB', 'Arial Unicode MS', 'Heiti TC', 'sans-serif'],
    'axes.unicode_minus': False, 'font.size': 10, 'axes.titlesize': 11.5, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#8a8a85', 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.8,
    'xtick.color': '#55544f', 'ytick.color': '#55544f', 'axes.labelcolor': '#33332f',
    'figure.facecolor': 'white', 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})
UP, DOWN, BLUE, ORANGE, VIOLET, GRAY = '#d93a3a', '#0f8a4a', '#2a78d6', '#eb6834', '#4a3aa7', '#9a9993'
NOTE = '数据：2026-09-23 上证50 成分股 Level-2（50 只）。'


def rd(name):
    with open(os.path.join(OUT, name + '.json'), encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))


def footnote(fig, text):
    fig.text(0.01, -0.01, NOTE + text, fontsize=8, color='#77766f', ha='left', va='top')


# ---------- 图1 下跌的时间结构
mm, fl = rd('market_minute'), rd('flow_minute')
times = list(mm.t)
pos = {t: i for i, t in enumerate(times)}
ticks = ['09:30', '10:30', '11:29', '14:00', '15:00']
fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True, gridspec_kw={'height_ratios': [1.1, 1, 1.1], 'hspace': 0.32})
a = axes[0]
a.plot(range(len(mm)), mm.ew_ret_pct, color=BLUE, lw=2)
a.axhline(0, color='#8a8a85', lw=0.8)
lo = mm.ew_ret_pct.idxmin()
a.scatter([lo], [mm.ew_ret_pct[lo]], color=BLUE, zorder=3, s=26)
a.annotate(f'{mm.t[lo]} 最低 {mm.ew_ret_pct[lo]:.2f}%', (lo, mm.ew_ret_pct[lo]), xytext=(-40, -26), textcoords='offset points', fontsize=9, ha='right',
           arrowprops=dict(arrowstyle='-', color='#8a8a85', lw=0.8))
a.set_ylim(-0.72, 0.03)
a.set_title('a  等权组合涨跌（相对昨收，%）')
a = axes[1]
a.plot(range(len(mm)), mm.up_count, color=UP, lw=1.6, label='上涨家数')
a.plot(range(len(mm)), mm.down_count, color=DOWN, lw=1.6, label='下跌家数')
a.set_ylim(0, 50)
a.legend(loc='center right', frameon=False, ncol=2)
a.set_title('b  市场宽度（只）')
a = axes[2]
labels = [('xlarge', '超大单 ≥100万', VIOLET), ('large', '大单 20–100万', ORANGE), ('medium', '中单 4–20万', GRAY), ('small', '小单 <4万', BLUE)]
x = [pos[t] for t in fl.t]
ends = {'xlarge': -3.5, 'large': 1.5, 'medium': 0, 'small': 0}      # nudge end labels apart (亿元)
for k, name, c in labels:
    y = fl[f'cum_{k}_yi']
    a.plot(x, y, color=c, lw=2, label=name)
    a.text(x[-1] + 4, y.iloc[-1] + ends[k], f'{y.iloc[-1]:+.1f}', color=c, fontsize=9, va='center')
a.set_ylim(-34, 8)
a.axhline(0, color='#8a8a85', lw=0.8)
a.legend(loc='lower left', frameon=False, ncol=2, fontsize=9)
a.set_title('c  主动资金分档累计净买入（亿元，连续竞价）')
a.set_xticks([pos[t] for t in ticks], ['09:30', '10:30', '11:30/13:00', '14:00', '15:00'])
a.set_xlim(-2, len(mm) + 16)
fig.suptitle('下跌贯穿全天：路径、宽度与主动资金', x=0.01, ha='left', fontsize=14, fontweight='bold', y=0.96)
footnote(fig, '横轴为交易分钟，午休压缩。主动单＝同一主动方委托号的成交合并，按全天成交额分档。')
fig.savefig(os.path.join(FIG, '图1_下跌的时间结构.png'))
plt.close(fig)

# ---------- 图2 因子 IC 热力图（尾段留出）
ic, comp = rd('factor_ic'), rd('composite_ic')
H = sorted(ic.horizon_s.unique())
fs = ic.drop_duplicates('factor_order').sort_values('factor_order').factor.tolist()
M = np.array([[ic[(ic.factor == f) & (ic.horizon_s == h)].ic_holdout.iloc[0] for h in H] for f in fs])
C = np.array([[comp[comp.horizon_s == h].ic_holdout.iloc[0] for h in H]])
fig, (a, b) = plt.subplots(2, 1, figsize=(7.6, 7.4), gridspec_kw={'height_ratios': [12, 1.15], 'hspace': 0.08})
for ax, mat, rows in [(a, M, fs), (b, C, ['12 因子岭回归组合'])]:
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-0.25, vmax=0.25, aspect='auto')
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f'{0.0 if abs(v) < 5e-4 else v:.3f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(v) > 0.15 else '#2a2a27', fontweight='bold' if ax is b else 'normal')
a.set_xticks([])
b.set_xticks(range(len(H)), [f'{h} 秒' for h in H])
b.set_xlabel('预测周期（未来中间价变化）')
cb = fig.colorbar(im, ax=[a, b], fraction=0.035, pad=0.02)
cb.set_label('逐股 Pearson IC 等权均值')
a.set_title('盘口因子与未来中间价变化的相关（尾段留出 14:00–14:57）', fontsize=12.5, pad=10)
footnote(fig, '3 秒快照计算因子；训练 9:30–11:30，验证 13:00–14:00，尾段留出只用于检验。')
fig.savefig(os.path.join(FIG, '图2_因子IC热力图.png'))
plt.close(fig)

# ---------- 图3 挂单存活时间
lt, ls = rd('lifetime'), rd('lifetime_stats').set_index('board')
fig, a = plt.subplots(figsize=(7.6, 4.6))
for b, c in [('主板', BLUE), ('科创板', ORANGE)]:
    d = lt[lt.board == b]
    a.plot(d.seconds, d.cum_share_pct, color=c, lw=2, marker='o', ms=3.5, label=f'{b}（中位数 {ls.median_s[b]:.1f} 秒）')
a.axhline(50, color='#8a8a85', lw=0.8, ls='--')
a.axvline(10, color='#bdbcb6', lw=0.8)
for b, c, dx, dy in [('主板', BLUE, 8, -14), ('科创板', ORANGE, -92, 4)]:
    v = ls.within_10s_pct[b]
    a.annotate(f'10 秒内 {v:.1f}%', (10, v), xytext=(dx, dy), textcoords='offset points', color=c, fontsize=9)
a.set_xscale('log')
a.set_xticks([1, 10, 60, 600, 3600], ['1 秒', '10 秒', '1 分', '10 分', '1 小时'])
a.set_ylim(0, 100)
a.set_ylabel('已撤销的累计比例（%）')
a.legend(frameon=False, loc='upper left')
a.set_title('被撤销的挂单，从挂出到撤销用了多久')
footnote(fig, f'按原始委托号配对新增与撤单，共 {int(ls.cancels.sum()):,} 笔；扣除午休。')
fig.savefig(os.path.join(FIG, '图3_挂单存活时间.png'))
plt.close(fig)

# ---------- 图4 价差与股价
st = rd('stocks')
tick = st.spread_ticks <= 1.1
fig, a = plt.subplots(figsize=(7.6, 5))
px = np.geomspace(st.close.min() * 0.85, 120, 50)
a.plot(px, 100 / px, color='#8a8a85', lw=1, ls='--')
a.annotate('虚线：价差恰为 1 分钱', (100, 1.0), xytext=(14, -2), textcoords='offset points', color='#77766f', fontsize=9, va='center')
a.scatter(st.close[tick], st.spread_bps[tick], s=34, color=BLUE, edgecolor='white', lw=0.8, label=f'平均价差 ≤1.1 个最小价位（{tick.sum()} 只）', zorder=3)
a.scatter(st.close[~tick], st.spread_bps[~tick], s=34, color=ORANGE, edgecolor='white', lw=0.8, label=f'其他（{(~tick).sum()} 只）', zorder=3)
for n, off in [('华电新能', (8, 2)), ('中国银行', (8, 4)), ('贵州茅台', (-22, -14)), ('兆易创新', (6, 4)), ('寒武纪', (6, 4))]:
    r = st[st.name == n].iloc[0]
    a.annotate(n, (r.close, r.spread_bps), xytext=off, textcoords='offset points', fontsize=8.5, color='#44433f')
a.set_xscale('log')
a.set_yscale('log')
a.set_xticks([2, 5, 10, 20, 50, 100, 200, 500, 1000], ['2', '5', '10', '20', '50', '100', '200', '500', '1000'])
a.set_ylim(0.9, 40)
a.set_yticks([1, 2, 5, 10, 20], ['1', '2', '5', '10', '20'])
a.set_xlabel('收盘价（元，对数刻度）')
a.set_ylabel('平均一档价差（bp，对数刻度）')
a.legend(frameon=False, loc='upper right', fontsize=9)
a.set_title('低价股的价差被 0.01 元最小价位卡住')
footnote(fig, '价差为连续竞价时段各快照一档相对价差的均值。')
fig.savefig(os.path.join(FIG, '图4_价差与股价.png'))
plt.close(fig)
print('saved to', FIG)
