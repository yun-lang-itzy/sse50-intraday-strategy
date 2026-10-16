# -*- coding: utf-8 -*-
"""Shared paths, loaders and reference tables for the SSE 50 Level-2 study."""
import argparse
import json
import os

import numpy as np
import pandas as pd

DATE = '20260923'

# Codes -> short names (the raw files carry only codes).
NAMES = {
    600028: '中国石化', 600030: '中信证券', 600031: '三一重工', 600036: '招商银行', 600050: '中国联通',
    600089: '特变电工', 600111: '北方稀土', 600150: '中国船舶', 600183: '生益科技', 600276: '恒瑞医药',
    600309: '万华化学', 600406: '国电南瑞', 600519: '贵州茅台', 600760: '中航沈飞', 600809: '山西汾酒',
    600887: '伊利股份', 600900: '长江电力', 600930: '华电新能', 601012: '隆基绿能', 601088: '中国神华',
    601127: '赛力斯', 601166: '兴业银行', 601211: '国泰海通', 601288: '农业银行', 601318: '中国平安',
    601328: '交通银行', 601398: '工商银行', 601600: '中国铝业', 601601: '中国太保', 601628: '中国人寿',
    601658: '邮储银行', 601668: '中国建筑', 601688: '华泰证券', 601728: '中国电信', 601857: '中国石油',
    601888: '中国中免', 601899: '紫金矿业', 601919: '中远海控', 601988: '中国银行', 603019: '中科曙光',
    603259: '药明康德', 603501: '豪威集团', 603986: '兆易创新', 603993: '洛阳钼业', 688008: '澜起科技',
    688012: '中微公司', 688041: '海光信息', 688111: '金山办公', 688256: '寒武纪', 688981: '中芯国际',
}

# Shenwan level-1 industry (2021 edition), assigned by hand.
SW1 = {
    600028: '石油石化', 600030: '非银金融', 600031: '机械设备', 600036: '银行', 600050: '通信',
    600089: '电力设备', 600111: '有色金属', 600150: '国防军工', 600183: '电子', 600276: '医药生物',
    600309: '基础化工', 600406: '电力设备', 600519: '食品饮料', 600760: '国防军工', 600809: '食品饮料',
    600887: '食品饮料', 600900: '公用事业', 600930: '公用事业', 601012: '电力设备', 601088: '煤炭',
    601127: '汽车', 601166: '银行', 601211: '非银金融', 601288: '银行', 601318: '非银金融',
    601328: '银行', 601398: '银行', 601600: '有色金属', 601601: '非银金融', 601628: '非银金融',
    601658: '银行', 601668: '建筑装饰', 601688: '非银金融', 601728: '通信', 601857: '石油石化',
    601888: '商贸零售', 601899: '有色金属', 601919: '交通运输', 601988: '银行', 603019: '计算机',
    603259: '医药生物', 603501: '电子', 603986: '电子', 603993: '有色金属', 688008: '电子',
    688012: '电子', 688041: '电子', 688111: '计算机', 688256: '电子', 688981: '电子',
}

# Field codes in the raw files (ASCII values).
BUY, SELL = 66, 83          # 'B', 'S' — exe.BSFlag (aggressor side) and ord.FunctionCode
ADD, DEL = 65, 68           # 'A', 'D' — ord.OrderKind

# Taker-order size buckets by total traded amount of one aggressor order (yuan).
SIZE_BINS = [0, 4e4, 2e5, 1e6, np.inf]
SIZE_LABELS = ['small', 'medium', 'large', 'xlarge']


def parse_args(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('--data-dir', default='data', help='folder holding exe/snp/ord_<date>.parquet')
    p.add_argument('--out-dir', default='output', help='folder for the JSON results')
    p.add_argument('--date', default=DATE)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    return a


def load(args, kind, columns=None):
    """kind: 'exe' (trades), 'snp' (snapshots) or 'ord' (orders)."""
    return pd.read_parquet(os.path.join(args.data_dir, f'{kind}_{args.date}.parquet'), columns=columns)


def to_ms(t):
    """HHMMSSmmm integer -> milliseconds since midnight."""
    t = np.asarray(t).astype(np.int64)
    return (t // 10000000) * 3600000 + (t // 100000 % 100) * 60000 + (t // 1000 % 100) * 1000 + t % 1000


def minute_bucket(t):
    """HHMMSSmmm -> HHMM bucket; 11:30:00 folds into 11:29, the closing call (14:57–15:00) into 15:00."""
    hm = (np.asarray(t) // 100000).astype(int)
    hm = np.where(hm == 1130, 1129, hm)
    return np.where((hm >= 1457) & (hm <= 1500), 1500, hm)


def hm_label(hm):
    return f'{hm // 100:02d}:{hm % 100:02d}'


def continuous(t):
    """Mask for continuous-auction time (09:30–11:30, 13:00–14:57)."""
    t = np.asarray(t)
    return ((t >= 93000000) & (t < 113000000)) | ((t >= 130000000) & (t < 145700000))


def dump(rows, args, name):
    path = os.path.join(args.out_dir, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, separators=(',', ':'), default=float)
    print(f'wrote {path} ({len(rows)} rows)')
