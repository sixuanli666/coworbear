# -*- coding: utf-8 -*-
# 因子综合打分 · Streamlit 一页式应用（云端可用版）
# - 云端优先使用上传；本地仍可走硬盘路径
# - 所有 D:\ 路径均已替换为“上传/示例文件兜底”
# - 需要导出 PNG 时需 kaleido

import os, io, json, re, time
os.environ["PLOTLY_JSON_ENGINE"] = "json"

import numpy as np
import pandas as pd

import plotly.io as pio
pio.json.config.default_engine = "json"
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from datetime import datetime

# ========== 运行环境与配置 ==========
IN_CLOUD = os.environ.get("HOME", "").startswith("/home")

DEFAULT_CFG = {
    "app_mode": "cloud",
    "columns": {
        "date": ["date", "trd_date"],
        "index_candidates": ["clqn_prc", "close", "index", "px"],
        "score_prefix": ["score_", "scoreh", "sum_score", "f4"]
    },
    "modules": {
        "m11_div_vs_10y": True,
        "m12_ep_minus_10y": True,
        "m21_rotation": True,
        "m22_crowding": True,
        "m31_turnover_std": True
    }
}

def read_json_upl_or_local(upl, fallback_path):
    cfg = DEFAULT_CFG.copy()
    try:
        if upl is not None:
            cfg.update(json.load(upl))
        elif os.path.exists(fallback_path):
            with open(fallback_path, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
    except Exception as e:
        st.warning(f"读取配置失败：{e}")
    return cfg

def read_csv_or_demo(uploaded, demo_path, **kwargs):
    if uploaded is not None:
        return pd.read_csv(uploaded, **kwargs)
    if os.path.exists(demo_path):
        return pd.read_csv(demo_path, **kwargs)
    return pd.DataFrame()

def read_excel_or_demo(uploaded, demo_path, **kwargs):
    if uploaded is not None:
        return pd.read_excel(uploaded, **kwargs)
    if os.path.exists(demo_path):
        return pd.read_excel(demo_path, **kwargs)
    return pd.DataFrame()

def to_num(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).replace(',', '').strip()
    try: return float(s)
    except: return np.nan

def zstats(arr: pd.Series):
    s = pd.to_numeric(arr, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0: return 0.0, 1.0
    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    if sd == 0 or np.isnan(sd): sd = 1.0
    return mu, sd

def rolling_slope(y: pd.Series, win: int) -> pd.Series:
    y = pd.to_numeric(y, errors='coerce')
    out = np.full(len(y), np.nan)
    X = np.arange(win)
    Xmean = (win - 1) / 2
    denom = np.sum((X - Xmean) ** 2) or 1e-9
    for i in range(win - 1, len(y)):
        seg = y.iloc[i - win + 1 : i + 1]
        if seg.isna().any(): continue
        Ymean = seg.mean()
        num = np.sum((np.arange(win) - Xmean) * (seg.values - Ymean))
        out[i] = num / denom
    return pd.Series(out, index=y.index)

def find_signals(df: pd.DataFrame, score_col: str, index_col: str,
                 mu: float, sigma: float, k: float = 1.0,
                 slope_win: int = 28, slope_th: float = 0.0):
    score = pd.to_numeric(df[score_col], errors='coerce')
    slope = rolling_slope(score, slope_win)
    hi = mu + k * sigma
    lo = mu - 1.5 * sigma
    buys = (score <= lo) & (slope > slope_th)
    sells = (score >= hi) & (slope < -slope_th)
    out = df.loc[buys | sells, ['date', score_col, index_col]].copy()
    out['type'] = np.where(buys.loc[out.index], '强买', '强卖')
    out.rename(columns={score_col: 'score', index_col: 'index_px'}, inplace=True)
    return out

def make_main_figure(df: pd.DataFrame, score_col: str, index_col: str,
                     show_bands: bool, mu: float, sigma: float, k: float,
                     signals: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df[score_col], mode='lines', name=score_col, yaxis='y1'))
    fig.add_trace(go.Scatter(x=df['date'], y=df[index_col], mode='lines', name=index_col, yaxis='y2'))
    if show_bands:
        fig.add_trace(go.Scatter(x=df['date'], y=[mu]*len(df), mode='lines', name='均值',
                                 line=dict(dash='dot'), yaxis='y1'))
        fig.add_trace(go.Scatter(x=df['date'], y=[mu + k*sigma]*len(df), mode='lines', name=f'+{k}σ',
                                 line=dict(dash='dash'), yaxis='y1'))
        fig.add_trace(go.Scatter(x=df['date'], y=[mu - 1.0*sigma]*len(df), mode='lines', name='-1σ',
                                 line=dict(dash='dash'), yaxis='y1'))
    if signals is not None and not signals.empty:
        buys = signals[signals['type'] == '强买']
        sells = signals[signals['type'] == '强卖']
        if len(buys):
            fig.add_trace(go.Scatter(x=buys['date'], y=buys['index_px'], mode='markers', name='强买',
                                     marker=dict(symbol='triangle-up', size=10), yaxis='y2'))
        if len(sells):
            fig.add_trace(go.Scatter(x=sells['date'], y=sells['index_px'], mode='markers', name='强卖',
                                     marker=dict(symbol='triangle-down', size=10), yaxis='y2'))
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=60, r=70, t=40, b=40),
        legend=dict(orientation='h', x=0, y=1.12),
        xaxis=dict(title='日期'),
        yaxis=dict(title=score_col, side='left'),
        yaxis2=dict(title='指数', side='right', overlaying='y'),
        height=560,
    )
    return fig

def _fmt_range_text(start_str: str, end_str: str) -> str:
    s = (start_str or "").strip(); e = (end_str or "").strip()
    if s and e: return f"{s} ~ {e}"
    if s and not e: return f"{s} ~ 最新"
    if (not s) and e: return f"截至 {e}"
    return "全区间"

def _parse_dt(s: str):
    s = (s or "").strip()
    if not s: return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try: return pd.to_datetime(s, format=fmt)
        except: pass
    return None

# ========== 页面信息 ==========
st.set_page_config(page_title='因子综合打分 · 交互可视化', layout='wide')
st.title('因子综合打分交互可视化')
st.caption('云端建议：通过侧边栏上传数据文件；本地可使用硬盘路径。')

# ========== 侧边栏：总配置与上传 ==========
with st.sidebar:
    st.header("数据与配置")
    cfg_upl = st.file_uploader("上传配置 config_info.json（可选）", type=["json"])
cfg = read_json_upl_or_local(cfg_upl, "config_info.json")

# ========== 主数据：分数 × 指数（主图） ==========
with st.sidebar:
    st.header('主数据（分数×指数）')
    # 云端默认上传；本地可给路径
    default_path = '' if IN_CLOUD else 'D:/projects/权益投资部模型库/分数/因子综合打分_合并版本.csv'
    use_local = st.toggle('使用本地路径', value=bool(default_path and not IN_CLOUD),
                          help='云端请关闭本开关并上传CSV')
    main_csv_upl = None
    if not use_local:
        main_csv_upl = st.file_uploader('上传 主CSV（含 date/trd_date、指数列(clqn_prc/close/index/px)、score_*）',
                                        type=['csv'], key="main_csv")
    start_date = st.text_input('起始日期(YYYYMMDD，可空)', value='', key='main_start')
    sigma_k = st.number_input('强弱带宽 K（σ倍数）', min_value=0.1, max_value=3.0, value=1.0, step=0.1, key='main_k')
    show_bands = st.checkbox('显示均值与 ±σ 带', value=True, key='main_bands')
    use_quadrant = st.checkbox('启用强买强卖信号', value=True, key='main_quad')

# 读主CSV
try:
    if not use_local:
        if main_csv_upl is None:
            st.info('请上传主CSV，或启用“使用本地路径”。'); st.stop()
        df0 = pd.read_csv(main_csv_upl)
    else:
        if (not default_path) or (not os.path.exists(default_path)):
            st.error(f'找不到文件：{default_path}。请关闭“使用本地路径”并上传 CSV。'); st.stop()
        try:
            df0 = pd.read_csv(default_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df0 = pd.read_csv(default_path, encoding='gbk')
except Exception as e:
    st.error(f'读取 CSV 失败：{e}')
    st.stop()

# 规范列
_df = df0.copy()
if 'date' not in _df.columns:
    d_alt = [c for c in _df.columns if c.lower() == 'trd_date']
    if d_alt:
        _df['date'] = pd.to_datetime(_df[d_alt[0]], errors='coerce')
    else:
        st.error('未找到日期列（需要 date 或 trd_date）'); st.stop()
else:
    _df['date'] = pd.to_datetime(_df['date'], errors='coerce')
_df = _df.sort_values('date').dropna(subset=['date']).reset_index(drop=True)

# 指数候选
index_candidates = [c for c in _df.columns if c in cfg["columns"]["index_candidates"]]
if not index_candidates:
    numeric_candidates = [c for c in _df.columns if c not in ['date']]
    index_candidates = numeric_candidates[:1]
# 分数候选
prefixes = tuple(cfg["columns"]["score_prefix"])
score_candidates = [c for c in _df.columns if c.startswith(prefixes)]
if not score_candidates:
    score_candidates = [c for c in _df.columns if c.startswith('f4')]

with st.sidebar:
    score_col = st.selectbox('分数列', score_candidates, index=0 if score_candidates else None, key='main_score')
    index_col = st.selectbox('指数列', index_candidates, index=0 if index_candidates else None, key='main_index')

# 过滤日期
if start_date:
    sdt = _parse_dt(start_date)
    if sdt is None:
        st.warning("起始日期格式应为 YYYYMMDD 或 YYYY-MM-DD，已忽略。")
    else:
        _df = _df[_df['date'] >= sdt]

# 数值化
_df[index_col] = _df[index_col].map(to_num)
_df[score_col] = _df[score_col].map(to_num)

# 统计与信号
mu, sigma = zstats(_df[score_col])
sig_df = find_signals(_df, score_col, index_col, mu, sigma, k=sigma_k, slope_win=28, slope_th=0.0) if use_quadrant \
         else pd.DataFrame(columns=['date', 'score', 'index_px', 'type'])

# 主图
fig = make_main_figure(_df, score_col, index_col, show_bands, mu, sigma, sigma_k, sig_df)
st.plotly_chart(fig, use_container_width=True)

with st.expander('导出主图 / 数据'):
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button('下载主图 PNG'):
            try:
                _ = __import__("kaleido")  # 触发 ImportError 提示
                png_bytes = fig.to_image(format='png', scale=2)
                st.download_button('点击保存 PNG', data=png_bytes, file_name='score_vs_index.png', mime='image/png')
            except Exception as e:
                st.warning(f'导出失败（可能未安装 kaleido）：{e}')
    with colB:
        st.download_button('下载强买强卖信号 CSV', data=sig_df.to_csv(index=False).encode('utf-8-sig'),
                           file_name='signals.csv', mime='text/csv')
    with colC:
        st.download_button('下载筛选后数据 CSV', data=_df.to_csv(index=False).encode('utf-8-sig'),
                           file_name='filtered_data.csv', mime='text/csv')

st.subheader('强买强卖信号集合')
if not sig_df.empty:
    st.dataframe(sig_df.tail(12).iloc[::-1].reset_index(drop=True))
else:
    st.info('暂无信号。')

# ========== 单因子分数图（改为上传/示例兜底） ==========
st.subheader('单因子分数图（上传图片目录的单张 PNG/JPG）')
with st.sidebar:
    st.header('单因子图 · 资源')
    imgs_zip = st.file_uploader("（可选）上传打包图片ZIP（f41_*.png等）", type=['zip'], key='fac_zip')
    # 也允许单独多文件上传
    imgs_multi = st.file_uploader("或多选上传图片（png/jpg/jpeg）", type=['png','jpg','jpeg'], accept_multiple_files=True, key='fac_multi')

fac_desc = {
    "f41": "成交量相对地量倍数 × 价格高低位，高位放量记 +1，低位缩量记 -1。",
    "f42": "振幅相对半年基线显著扩张且价格动量强记 +1，否则 -1。",
    "f43": "散户分位走高、机构分位走低记 +1，反之 -1。",
    "f44": "公募基金仓位高分位记 +1，低分位记 -1。",
    "f45": "价格在高位区且北向波动放大记 +1，否则 -1。",
    "f47": "创新高占比趋势转弱记 +1，未转弱记 -1。",
    "f48": "PE反弹≥40%且动量转弱首次出现记 +1，其余 -1。",
    "f49": "次新板块波动高分位（或抬升）记 +1，低分位记 -1。",
    "f411": "产业资本净减持强度高分位且趋势走强记 +1，其余 -1。"
}

def extract_zip_images(file):
    from zipfile import ZipFile
    files = []
    try:
        with ZipFile(file) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith(('.png','.jpg','.jpeg')):
                    files.append((info.filename, zf.read(info)))
    except Exception as e:
        st.warning(f"解压失败：{e}")
    return files

img_entries = []  # [(name, bytes)]
if imgs_zip is not None:
    img_entries.extend(extract_zip_images(imgs_zip))
if imgs_multi:
    for up in imgs_multi:
        img_entries.append((up.name, up.getvalue()))

if img_entries:
    custom_order = ["f41", "f42", "f43", "f44", "f45", "f47", "f48", "f49", "f411"]
    order_rank = {fac: i for i, fac in enumerate(custom_order)}
    def get_fac_key(fname: str):
        return os.path.splitext(os.path.basename(fname))[0].split("_")[0]
    img_entries = sorted(img_entries, key=lambda it: order_rank.get(get_fac_key(it[0]), 10_000))
    per_row = 3
    for i in range(0, len(img_entries), per_row):
        cols = st.columns(per_row)
        for j, (nm, data) in enumerate(img_entries[i:i+per_row]):
            base_name = os.path.splitext(os.path.basename(nm))[0]
            fac_key = base_name.split("_")[0]
            desc_text = fac_desc.get(fac_key, "")
            with cols[j]:
                st.markdown(f"**{base_name}**")
                if desc_text: st.caption(desc_text)
                try:
                    st.image(Image.open(io.BytesIO(data)))
                except Exception as e:
                    st.error(f"加载图片失败: {nm}\n错误: {e}")
else:
    st.caption("（可上传 ZIP 或多选图片进行展示）")

# ========== 1.1 300股息率/10Y × 上证综指 ==========
st.markdown("---")
st.subheader("1.1 300指数股息率 / 十年国债 × 上证综指（右轴）")
with st.expander("指标说明", expanded=False):
    st.write("上传 1.1 结果CSV（weighted_dividend_rate）、十年国债CSV（nation10_yield）与可选的上证综指CSV（sh_close）。")

with st.sidebar:
    st.header("1.1 参数")
    div_csv_upl = st.file_uploader("上传 1.1 结果CSV（trade_date, weighted_dividend_rate）", type=['csv'], key="div_upl_csv")
    teny_csv_upl = st.file_uploader("上传 十年国债CSV（trade_date, nation10_yield）", type=['csv'], key="div_upl_10y")
    sh_csv_upl   = st.file_uploader("上传 上证综指CSV（trade_date, sh_close）(可选)", type=['csv'], key="div_upl_sh")
    show_bands_11 = st.checkbox("显示均值与±1σ", value=True, key="div_bands")
btn_11 = st.button("生成 1.1 图表", type="primary", key="div_btn")
if btn_11:
    try:
        base = read_csv_or_demo(div_csv_upl, "assets/demo_div.csv", dtype={'trade_date': str})
        if base.empty or 'weighted_dividend_rate' not in base.columns:
            st.error("请上传含 weighted_dividend_rate 的CSV。"); st.stop()
        base['trade_date'] = pd.to_datetime(base['trade_date'], errors='coerce')
        base = base.dropna(subset=['trade_date']).drop_duplicates('trade_date', keep='last').sort_values('trade_date')

        ten_y = read_csv_or_demo(teny_csv_upl, "assets/demo_10y.csv")
        if ten_y.empty or ('trade_date' not in ten_y.columns) or ('nation10_yield' not in ten_y.columns):
            st.error("请上传 10Y CSV（trade_date, nation10_yield）。"); st.stop()
        ten_y['trade_date'] = pd.to_datetime(ten_y['trade_date'], errors='coerce')
        ten_y['nation10_yield'] = pd.to_numeric(ten_y['nation10_yield'], errors='coerce')
        ten_y = ten_y.dropna(subset=['trade_date','nation10_yield']).drop_duplicates('trade_date', keep='last').sort_values('trade_date')

        df = base.merge(ten_y, on='trade_date', how='left').sort_values('trade_date')
        df['nation10_yield'] = df['nation10_yield'].ffill()
        df['ratio'] = df['weighted_dividend_rate'] / df['nation10_yield']

        mu = float(pd.to_numeric(df['ratio'], errors='coerce').mean())
        sd = float(pd.to_numeric(df['ratio'], errors='coerce').std(ddof=1))

        fig11 = go.Figure()
        fig11.add_trace(go.Scatter(x=df['trade_date'], y=df['ratio'],
                                   mode='lines', name='300股息率/10Y(%)', yaxis='y1'))
        if show_bands_11:
            fig11.add_trace(go.Scatter(x=df['trade_date'], y=[mu]*len(df), mode='lines', name='均值',
                                       line=dict(dash='dot'), yaxis='y1'))
            fig11.add_trace(go.Scatter(x=df['trade_date'], y=[mu+sd]*len(df), mode='lines', name='均值+1σ',
                                       line=dict(dash='dash'), yaxis='y1'))
            fig11.add_trace(go.Scatter(x=df['trade_date'], y=[mu-sd]*len(df), mode='lines', name='均值-1σ',
                                       line=dict(dash='dash'), yaxis='y1'))

        sh = read_csv_or_demo(sh_csv_upl, "assets/demo_sh.csv")
        if not sh.empty and {'trade_date','sh_close'}.issubset(sh.columns):
            sh['trade_date'] = pd.to_datetime(sh['trade_date'], errors='coerce')
            sh['sh_close'] = pd.to_numeric(sh['sh_close'], errors='coerce')
            sh = sh.dropna().drop_duplicates('trade_date', keep='last').sort_values('trade_date')
            fig11.add_trace(go.Scatter(x=sh['trade_date'], y=sh['sh_close'], mode='lines', name='上证综指', yaxis='y2'))

        fig11.update_layout(
            template='plotly_dark', height=560,
            legend=dict(orientation='h', x=0, y=1.12),
            margin=dict(l=60, r=70, t=40, b=40),
            xaxis=dict(title='日期'),
            yaxis=dict(title='股息率/10Y(%)', autorange='reversed'),
            yaxis2=dict(title='上证综指', side='right', overlaying='y'),
        )
        st.plotly_chart(fig11, use_container_width=True)

        with st.expander("下载当前视图数据"):
            st.download_button("下载CSV", data=df.to_csv(index=False).encode('utf-8-sig'),
                               file_name='dividend_ratio_10y_merged.csv', mime='text/csv')
    except Exception as e:
        st.error(f"生成图表失败：{e}")

# ========== 1.2 全A E/P − 10Y ==========
st.markdown("---")
st.subheader("1.2 全A E/P(市盈率倒数) − 十年期国债")
with st.expander("指标说明", expanded=False):
    st.write("上传 E/P−10Y 结果CSV（weighted_ep_10bond，小数），可选择去极值与带状显示。")

with st.sidebar:
    st.header("1.2 参数")
    ep_csv_upl = st.file_uploader("上传 E/P−10Y 结果CSV（trade_date, weighted_ep_10bond）", type=['csv'], key="ep_path")
    ep_start = st.text_input("起始日(YYYYMMDD，可空)", value="", key="ep_start")
    ep_end   = st.text_input("结束日(YYYYMMDD，可空)", value="", key="ep_end")
    ep_clip  = st.checkbox("1%/99% 去极值", value=True, key="ep_clip")
    ep_bands = st.checkbox("显示均值与±1σ", value=True, key="ep_bands")

@st.cache_data(show_spinner=False)
def load_ep10_csv_from_upl(upl, demo="assets/demo_ep10.csv"):
    df = read_csv_or_demo(upl, demo)
    if df.empty: return df
    if "trade_date" not in df.columns: raise ValueError("缺 trade_date 列")
    if "weighted_ep_10bond" not in df.columns: raise ValueError("缺 weighted_ep_10bond 列")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    df["weighted_ep_10bond"] = pd.to_numeric(df["weighted_ep_10bond"], errors="coerce")
    return df

btn_ep = st.button("生成 1.2 图表", type="primary", key="ep_btn")
if btn_ep:
    try:
        epdf = load_ep10_csv_from_upl(ep_csv_upl)
        if epdf.empty: st.error("请上传 E/P−10Y CSV。"); st.stop()
        if ep_start:
            try: epdf = epdf[epdf["trade_date"] >= pd.to_datetime(ep_start, format="%Y%m%d")]
            except: st.warning("起始日格式应为 YYYYMMDD，已忽略。")
        if ep_end:
            try: epdf = epdf[epdf["trade_date"] <= pd.to_datetime(ep_end,   format="%Y%m%d")]
            except: st.warning("结束日格式应为 YYYYMMDD，已忽略。")
        if epdf.empty: st.warning("过滤后无数据。"); st.stop()
        s = pd.to_numeric(epdf["weighted_ep_10bond"], errors="coerce")
        if ep_clip:
            lo, hi = s.quantile([0.01, 0.99]); s = s.clip(lo, hi)
        mu = float(s.mean()); sd = float(s.std(ddof=1))
        fig12 = go.Figure()
        fig12.add_trace(go.Scatter(x=epdf["trade_date"], y=s, mode="lines", name="E/P − 10Y（小数）"))
        if ep_bands:
            fig12.add_trace(go.Scatter(x=epdf["trade_date"], y=[mu]*len(epdf), mode="lines", name="均值", line=dict(dash="dot")))
            fig12.add_trace(go.Scatter(x=epdf["trade_date"], y=[mu+sd]*len(epdf), mode="lines", name="均值+1σ", line=dict(dash="dash")))
            fig12.add_trace(go.Scatter(x=epdf["trade_date"], y=[mu-sd]*len(epdf), mode="lines", name="均值-1σ", line=dict(dash="dash")))
        fig12.update_layout(template="plotly_dark", height=520, legend=dict(orientation="h", x=0, y=1.12),
                            margin=dict(l=60, r=40, t=40, b=40), xaxis=dict(title="日期"), yaxis=dict(title="E/P − 10Y（小数）"))
        st.plotly_chart(fig12, use_container_width=True)
        with st.expander("下载当前视图数据"):
            out_df = epdf.copy(); out_df["weighted_ep_10bond_clean"] = s.values
            st.download_button("下载CSV", data=out_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="ep_minus_10y_clean.csv", mime="text/csv")
    except Exception as e:
        st.error(f"生成失败：{e}")

# ========== 2.1 大小盘轮动 ==========
st.markdown("---")
st.subheader("2.1 大小盘轮动")
with st.expander("指标说明", expanded=False):
    st.write("上传日/月/季 CSV；若只传日度，系统可聚合为月/季。")

with st.sidebar:
    st.header("2.1 参数")
    rot_daily_upl   = st.file_uploader("日度CSV", type=['csv'], key="rot_daily")
    rot_monthly_upl = st.file_uploader("月度CSV（可空）", type=['csv'], key="rot_month")
    rot_quarter_upl = st.file_uploader("季度CSV（可空）", type=['csv'], key="rot_quarter")
    freq = st.radio("频率", ["日度", "月度", "季度"], index=1, key="rot_freq")
    view = st.radio("指标", ["收益", "净值(NAV)"], index=0, key="rot_view")
    k_sigma = st.number_input("±σ 带宽（仅收益）", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key="rot_k")
    show_band = st.checkbox("显示均值与±σ（仅收益）", value=True, key="rot_band")
    date_start = st.text_input("起始日(YYYYMMDD，可空)", value="", key="rot_start")
    date_end   = st.text_input("结束日(YYYYMMDD，可空)", value="", key="rot_end")

@st.cache_data(show_spinner=False)
def _load_df(upl, idx_name: str, demo=None):
    df = read_csv_or_demo(upl, demo) if upl is not None or demo else (pd.DataFrame())
    if df.empty: return df
    if idx_name in df.columns:
        dt = pd.to_datetime(df[idx_name], errors="coerce")
    else:
        dt = pd.to_datetime(df.iloc[:,0], errors="coerce")
    df.insert(0, "dt", dt)
    df = df.dropna(subset=["dt"]).drop(columns=[c for c in ["date","month","quarter"] if c in df.columns], errors="ignore")
    df = df.set_index("dt").sort_index()
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def _agg_compound(ret_df: pd.DataFrame, rule: str):
    cols_ret = [c for c in ret_df.columns if not c.endswith("_NAV")]
    if not cols_ret: return pd.DataFrame(), pd.DataFrame()
    ret = ret_df[cols_ret].resample(rule).apply(lambda x: (1 + x).prod() - 1)
    nav = (1 + ret).cumprod(); nav.columns = [c + "_NAV" for c in nav.columns]
    return ret, nav

def _clip_by_date(df: pd.DataFrame, start_str: str, end_str: str):
    out = df.copy()
    sdt = _parse_dt(start_str); edt = _parse_dt(end_str)
    if sdt is not None: out = out[out.index >= sdt]
    if edt is not None: out = out[out.index <= edt]
    return out

def _mean_std(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
    if s2.empty: return 0.0, 1.0
    return float(s2.mean()), float(s2.std(ddof=1)) or 1.0

daily_df   = _load_df(rot_daily_upl, "date", demo="assets/demo_rot_daily.csv")
month_df   = _load_df(rot_monthly_upl, "month", demo="assets/demo_rot_month.csv")
quarter_df = _load_df(rot_quarter_upl, "quarter", demo="assets/demo_rot_quarter.csv")

source_df = {"日度": daily_df, "月度": month_df, "季度": quarter_df}[freq]
if (source_df is None) or source_df.empty:
    if (freq in ["月度","季度"]) and (daily_df is not None) and (not daily_df.empty):
        agg_rule = "M" if freq == "月度" else "Q"
        ret_agg, nav_agg = _agg_compound(daily_df, agg_rule)
        source_df = pd.concat([ret_agg, nav_agg], axis=1)
    else:
        st.info(f"未检测到 {freq} 数据。请上传相应CSV或提供日度数据以回退聚合。")
else:
    spread_cols = [c for c in source_df.columns if not c.endswith("_NAV")]
    nav_cols    = [c for c in source_df.columns if c.endswith("_NAV")]
    default_choices = spread_cols[:2] if view == "收益" else nav_cols[:2]
    choices = st.multiselect(f"选择要展示的{'收益' if view=='收益' else '净值'}组合", options=(spread_cols if view=="收益" else nav_cols),
                             default=default_choices, key="rot_chosen")
    if not choices:
        st.info("请选择至少一个组合列。")
    else:
        plot_df = _clip_by_date(source_df[choices], date_start, date_end)
        if plot_df.empty:
            st.warning("时间过滤后无数据。")
        else:
            fig21 = go.Figure()
            for c in choices:
                fig21.add_trace(go.Scatter(x=plot_df.index, y=plot_df[c], mode="lines", name=c, yaxis="y1"))
            if (view == "收益") and show_band and (len(choices) == 1):
                s = plot_df[choices[0]]; mu, sd = _mean_std(s)
                fig21.add_trace(go.Scatter(x=plot_df.index, y=[mu]*len(plot_df), mode="lines", name="均值", line=dict(dash="dot"), yaxis="y1"))
                fig21.add_trace(go.Scatter(x=plot_df.index, y=[mu + k_sigma*sd]*len(plot_df), mode="lines", name=f"+{k_sigma}σ", line=dict(dash="dash"), yaxis="y1"))
                fig21.add_trace(go.Scatter(x=plot_df.index, y=[mu - k_sigma*sd]*len(plot_df), mode="lines", name=f"-{k_sigma}σ", line=dict(dash="dash"), yaxis="y1"))
            fig21.update_layout(template="plotly_dark", height=560, legend=dict(orientation="h", x=0, y=1.12),
                                margin=dict(l=60, r=70, t=40, b=40), xaxis=dict(title="日期"),
                                yaxis=dict(title=("收益" if view=="收益" else "净值"), side="left"))
            st.plotly_chart(fig21, use_container_width=True)
            with st.expander("下载当前视图数据"):
                st.download_button("下载CSV", data=plot_df.reset_index().to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"大小盘_{freq}_{'收益' if view=='收益' else '净值'}.csv", mime="text/csv")

# ========== 2.2 行业拥挤度（展示离线产物） ==========
st.markdown("---")
st.subheader("2.2 行业拥挤度 / TMT 拥挤度 / 行业热度榜")
with st.expander("指标说明", expanded=False):
    st.write("上传离线导出的 PNG/Excel；页面仅展示与下载，不在线重算。")

with st.sidebar:
    st.header("2.2 参数与上传")
    crowd_start = st.text_input("起始日(YYYYMMDD，可空)", value="", key="crowd_start")
    crowd_end   = st.text_input("结束日(YYYYMMDD，可空)", value="", key="crowd_end")
    img1_upl = st.file_uploader("图1 PNG（行业成交额占比）", type=['png'], key="upl_img1")
    img2_upl = st.file_uploader("图2 PNG（TMT拥挤度）", type=['png'], key="upl_img2")
    xlsx_upl = st.file_uploader("行业热度榜 Excel", type=['xlsx'], key="upl_xlsx")

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("图1：行业成交额占比")
    st.caption(f"区间：{_fmt_range_text(crowd_start, crowd_end)}")
    if img1_upl is not None:
        img = Image.open(img1_upl); st.image(img)
        st.download_button("下载图1 PNG", data=img1_upl.getvalue(), file_name="2.2_行业成交占比.png", mime="image/png")
    else:
        st.info("请上传 图1 PNG")
with col_right:
    st.markdown("**图2：TMT拥挤度（MA5）**")
    st.caption(f"区间：{_fmt_range_text(crowd_start, crowd_end)}")
    if img2_upl is not None:
        img = Image.open(img2_upl); st.image(img)
        st.download_button("下载图2 PNG", data=img2_upl.getvalue(), file_name="2.2_TMT拥挤度.png", mime="image/png")
    else:
        st.info("请上传 图2 PNG")

st.markdown("**表3：行业热度榜（区间最后一周截面）**")
st.caption(f"区间：{_fmt_range_text(crowd_start, crowd_end)}")
if xlsx_upl is not None:
    try:
        df_table = pd.read_excel(xlsx_upl)
        st.dataframe(df_table)
        st.download_button("下载行业周度概览 (Excel)", data=xlsx_upl.getvalue(),
                           file_name="2.2_行业热度榜.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.warning(f"无法读取表3: {e}")
else:
    st.info("请上传 行业热度榜 Excel")

# ========== 3.1 换手率横截面标准差（全A + 行业） ==========
st.markdown("---")
st.subheader("3.1 换手率横截面标准差（全A + 行业）")
with st.expander("指标说明", expanded=False):
    st.write("上传 `3.1 turn_daily_std_全量.csv`（trade_date/scope/level1_industry_name/turn_daily_std）。")

with st.sidebar:
    st.header("3.1 参数")
    turnstd_csv_upl = st.file_uploader("上传 3.1 全量CSV", type=['csv'], key="std31_csv")
    dt_start_31 = st.text_input("起始日(YYYYMMDD，可空)", value="", key="std31_start")
    dt_end_31   = st.text_input("结束日(YYYYMMDD，可空)", value="", key="std31_end")
    agg_rule_31   = st.selectbox("聚合", ["不聚合(逐日)", "月度", "季度"], index=0, key="std31_agg")
    smooth_win_31 = st.number_input("平滑窗口（移动平均，期）", min_value=1, max_value=60, value=3, step=1, key="std31_smooth")
    unit_view_31  = st.selectbox("展示单位", ["小数", "百分比(%)"], index=0, key="std31_unit")

@st.cache_data(show_spinner=False)
def load_turn_std_csv_from_upl(upl, demo="assets/demo_turnstd.csv"):
    df = read_csv_or_demo(upl, demo)
    if df.empty: return df
    need_cols = {"trade_date", "scope", "level1_industry_name", "turn_daily_std"}
    miss = need_cols - set(df.columns)
    if miss: raise ValueError(f"CSV 缺少必要列：{miss}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
    df["turn_daily_std"] = pd.to_numeric(df["turn_daily_std"], errors="coerce")
    return df

def _clip_range(df: pd.DataFrame, s_dt, e_dt) -> pd.DataFrame:
    out = df.copy()
    if s_dt is not None: out = out[out["trade_date"] >= s_dt]
    if e_dt is not None: out = out[out["trade_date"] <= e_dt]
    return out

def _agg_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "不聚合(逐日)": return df
    rule_map = {"月度": "M", "季度": "Q"}
    res_rule = rule_map.get(rule)
    if not res_rule: return df
    def _agg_one(g):
        g = g.set_index("trade_date").sort_index()
        out = g.resample(res_rule).mean(numeric_only=True).reset_index()
        for c in ["scope", "level1_industry_name"]:
            out[c] = g[c].iloc[-1] if not g.empty else None
        return out[["trade_date", "scope", "level1_industry_name", "turn_daily_std"]]
    parts = []
    for _, g in df.groupby(["scope", "level1_industry_name"], dropna=False):
        parts.append(_agg_one(g))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]

df31_raw = load_turn_std_csv_from_upl(turnstd_csv_upl)
if df31_raw.empty:
    st.info("请上传 3.1 全量CSV。")
else:
    s_dt = _parse_dt(dt_start_31); e_dt = _parse_dt(dt_end_31)
    df31 = _clip_range(df31_raw, s_dt, e_dt)
    df31 = _agg_df(df31, agg_rule_31)

    if unit_view_31 == "百分比(%)":
        df31["turn_std_view"] = df31["turn_daily_std"] * 100.0
        y_label = "turn_daily_std(%)"
    else:
        df31["turn_std_view"] = df31["turn_daily_std"]
        y_label = "turn_daily_std"

    industries_all = sorted(df31.loc[df31["scope"]=="行业","level1_industry_name"].dropna().unique().tolist())
    has_allA = (df31["scope"]=="全A").any()
    series_all = (["全A"] if has_allA else []) + industries_all

    st.caption(f"区间：{_fmt_range_text(dt_start_31, dt_end_31)} ｜ 频率：{agg_rule_31} ｜ 平滑窗口：{smooth_win_31}")
    chosen_series = st.multiselect("选择系列（行业或全A，可多选）", options=series_all,
                                   default=series_all[:6] if series_all else [], key="std31_series")
    if not chosen_series:
        st.info("请选择至少一个系列。")
    else:
        fig31 = go.Figure(); export_parts = []
        for name in chosen_series:
            if name == "全A":
                g = df31[df31["scope"]=="全A"].sort_values("trade_date")
            else:
                g = df31[(df31["scope"]=="行业") & (df31["level1_industry_name"]==name)].sort_values("trade_date")
            if g.empty: continue
            y = g["turn_std_view"].rolling(window=smooth_win_31, min_periods=1).mean()
            fig31.add_trace(go.Scatter(x=g["trade_date"], y=y, mode="lines", name=str(name)))
            tmp = g[["trade_date","turn_std_view"]].copy(); tmp["series"] = name; export_parts.append(tmp)

        fig31.update_layout(template="plotly_dark", height=560, legend=dict(orientation="h", x=0, y=1.12),
                            margin=dict(l=60, r=40, t=40, b=40), xaxis=dict(title="日期"), yaxis=dict(title=y_label))
        st.plotly_chart(fig31, use_container_width=True)

        if export_parts:
            export_df = pd.concat(export_parts, ignore_index=True).rename(columns={"turn_std_view": "turnover_std_view"})
            st.download_button("下载当前筛选数据（CSV）",
                               data=export_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="3.1_turnover_std_selected.csv",
                               mime="text/csv")
