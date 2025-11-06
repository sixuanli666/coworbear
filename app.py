# ------------------------------
# 因子综合打分 · Streamlit 一页式交互应用
# 说明：
#   - 运行：
#       pip install -r requirements.txt
#       streamlit run app.py
#   - 本应用可本地读取你导出的合并 CSV（例如：D:/projects/权益投资部模型库/分数/因子综合打分_合并版本.csv），
#     也支持页面内文件上传；并可选择分数列与指数列，计算四象限信号并作图。
#   - 若需要导出主图 PNG，需要安装 kaleido。
# ------------------------------
import os
os.environ["PLOTLY_JSON_ENGINE"] = "json"

import io
import numpy as np
import pandas as pd

import os
os.environ["PLOTLY_JSON_ENGINE"] = "json"

import plotly.io as pio
pio.json.config.default_engine = "json"  # ← 再保险一层

import plotly.graph_objects as go
import streamlit as st

from pathlib import Path
import json

def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()


def get_path(key):
    """从 config.json 读取路径"""
    return CONFIG["paths"].get(key, "")


# ========== 工具函数 ==========

def to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).replace(',', '').strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def zstats(arr: pd.Series):
    s = pd.to_numeric(arr, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return 0.0, 1.0
    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    if sd == 0 or np.isnan(sd):
        sd = 1.0
    return mu, sd


def rolling_slope(y: pd.Series, win: int) -> pd.Series:
    y = pd.to_numeric(y, errors='coerce')
    out = np.full(len(y), np.nan)
    X = np.arange(win)
    Xmean = (win - 1) / 2
    denom = np.sum((X - Xmean) ** 2) or 1e-9
    for i in range(win - 1, len(y)):
        seg = y.iloc[i - win + 1 : i + 1]
        if seg.isna().any():
            continue
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
    lo = mu - 1.5 * sigma  # 与你脚本保持一致

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

    # 分数（左轴）
    fig.add_trace(go.Scatter(
        x=df['date'], y=df[score_col], mode='lines', name=score_col,
        yaxis='y1'))

    # 指数（右轴）
    fig.add_trace(go.Scatter(
        x=df['date'], y=df[index_col], mode='lines', name=index_col,
        yaxis='y2'))

    # 均值与带
    if show_bands:
        fig.add_trace(go.Scatter(
            x=df['date'], y=[mu]*len(df), mode='lines', name='均值',
            line=dict(dash='dot'), yaxis='y1'))
        fig.add_trace(go.Scatter(
            x=df['date'], y=[mu + k*sigma]*len(df), mode='lines', name=f'+{k}σ',
            line=dict(dash='dash'), yaxis='y1'))
        fig.add_trace(go.Scatter(
            x=df['date'], y=[mu - 1.0*sigma]*len(df), mode='lines', name='-1σ',
            line=dict(dash='dash'), yaxis='y1'))

    # 信号点画在指数轴上
    if not signals.empty:
        buys = signals[signals['type'] == '强买']
        sells = signals[signals['type'] == '强卖']
        if len(buys):
            fig.add_trace(go.Scatter(
                x=buys['date'], y=buys['index_px'], mode='markers', name='强买',
                marker=dict(symbol='triangle-up', size=10), yaxis='y2'))
        if len(sells):
            fig.add_trace(go.Scatter(
                x=sells['date'], y=sells['index_px'], mode='markers', name='强卖',
                marker=dict(symbol='triangle-down', size=10), yaxis='y2'))

    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=60, r=70, t=40, b=40),
        legend=dict(orientation='h', x=0, y=1.12),
        xaxis=dict(title='日期'),
        yaxis=dict(title=score_col, side='left'),
        yaxis2=dict(title=index_col, side='right', overlaying='y'),
        height=560,
    )
    return fig

from pathlib import Path
APP_DIR = Path(__file__).parent

def resolve_first_existing(p: str) -> Path | None:
    if not p:
        return None
    cands = []
    P = Path(p)

    # 1) 原样 / 展开 ~
    cands += [P, P.expanduser()]
    # 2) 以 cwd 为基准
    cands += [Path(os.getcwd()) / P, (Path(os.getcwd()) / P).expanduser()]
    # 3) 以脚本目录为基准（**关键**：assets 多半跟 app.py 放一起）
    cands += [APP_DIR / P, (APP_DIR / P).expanduser()]

    for c in cands:
        try:
            if c.exists():
                return c
        except:
            pass
    return None

# ========== Streamlit UI ==========
# —— 避免 ep_path 为空导致 value 被忽略 —— 
if "ep_path" not in st.session_state or not st.session_state.get("ep_path"):
    st.session_state["ep_path"] = get_path("div_result_csv2")

st.set_page_config(page_title='因子综合打分 · 交互可视化', layout='wide')
st.title('因子综合打分交互可视化')

st.caption('本应用在本地浏览器运行，可读取你导出的合并 CSV；选择分数列与指数列后进行可视化与信号标注。')

# 侧边栏：数据输入
with st.sidebar:
    st.header('最终总分判断牛熊及买卖点·参数')
    default_path = get_path("merged_csv")
    use_default = st.toggle('使用默认路径', value=True, help='使用你本地导出的合并 CSV。取消后可在下方上传文件。')
    uploaded = None
    if not use_default:
        uploaded = st.file_uploader('上传 CSV（含 date 与 clqn_prc / score_*）', type=['csv'])

# 读取数据
try:
    if use_default:
        df0 = pd.read_csv(default_path)
    else:
        if uploaded is None:
            st.info('请在侧边栏上传 CSV，或启用“使用默认路径”。')
            st.stop()
        df0 = pd.read_csv(uploaded)
except Exception as e:
    st.error(f'读取 CSV 失败：{e}')
    st.stop()

# 规范化列
_df = df0.copy()
cols = list(_df.columns)
# 日期列容错：date 或 trd_date
if 'date' not in _df.columns:
    d_alt = [c for c in _df.columns if c.lower() == 'trd_date']
    if d_alt:
        _df['date'] = pd.to_datetime(_df[d_alt[0]], errors='coerce')
    else:
        st.error('未找到日期列（需要 date 或 trd_date）')
        st.stop()
else:
    _df['date'] = pd.to_datetime(_df['date'], errors='coerce')

_df = _df.sort_values('date').dropna(subset=['date']).reset_index(drop=True)

# 指数候选列
index_candidates = [c for c in _df.columns if c in ['clqn_prc', 'close', 'index', 'px']]
if not index_candidates:
    # 尝试猜一个数值列作为指数
    numeric_candidates = [c for c in _df.columns if c not in ['date']]
    index_candidates = numeric_candidates[:1]

# 分数候选列
score_candidates = [c for c in _df.columns if c.startswith('score_') or c.startswith('scoreh') or c.startswith('sum_score')]
if not score_candidates:
    # 回退：尝试挑选 f41..f411 做“分数”（虽然不是你滚动回归的总分）
    score_candidates = [c for c in _df.columns if c.startswith('f4')]

# 侧边栏参数
with st.sidebar:
    st.header('最终总分判断牛熊及买卖点·参数')
    score_col = st.selectbox('分数列', score_candidates, index=0 if score_candidates else None, key='main_score')
    index_col = st.selectbox('指数列', index_candidates, index=0 if index_candidates else None, key='main_index')
    start_date = st.date_input('起始日期（可选）', value=None, key='main_start')  # 若报错，可改成 text_input
    sigma_k = st.number_input('强弱带宽 K（σ倍数）', min_value=0.1, max_value=3.0, value=1.0, step=0.1, key='main_k')
    show_bands = st.checkbox('显示均值与 ±σ 带', value=True, key='main_bands')
    use_quadrant = st.checkbox('启用强买强卖信号', value=True, key='main_quad')


# 过滤起始日期
if start_date:
    _df = _df[_df['date'] >= pd.Timestamp(start_date)]

# 数值转换
_df[index_col] = _df[index_col].map(to_num)
_df[score_col] = _df[score_col].map(to_num)

# 计算统计量与信号
mu, sigma = zstats(_df[score_col])
if use_quadrant:
    sig_df = find_signals(_df, score_col, index_col, mu, sigma, k=sigma_k, slope_win=28, slope_th=0.0)
else:
    sig_df = pd.DataFrame(columns=['date', 'score', 'index_px', 'type'])

# 主图
fig = make_main_figure(_df, score_col, index_col, show_bands, mu, sigma, sigma_k, sig_df)
st.plotly_chart(fig, use_container_width=True)

# 下载主图（PNG）
with st.expander('导出主图 / 数据'):
    # st.write('若下载 PNG 失败，请先 `pip install -U kaleido`。')
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button('下载主图 PNG'):
            try:
                import kaleido  # noqa: F401
                png_bytes = fig.to_image(format='png', scale=2)
                st.download_button('点击保存 PNG', data=png_bytes, file_name='score_vs_index.png', mime='image/png')
            except Exception as e:
                st.warning(f'导出失败：{e}')
    with colB:
        st.download_button('下载强买强卖信号 CSV', data=sig_df.to_csv(index=False).encode('utf-8-sig'),
                           file_name='signals.csv', mime='text/csv')
    with colC:
        st.download_button('下载详细分数表 CSV', data=_df.to_csv(index=False).encode('utf-8-sig'),
                           file_name='filtered_data.csv', mime='text/csv')

# 最近信号表
st.subheader('强买强卖信号集合')
if not sig_df.empty:
    st.dataframe(sig_df.tail(12).iloc[::-1].reset_index(drop=True))
else:
    st.info('暂无信号。')



# ========== 可选：在正文里配置因子释义（不想配置可忽略） ==========
# with st.expander("（可选）配置因子名称与释义", expanded=False):
#     st.caption("优先顺序：上传CSV/JSON > 粘贴JSON文本 > 内置默认映射。若都不提供，则仅显示列名。")
#     fac_meta_file = st.file_uploader('上传因子字典（CSV或JSON）', type=['csv','json'], key='fac_meta_upl_main',
#                                      help='CSV需含列：col,name,desc；JSON为 {列名: {"name": 名称, "desc": 释义}}')
#     fac_meta_text = st.text_area('或粘贴JSON映射（可空）', value='',
#                                  placeholder='{"f41": {"name": "动量(短期)", "desc": "近N日收益动量，衡量趋势延续性"}}',
#                                  key='fac_meta_text_main')

# ================== 单因子图（来自本地导出的PNG/JPG） ==================
st.subheader('单因子分数图')

import os
from PIL import Image

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

img_dir = get_path("single_factor_img_dir")





if not os.path.isdir(img_dir):
    st.warning(f"找不到图片目录：{img_dir}")
else:
    custom_order = ["f41", "f42", "f43", "f44", "f45", "f47", "f48", "f49", "f411"]
    order_rank = {fac: i for i, fac in enumerate(custom_order)}


    def get_fac_key(fname: str):
        # 从文件名里提取 f41 这种前缀
        return os.path.splitext(fname)[0].split("_")[0]


    img_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # 未在 custom_order 里的放到最后
    img_files = sorted(
        img_files,
        key=lambda f: order_rank.get(get_fac_key(f), 10_000)
    )

    if not img_files:
        st.info("目录里没找到单因子图（png/jpg/jpeg）。")
    else:
        per_row = 3
        for i in range(0, len(img_files), per_row):
            cols = st.columns(per_row)
            for j, img_file in enumerate(img_files[i:i+per_row]):
                img_path = os.path.join(img_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                fac_key = base_name.split("_")[0]  # 提取 f41 这种前缀

                desc_text = fac_desc.get(fac_key, "")
                with cols[j]:
                    st.markdown(f"**{base_name}**")
                    if desc_text:
                        st.caption(desc_text)
                    try:
                        # 用 PIL 打开避免中文路径兼容问题
                        image = Image.open(img_path)
                        st.image(image)
                    except Exception as e:
                        st.error(f"加载图片失败: {img_path}\n错误: {e}")


# ======== 1.1（只读CSV·固定列名） ========
st.markdown("---")
st.subheader("1.1 300指数股息率 / 十年国债")

with st.expander("指标说明", expanded=False):
    st.write("300指数（剔除了金融股）的股息率与十年期国债收益率的比较。股息率是指股票的年度股息除以股价，而十年期国债收益率则是债券投资的回报率。图中展示了这两者的变化趋势。蓝色虚线表示平均值，+1倍和-1倍标准差，图示了股息率与国债收益率的相对收益率情况。(已增加上证综指)")
    


with st.sidebar:
    st.header("1.1 300指数股息率 / 十年国债·参数")
    div_csv_path = st.text_input(
        "CSV路径",
        value=get_path("div_result_csv"),  # 你原来的路径键
        key="div_csv_path_fixed"
    )
    # div_uploaded = st.file_uploader("或上传CSV（留空则使用路径）", type=["csv"], key="div_csv_upload_fixed")
    div_start = st.text_input("起始日(YYYYMMDD，可空)", value="", key="div_start_fixed")
    div_end   = st.text_input("结束日(YYYYMMDD，可空)", value="", key="div_end_fixed")
    show_bands_11 = st.checkbox("显示均值与±1σ", value=True, key="div_bands_fixed")

# # —— 1.1 读源诊断——
# with st.sidebar:
#     st.caption("— 1.1 路径诊断 —")
#     st.write("配置默认：", get_path("div_result_csv"))
#     _raw = st.session_state.get("div_csv_path_fixed", "")
#     st.write("输入框当前值：", _raw)
#     _resolved = resolve_first_existing(_raw)
#     st.write("解析后路径：", str(_resolved) if _resolved else None)
#     st.write("是否存在：", os.path.exists(_raw))
#     st.write("是否使用上传文件：", div_uploaded is not None)

def _read_csv_smart(src):
    # 自动识别分隔符（逗号/制表符），容错编码
    try:
        return pd.read_csv(src, sep=None, engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(src, sep=None, engine="python", encoding="utf-8-sig")

btn_csv_11 = st.button("生成图表", type="primary", key="div_btn_fixed")
if btn_csv_11:
    try:
        # 读取
        _p = resolve_first_existing(div_csv_path)
        if _p is None:
            st.error(f"路径无效：{div_csv_path}")
            st.stop()
        df = _read_csv_smart(_p)

        # 固定列名校验
        need = ["trade_date", "weighted_dividend_rate", "sh_close",
                "nation10_yield", "weighted_dividend_rate_div_nation10"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            st.error(f"CSV 缺少必要列：{miss}\n读取到的列：{list(df.columns)}")
            st.stop()

        # 规范与过滤
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = (df.dropna(subset=["trade_date"])
                .sort_values("trade_date")
                .drop_duplicates("trade_date", keep="last"))
        if div_start:
            try: df = df[df["trade_date"] >= pd.to_datetime(div_start, format="%Y%m%d")]
            except: st.warning("起始日格式应为 YYYYMMDD，已忽略。")
        if div_end:
            try: df = df[df["trade_date"] <= pd.to_datetime(div_end,   format="%Y%m%d")]
            except: st.warning("结束日格式应为 YYYYMMDD，已忽略。")
        if df.empty:
            st.warning("过滤后无数据。"); st.stop()

        # 数值化
        for c in ["weighted_dividend_rate", "nation10_yield",
                  "weighted_dividend_rate_div_nation10", "sh_close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 使用你已给好的比值列；它前期为 NaN 也没事
        df["ratio"] = df["weighted_dividend_rate_div_nation10"]

        # 统计只对有效值
        s = df["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        mu = float(s.mean()) if len(s) else 0.0
        sd = float(s.std(ddof=1)) if len(s) else 1.0
        if sd == 0 or np.isnan(sd): sd = 1.0

        # 画图
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["trade_date"], y=df["ratio"],
                                 mode="lines", name="300股息率/十年国债", yaxis="y1"))
        if show_bands_11 and len(s):
            fig.add_trace(go.Scatter(x=df["trade_date"], y=[mu]*len(df),
                                     mode="lines", name="均值", line=dict(dash="dot"), yaxis="y1"))
            fig.add_trace(go.Scatter(x=df["trade_date"], y=[mu+sd]*len(df),
                                     mode="lines", name="均值+1σ", line=dict(dash="dash"), yaxis="y1"))
            fig.add_trace(go.Scatter(x=df["trade_date"], y=[mu-sd]*len(df),
                                     mode="lines", name="均值-1σ", line=dict(dash="dash"), yaxis="y1"))

        fig.add_trace(go.Scatter(x=df["trade_date"], y=df["sh_close"],
                                 mode="lines", name="上证综指", yaxis="y2"))

        fig.update_layout(
            template="plotly_dark",
            height=560,
            legend=dict(orientation="h", x=0, y=1.12),
            margin=dict(l=60, r=70, t=40, b=40),
            xaxis=dict(title="日期"),
            yaxis=dict(title="股息率/十年国债", autorange="reversed"),
            yaxis2=dict(title="上证综指", side="right", overlaying="y")
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("下载当前视图数据"):
            st.download_button(
                "下载CSV",
                data=df[["trade_date","weighted_dividend_rate","nation10_yield",
                         "weighted_dividend_rate_div_nation10","sh_close","ratio"]]
                    .to_csv(index=False).encode("utf-8-sig"),
                file_name="1.1_div_vs_10y_with_sh.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"生成图表失败：{type(e).__name__}: {e}")



####################
# ======== 新增：1.2 全A E/P 减 10Y 国债（风险溢价） ========
import plotly.graph_objects as go

st.markdown("---")
st.subheader("1.2 全A E/P(市盈率倒数) − 十年期国债")

with st.expander("指标说明", expanded=False):
    st.write("""
    本图展示了A股的风险溢价（E/P，即市盈率倒数）与十年期国债收益率的时序关系。风险溢价表示股票市场的收益率相对于国债的超额回报，E/P 值越高，表示股票的风险溢价越高。通过计算市盈率倒数和股票的市值加权，得出加权的风险溢价。图中展示了A股风险溢价随时间的变化趋势，以及其相对于十年期国债收益率的波动范围。
    蓝色虚线表示风险溢价的均值，+1倍和-1倍标准差，图示了股市的波动性和国债收益率的变化关系。
    """)

# 侧边栏参数
with st.sidebar:
    st.header("1.2 全A E/P(市盈率倒数)−十年期国债·参数")
    ep_default = get_path("div_result_csv2")
    ep_csv_path = st.text_input(
        "E/P−10Y 结果CSV路径",
        value=get_path("div_result_csv2"),
        key="ep_path"
    )





    ep_start = st.text_input("起始日(YYYYMMDD，可空)", value="", key="ep_start")
    ep_end   = st.text_input("结束日(YYYYMMDD，可空)", value="", key="ep_end")
    ep_clip  = st.checkbox("1%/99% 去极值", value=True, key="ep_clip")
    ep_bands = st.checkbox("显示均值与±1σ", value=True, key="ep_bands")




@st.cache_data(show_spinner=False)
def load_ep10_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 容错：列名/类型
    if "trade_date" not in df.columns:
        raise ValueError("CSV 缺少 trade_date 列")
    if "weighted_ep_10bond" not in df.columns:
        # 兼容旧文件：若只有 weighted_ep 与 ten_y_dec，也可在外部先处理后再读取
        raise ValueError("CSV 缺少 weighted_ep_10bond 列")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
    # 若你之前不小心重复 append（一天写两行），这里取“同日最后一条”
    df = df.drop_duplicates("trade_date", keep="last")
    df["weighted_ep_10bond"] = pd.to_numeric(df["weighted_ep_10bond"], errors="coerce")
    return df



# —— 读源诊断 —— #
# with st.sidebar:
#     st.caption("— 1.2 路径诊断 —")
#     st.write("ep_csv_path(原值)：", repr(st.session_state.get("ep_path")))
#     st.write("cwd：", os.getcwd())
#     _resolved = resolve_first_existing(st.session_state.get("ep_path", ""))
#     st.write("解析结果：", repr(str(_resolved) if _resolved else None))
#     st.write("存在性：", os.path.exists(st.session_state.get("ep_path","")))
#     if st.button("恢复默认(1.2)", key="ep_reset_btn"):
#         st.session_state["ep_path"] = get_path("div_result_csv2")
#         st.success(f"已恢复默认：{st.session_state['ep_path']}")


btn_ep = st.button("生成图表", type="primary", key="ep_btn")
if btn_ep:
    try:
        ep_path_raw = st.session_state.get("ep_path", "")
        ep_path = resolve_first_existing(ep_path_raw)
        if ep_path is None:
            st.error(f"路径无效：{ep_path_raw}\n"
                     f"工作目录：{os.getcwd()}\n"
                     f"建议：把文件放到以上工作目录下的 {ep_path_raw}，或点“恢复默认(1.2)”，"
                     f"或手动在输入框里粘贴绝对路径。")
            st.stop()

        # 真正读取（带一点容错编码）
        try:
            epdf = pd.read_csv(ep_path)
        except UnicodeDecodeError:
            epdf = pd.read_csv(ep_path, encoding="utf-8-sig")
        except Exception:
            epdf = pd.read_csv(ep_path, engine="python")

        # 列/类型规范化（避免后续筛选/作图出错）
        if "trade_date" not in epdf.columns:
            st.error("CSV 缺少 trade_date 列"); st.stop()
        if "weighted_ep_10bond" not in epdf.columns:
            st.error("CSV 缺少 weighted_ep_10bond 列"); st.stop()

        epdf["trade_date"] = pd.to_datetime(epdf["trade_date"], errors="coerce")
        epdf = epdf.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        epdf["weighted_ep_10bond"] = pd.to_numeric(epdf["weighted_ep_10bond"], errors="coerce")

        # 时间过滤
        if ep_start:
            try: epdf = epdf[epdf["trade_date"] >= pd.to_datetime(ep_start, format="%Y%m%d")]
            except: st.warning("起始日格式应为 YYYYMMDD，已忽略。")
        if ep_end:
            try: epdf = epdf[epdf["trade_date"] <= pd.to_datetime(ep_end,   format="%Y%m%d")]
            except: st.warning("结束日格式应为 YYYYMMDD，已忽略。")

        if epdf.empty:
            st.warning("过滤后无数据。"); st.stop()

        s = epdf["weighted_ep_10bond"].copy()
        if ep_clip:
            lo, hi = s.quantile([0.01, 0.99])
            s = s.clip(lo, hi)
        mu = float(pd.to_numeric(s, errors="coerce").mean())
        sd = float(pd.to_numeric(s, errors="coerce").std(ddof=1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epdf["trade_date"], y=s, mode="lines",
                                 name="A股风险溢价：E/P − 10Y（小数）", yaxis="y1"))
        if ep_bands:
            fig.add_trace(go.Scatter(x=epdf["trade_date"], y=[mu]*len(epdf),
                                     mode="lines", name="均值", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=epdf["trade_date"], y=[mu+sd]*len(epdf),
                                     mode="lines", name="均值+1σ", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=epdf["trade_date"], y=[mu-sd]*len(epdf),
                                     mode="lines", name="均值-1σ", line=dict(dash="dash")))
        fig.update_layout(template="plotly_dark", height=520,
                          legend=dict(orientation="h", x=0, y=1.12),
                          margin=dict(l=60, r=40, t=40, b=40),
                          xaxis=dict(title="日期"),
                          yaxis=dict(title="E/P − 10Y（小数）"))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("下载当前视图数据"):
            out_df = epdf.copy()
            out_df["weighted_ep_10bond_clean"] = s.values
            st.download_button("下载CSV", data=out_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="ep_minus_10y_clean.csv", mime="text/csv")

    except Exception as e:
        st.error(f"生成失败：{type(e).__name__}: {e}")


######################大小盘轮动
# ======== 新增：2.1 大小盘轮动（读取线下CSV，交互展示） ========
import plotly.graph_objects as go

st.markdown("---")
st.subheader("2.1 大小盘轮动")

with st.expander("指标说明", expanded=False):
    st.write("""
    跟踪“大盘vs小盘”等多空组合的收益和净值，在任意周期（日/月/季）上显示谁在持续占优，并用均值±σ告诉你这个风格强弱是否已经走到极端、有没有到该防风格反转/调仓的时点。
    """)

# 侧边栏参数
with st.sidebar:
    st.header("2.1 大小盘轮动·参数")
    base_dir = st.text_input("CSV 目录", value=get_path("turn_std_png_dir"), key="rot_dir")
    path_daily   = st.text_input("日度CSV", value=get_path("rot_daily_csv"), key="rot_path_daily")
    path_monthly = st.text_input("月度CSV", value=get_path("rot_month_csv"), key="rot_path_month")
    path_quarter = st.text_input("季度CSV", value=get_path("rot_quarter_csv"), key="rot_path_quarter")
    freq = st.radio("频率", ["日度", "月度", "季度"], index=1, key="rot_freq")
    view = st.radio("指标", ["收益", "净值(NAV)"], index=0, key="rot_view")
    k_sigma = st.number_input("±σ 带宽（σ倍数，仅收益视图有效）", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key="rot_k")
    show_band = st.checkbox("显示均值与±σ（仅收益视图）", value=True, key="rot_band")
    date_start = st.text_input("起始日(YYYYMMDD，可空)", value="", key="rot_start")
    date_end   = st.text_input("结束日(YYYYMMDD，可空)", value="", key="rot_end")

@st.cache_data(show_spinner=False)
def _load_df(path: str, idx_name: str):
    df = pd.read_csv(path)
    # 兼容：索引列可能已经叫 date/month/quarter，也可能就没明示
    # 尝试识别第1列为日期索引
    # 若文件有名为 idx_name 的列，用它；否则找第1列
    if idx_name in df.columns:
        dt = pd.to_datetime(df[idx_name], errors="coerce")
    else:
        dt = pd.to_datetime(df.iloc[:,0], errors="coerce")
    df.insert(0, "dt", dt)
    df = df.dropna(subset=["dt"]).drop(columns=[c for c in ["date","month","quarter"] if c in df.columns], errors="ignore")
    df = df.set_index("dt").sort_index()
    # 数值化
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def _agg_compound(ret_df: pd.DataFrame, rule: str):
    """从日收益复利聚合到 rule（'M'或'Q'）"""
    # 只取收益列（剔除 _NAV）
    cols_ret = [c for c in ret_df.columns if not c.endswith("_NAV")]
    if not cols_ret:
        return pd.DataFrame(), pd.DataFrame()
    ret = ret_df[cols_ret].resample(rule).apply(lambda x: (1 + x).prod() - 1)
    nav = (1 + ret).cumprod()
    nav.columns = [c + "_NAV" for c in nav.columns]
    return ret, nav

def _clip_by_date(df: pd.DataFrame, start_str: str, end_str: str):
    out = df.copy()
    if start_str:
        try:
            out = out[out.index >= pd.to_datetime(start_str, format="%Y%m%d")]
        except:
            pass
    if end_str:
        try:
            out = out[out.index <= pd.to_datetime(end_str,   format="%Y%m%d")]
        except:
            pass
    return out

def _mean_std(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
    if s2.empty:
        return 0.0, 1.0
    return float(s2.mean()), float(s2.std(ddof=1)) or 1.0

# 读取基础数据
daily_df, month_df, quarter_df = None, None, None
try:
    if os.path.exists(path_daily):
        daily_df = _load_df(path_daily, idx_name="date")
    if os.path.exists(path_monthly):
        month_df = _load_df(path_monthly, idx_name="month")
    if os.path.exists(path_quarter):
        quarter_df = _load_df(path_quarter, idx_name="quarter")
except Exception as e:
    st.error(f"读取大小盘CSV失败：{e}")
    st.stop()

# 确定可选组合
source_df = {"日度": daily_df, "月度": month_df, "季度": quarter_df}[freq]
if (source_df is None) or source_df.empty:
    # 回退：若只给了日度，而你选了月/季，则从日度聚合
    if (freq in ["月度","季度"]) and (daily_df is not None) and (not daily_df.empty):
        agg_rule = "M" if freq == "月度" else "Q"
        ret_agg, nav_agg = _agg_compound(daily_df, agg_rule)
        source_df = pd.concat([ret_agg, nav_agg], axis=1)
    else:
        st.warning(f"未检测到 {freq} 数据且无法回退。请检查对应CSV是否存在。")
        st.stop()

# 拿出“收益列名”（非 _NAV）和“净值列名”（_NAV）
spread_cols = [c for c in source_df.columns if not c.endswith("_NAV")]
nav_cols    = [c for c in source_df.columns if c.endswith("_NAV")]

default_choices = spread_cols[:2] if view == "收益" else nav_cols[:2]
choices = st.multiselect(
    f"选择要展示的{'收益组合' if view=='收益' else '净值组合'}（可多选）",
    options=(spread_cols if view=="收益" else nav_cols),
    default=default_choices,
    key="rot_chosen"
)

if not choices:
    st.info("请选择至少一个组合列。")
    st.stop()

# 时间过滤
plot_df = _clip_by_date(source_df[choices], date_start, date_end)
if plot_df.empty:
    st.warning("时间过滤后无数据。"); st.stop()

# 画图
fig = go.Figure()
for c in choices:
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[c], mode="lines", name=c, yaxis="y1"))

title_suffix = "（收益）" if view == "收益" else "（净值）"
fig_title = f"大小盘轮动 {freq} {title_suffix}"

# 均值 ± σ（仅收益）
if (view == "收益") and show_band and (len(choices) == 1):
    s = plot_df[choices[0]]
    mu, sd = _mean_std(s)
    fig.add_trace(go.Scatter(x=plot_df.index, y=[mu]*len(plot_df), mode="lines",
                             name="均值", line=dict(dash="dot"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=[mu + k_sigma*sd]*len(plot_df), mode="lines",
                             name=f"+{k_sigma}σ", line=dict(dash="dash"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=[mu - k_sigma*sd]*len(plot_df), mode="lines",
                             name=f"-{k_sigma}σ", line=dict(dash="dash"), yaxis="y1"))

fig.update_layout(
    template="plotly_dark",
    height=560,
    legend=dict(orientation="h", x=0, y=1.12),
    margin=dict(l=60, r=70, t=40, b=40),
    xaxis=dict(title="日期"),
    yaxis=dict(title=("收益" if view=="收益" else "净值"), side="left"),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("下载当前视图数据"):
    st.download_button("下载CSV", data=plot_df.reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"大小盘_{freq}_{'收益' if view=='收益' else '净值'}.csv", mime="text/csv")

# ======================= 2.2 行业拥挤度（离线产出展示版，不卡内存） =======================
import pandas as pd
from PIL import Image

# --- 侧边栏：2.2 时间区间参数 ---
with st.sidebar:
    st.header("2.2 行业拥挤度·参数")
    crowd_start = st.text_input(
        "起始日(YYYYMMDD，可空)",
        value="",
        key="crowd_start"
    )
    crowd_end = st.text_input(
        "结束日(YYYYMMDD，可空)",
        value="",
        key="crowd_end"
    )


def _fmt_range(start_str: str, end_str: str) -> str:
    """
    把用户输入的起止日期拼成展示文案。
    - start+end 都有：'20240101 ~ 20251024'
    - 只有 start：'20240101 ~ 最新'
    - 只有 end：'截至 20251024'
    - 都没有：'全区间'
    """
    s = (start_str or "").strip()
    e = (end_str or "").strip()
    if s and e:
        return f"{s} ~ {e}"
    elif s and not e:
        return f"{s} ~ 最新"
    elif (not s) and e:
        return f"截至 {e}"
    else:
        return "全区间"


st.markdown("---")
st.subheader("2.2 行业拥挤度 / TMT 拥挤度 / 行业热度榜")

with st.expander("指标说明", expanded=False):
    st.write("""
    本节数据由离线批处理脚本生成（全A逐日成交+行业映射+周度换手等大数据聚合）。
    本页面只负责展示与下载，不在页面内重新计算，避免全量重算导致卡顿。目前时间区间为'2025-08-01'到'2025-11-01'

    图1：行业成交额占比
    统计整个时间区间内各行业累计成交额在全市场中的比例，用来观察长期资金关注度。占比越高的行业，说明在时间区间里交易最活跃、资金集中度最高。

    图2：TMT 拥挤度（MA5）
    计算电子、通信、传媒、计算机四个行业的成交额占比，并取 5 日均值，反映短期市场热度。曲线上升说明资金流入 TMT 板块加快，下降则表示热度减退。

    表3：行业热度榜（最近一周）
    展示时间区间中最近一周各行业的成交额占比、分位数和平均换手率，用于识别“拥挤/冷清”的方向。
    """)

# 线下输出产物路径（保持你原来的路径）
img_path_industry = get_path("industry_img1")
img_path_tmt = get_path("industry_img2")
xlsx_path_table = get_path("industry_table_xlsx")

col_left, col_right = st.columns(2)

# 图1：行业期间成交占比柱图
with col_left:
    st.markdown("图1：行业成交额占比")
    # 新增：展示本批统计区间
    st.caption(f"区间：{_fmt_range(crowd_start, crowd_end)}")

    try:
        img1 = Image.open(img_path_industry)
        st.image(img1)
        with open(img_path_industry, "rb") as f:
            st.download_button(
                "下载图1 PNG",
                data=f,
                file_name="2.2_行业成交占比.png",
                mime="image/png"
            )
    except Exception as e:
        st.warning(f"无法加载图1: {e}")

# 图2：TMT拥挤度曲线
with col_right:
    st.markdown("**图2：TMT拥挤度（MA5）**")
    # 新增：展示本批统计区间
    st.caption(f"区间：{_fmt_range(crowd_start, crowd_end)}")

    try:
        img2 = Image.open(img_path_tmt)
        st.image(img2)
        with open(img_path_tmt, "rb") as f:
            st.download_button(
                "下载图2 PNG",
                data=f,
                file_name="2.2_TMT拥挤度.png",
                mime="image/png"
            )
    except Exception as e:
        st.warning(f"无法加载图2: {e}")

# 表3：行业热度榜
st.markdown("**表3：行业热度榜（区间最后一周截面）**")
# 新增：展示本批统计区间
st.caption(f"区间：{_fmt_range(crowd_start, crowd_end)}")

try:
    df_table = pd.read_excel(xlsx_path_table)
    st.dataframe(df_table)

    st.download_button(
        "下载行业周度概览 (Excel)",
        data=open(xlsx_path_table, "rb"),
        file_name="2.2_行业热度榜.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except Exception as e:
    st.warning(f"无法读取表3: {e}")

# ======================= 3.1 换手率横截面标准差（全A + 行业）·合并对比版 =======================
# 说明：
# - 该模块仅展示离线脚本已经生成的 CSV / PNG，不做在线重算，也不需要任何密钥。
# - “全A”与“行业”合并在同一个多选框里统一选择与对比。
# - 可选择时间区间、聚合频率（日/月/季）、平滑窗口，以及展示单位（小数/百分比）。

import os
import re
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- 侧边栏参数 ----------
with st.sidebar:
    st.header("3.1 换手率标准差·参数")
    # base_dir_31 = st.text_input("结果目录", value=get_path("turn_std_png_dir"), key="std31_base")
    csv_path_31 = st.text_input("全量CSV路径", value=get_path("turn_std_csv"), key="std31_csv")
    # png_dir_31  = st.text_input("PNG目录（可与结果目录相同）", value=base_dir_31, key="std31_pngdir")

    dt_start_31 = st.text_input("起始日(YYYYMMDD，可空)", value="", key="std31_start")
    dt_end_31   = st.text_input("结束日(YYYYMMDD，可空)", value="", key="std31_end")

    agg_rule_31   = st.selectbox("可选聚合频率", ["不聚合(逐日)", "月度", "季度"], index=0, key="std31_agg")
    smooth_win_31 = st.number_input("平滑窗口（移动平均，期）", min_value=1, max_value=60, value=3, step=1, key="std31_smooth")
    # unit_view_31  = st.selectbox("展示单位", ["小数", "百分比(%)"], index=0, key="std31_unit")

# ---------- 工具函数 ----------
def _parse_dt_31(s: str):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    st.warning("日期格式应为 YYYYMMDD 或 YYYY-MM-DD")
    return None

@st.cache_data(show_spinner=False)
def load_turn_std_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need_cols = {"trade_date", "scope", "level1_industry_name", "turn_daily_std"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"CSV 缺少必要列：{miss}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
    df["turn_daily_std"] = pd.to_numeric(df["turn_daily_std"], errors="coerce")
    return df

def _clip_range(df: pd.DataFrame, s_dt, e_dt) -> pd.DataFrame:
    out = df.copy()
    if s_dt is not None:
        out = out[out["trade_date"] >= s_dt]
    if e_dt is not None:
        out = out[out["trade_date"] <= e_dt]
    return out

def _agg_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "不聚合(逐日)":
        return df
    rule_map = {"月度": "M", "季度": "Q"}
    res_rule = rule_map.get(rule)
    if not res_rule:
        return df
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

def _smooth_series(x: pd.Series, win: int):
    if win <= 1:
        return x
    return x.rolling(window=win, min_periods=1).mean()

def _fmt_range_text(s_dt, e_dt):
    if s_dt and e_dt:
        return f"{s_dt.strftime('%Y%m%d')} ~ {e_dt.strftime('%Y%m%d')}"
    if s_dt and not e_dt:
        return f"{s_dt.strftime('%Y%m%d')} ~ 最新"
    if (not s_dt) and e_dt:
        return f"截至 {e_dt.strftime('%Y%m%d')}"
    return "全区间"

# ---------- 主体（合并选择：全A + 行业） ----------
st.markdown("---")
st.subheader("3.1 换手率横截面标准差（全A + 行业）")
with st.expander("指标说明", expanded=False):
    st.write(
        """
        - 本页汇总：时间过滤、按日/月/季聚合、平滑窗口、单位切换（小数/百分比），并把全A与行业统一对比。
     
        """
    )

# 读取 CSV
try:
    df31_raw = load_turn_std_csv(csv_path_31)
except Exception as e:
    st.warning(f"无法读取 3.1 全量CSV：{e}")
    df31_raw = pd.DataFrame(columns=["trade_date","scope","level1_industry_name","turn_daily_std"])  # 空壳

if df31_raw.empty:
    st.info("未检测到 3.1 离线结果。请先运行离线脚本生成 CSV/PNG 再查看。")
else:
    # 时间与聚合
    s_dt = _parse_dt_31(dt_start_31)
    e_dt = _parse_dt_31(dt_end_31)
    df31 = _clip_range(df31_raw, s_dt, e_dt)
    df31 = _agg_df(df31, agg_rule_31)

    # 单位
    # if unit_view_31 == "百分比(%)":
    #     df31["turn_std_view"] = df31["turn_daily_std"] * 100.0
    #     y_label = "turn_daily_std(%)"
    # else:
    #     df31["turn_std_view"] = df31["turn_daily_std"]
    #     y_label = "turn_daily_std"

    df31["turn_std_view"] = df31["turn_daily_std"]
    y_label = "turn_daily_std"

    # 系列集合：把“全A”也当作一个选项和行业并列
    industries_all = sorted(
        df31.loc[df31["scope"]=="行业","level1_industry_name"].dropna().unique().tolist()
    )
    has_allA = (df31["scope"]=="全A").any()
    series_all = (["全A"] if has_allA else []) + industries_all

    st.caption(f"区间：{_fmt_range_text(s_dt, e_dt)} ｜ 频率：{agg_rule_31} ｜ 平滑窗口：{smooth_win_31}")
    chosen_series = st.multiselect(
        "选择系列（行业或全A，可多选）",
        options=series_all,
        default=series_all[:6] if series_all else [],
        key="std31_series"
    )

    if not chosen_series:
        st.info("请选择至少一个系列（行业或全A）。")
    else:
        # 画图
        fig = go.Figure()
        export_parts = []
        for name in chosen_series:
            if name == "全A":
                g = df31[df31["scope"]=="全A"].sort_values("trade_date")
            else:
                g = df31[(df31["scope"]=="行业") & (df31["level1_industry_name"]==name)].sort_values("trade_date")
            if g.empty:
                continue
            y = _smooth_series(g["turn_std_view"], smooth_win_31)
            fig.add_trace(go.Scatter(x=g["trade_date"], y=y, mode="lines", name=str(name)))
            tmp = g[["trade_date","turn_std_view"]].copy()
            tmp["series"] = name
            export_parts.append(tmp)

        fig.update_layout(
            template="plotly_dark",
            height=560,
            legend=dict(orientation="h", x=0, y=1.12),
            margin=dict(l=60, r=40, t=40, b=40),
            xaxis=dict(title="日期"),
            yaxis=dict(title=y_label),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 下载当前筛选数据
        if export_parts:
            export_df = pd.concat(export_parts, ignore_index=True)
            export_df = export_df.rename(columns={"turn_std_view": "turnover_std_view"})
            st.download_button(
                "下载当前筛选数据（CSV）",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="3.1_turnover_std_selected.csv",
                mime="text/csv"
            )

        # 原PNG预览：根据所选系列展示
        # st.markdown("**离线脚本导出的原图预览（按所选系列）**")
        # cols = st.columns(3)
        # col_idx = 0
        # for name in chosen_series:
        #     try:
        #         if name == "全A":
        #             p = os.path.join(png_dir_31, "3.1_全A_turn_daily_std_逐年趋势.png")
        #         else:
        #             # safe_ind = re.sub(r'[\\/:*?"<>|]', '_', str(name))
        #             p = os.path.join(png_dir_31, f"3.1_{name}_turn_daily_std_逐年趋势.png")
        #         if os.path.exists(p):
        #             # with cols[col_idx % 3]:
        #             #     st.image(p, caption=str(name))
        #             with open(p, "rb") as f:
        #                 st.download_button(f"下载「{name}」PNG", data=f, file_name=os.path.basename(p), mime="image/png")
        #             col_idx += 1
        #     except Exception as e:
        #         st.warning(f"读取「{name}」PNG 失败：{e}")

















































