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

import plotly.express as px

import os

os.environ["PLOTLY_JSON_ENGINE"] = "json"

import io
import numpy as np
import pandas as pd

import plotly.io as pio

pio.json.config.default_engine = "json"  # 再保险一层

import plotly.graph_objects as go
import streamlit as st

from pathlib import Path
import json

import streamlit.components.v1 as components
from typing import Union

st.set_page_config(page_title='自营牛熊因子模型', layout='wide')
# ------- 旧浏览器 .at() Polyfill（Edge 90 用） -------

polyfill = """
<script>
(function() {
  // 在指定的 window 上给 Array / Object 挂 at
  function ensureAt(root) {
    if (!root) return;
    var ObjProto = root.Object && root.Object.prototype;
    var ArrProto = root.Array  && root.Array.prototype;

    function defineAtOn(proto) {
      if (!proto) return;
      var needPatch = !("at" in proto) || typeof proto.at !== "function";
      if (!needPatch) return;
      Object.defineProperty(proto, "at", {
        value: function(n) {
          if (this == null) return undefined;
          var len = this.length >>> 0;  // 转成非负整数
          n = Number(n) || 0;
          if (n < 0) n += len;
          if (n < 0 || n >= len) return undefined;
          return this[n];
        },
        writable: true,
        configurable: true
      });
    }

    defineAtOn(ArrProto);
    defineAtOn(ObjProto);
  }

  try {
    // 先打当前 iframe 自己
    ensureAt(window);
  } catch (e) {}

  try {
    // 再尝试给父页面打补丁（真正跑 Streamlit / Plotly 的地方）
    if (window.parent && window.parent !== window) {
      ensureAt(window.parent);
    }
  } catch (e) {}
})();
</script>
"""

components.html(polyfill, height=0, width=0)


# ------- Polyfill 结束 -------

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
        seg = y.iloc[i - win + 1: i + 1]
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
    out.rename(columns={score_col: 'score', index_col: '指数'}, inplace=True)
    return out


def make_main_figure(
    df: pd.DataFrame,
    score_col: str,
    signals: pd.DataFrame = None,
    shift_weeks: int = 0,
):
    """
    简化版主图（写死指数列 clqn_prc）：
    - 不需要 index_col / start_date / sigma_k / show_bands 参数
    - 默认显示：均值、±1σ（分数轴）
    - 指数轴固定用 clqn_prc
    - 分数线可右移 shift_weeks 周
    """

    d = df.copy()

    # ---- 必要列校验 ----
    if "date" not in d.columns:
        st.error("df 缺少 date 列")
        st.stop()
    if score_col not in d.columns:
        st.error(f"df 缺少分数列：{score_col}")
        st.stop()
    if "clqn_prc" not in d.columns:
        st.error("df 缺少指数列：clqn_prc（已写死）")
        st.stop()

    # ---- 类型规范 ----
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")

    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d["clqn_prc"] = pd.to_numeric(d["clqn_prc"], errors="coerce")

    # ---- 均值 & 标准差（默认±1σ）----
    s = d[score_col].replace([np.inf, -np.inf], np.nan).dropna()
    mu = float(s.mean()) if len(s) else 0.0
    sigma = float(s.std(ddof=1)) if len(s) else 1.0
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0

    fig = go.Figure()

    # 分数线 x 轴右移
    if shift_weeks and shift_weeks != 0:
        score_x = d["date"] + pd.to_timedelta(int(shift_weeks) * 7, unit="D")
    else:
        score_x = d["date"]

    # 分数（左轴）
    fig.add_trace(go.Scatter(
        x=score_x, y=d[score_col], mode="lines", name=score_col, yaxis="y1"
    ))

    # 指数（右轴）固定 clqn_prc
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["clqn_prc"], mode="lines", name="clqn_prc", yaxis="y2"
    ))

    # 默认显示：均值、±1σ（用原始日期，不跟随平移）
    fig.add_trace(go.Scatter(
        x=d["date"], y=[mu] * len(d), mode="lines", name="均值",
        line=dict(dash="dot"), yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=[mu + sigma] * len(d), mode="lines", name="+1σ",
        line=dict(dash="dash"), yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=[mu - sigma] * len(d), mode="lines", name="-1σ",
        line=dict(dash="dash"), yaxis="y1"
    ))

    # 信号点（画在指数轴上）
    if signals is not None and not signals.empty:
        sig = signals.copy()
        sig["date"] = pd.to_datetime(sig["date"], errors="coerce")
        sig = sig.dropna(subset=["date"])

        # 你现在 signals 里已经 rename 过 clqn_prc -> 指数
        if "指数" not in sig.columns and "clqn_prc" in sig.columns:
            sig.rename(columns={"clqn_prc": "指数"}, inplace=True)

        if ("type" in sig.columns) and ("指数" in sig.columns):
            buys = sig[sig["type"] == "强买"]
            sells = sig[sig["type"] == "强卖"]

            if len(buys):
                fig.add_trace(go.Scatter(
                    x=buys["date"], y=buys["指数"], mode="markers", name="强买",
                    marker=dict(symbol="triangle-up", size=10), yaxis="y2"
                ))
            if len(sells):
                fig.add_trace(go.Scatter(
                    x=sells["date"], y=sells["指数"], mode="markers", name="强卖",
                    marker=dict(symbol="triangle-down", size=10), yaxis="y2"
                ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=60, r=70, t=40, b=40),
        legend=dict(orientation="h", x=0, y=1.12),
        xaxis=dict(title="日期"),
        yaxis=dict(title=score_col, side="left"),
        yaxis2=dict(title="clqn_prc", side="right", overlaying="y"),
        height=560,
    )
    return fig



from pathlib import Path

APP_DIR = Path(__file__).parent


def resolve_first_existing(p: str) -> Union[Path, None]:
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

@st.cache_data(show_spinner=False)
def read_csv_default(key: str, *, required_cols=None, parse_dates=None) -> pd.DataFrame:
    """
    只从 config.json 的 paths[key] 读取 CSV，不允许上传/手输路径。
    - required_cols: 必须存在的列名列表
    - parse_dates: 需要 pd.to_datetime 的列名列表
    """
    raw = get_path(key)
    p = resolve_first_existing(raw)
    if p is None:
        st.error(f"找不到默认文件：paths['{key}'] = {raw}\n工作目录：{os.getcwd()}\n脚本目录：{APP_DIR}")
        st.stop()

    try:
        df = pd.read_csv(p)
    except UnicodeDecodeError:
        df = pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(p, engine="python")

    if required_cols:
        miss = [c for c in required_cols if c not in df.columns]
        if miss:
            st.error(f"{key} 缺少必要列：{miss}\n实际列：{list(df.columns)[:50]}")
            st.stop()

    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    return df



# ========== Streamlit UI ==========
# ================展示模型IC

st.title('自营牛熊因子模型可视化')
st.subheader('最新时点因子及模型IC')
# st.caption('在量化模型中，IC（Information Coefficient，信息系数）通常用来衡量一个因子与未来收益之间的相关性。简单来说，IC 反映了因子的预测能力，即它与股票未来回报之间的线性关系。')
st.markdown("""
<div style="line-height: 1.4; color: #808080;">
在量化模型中，IC（Information Coefficient，信息系数）通常用来衡量一个因子与未来收益之间的相关性。简单来说，IC 反映了因子的预测能力，即它与股票未来回报之间的线性关系。<br>
IC通常判别标准：<br>
- IC ≥ 0.05：良好<br>
- IC ≥ 0.1：较强<br>
- IC ≥ 0.2：非常强<br>
- IC ≥ 0.3：超强<br>
- IC < 0：无效或负相关<br><br>

</div>
""", unsafe_allow_html=True)
latest_time_path = get_path('factor_decomp_all_h')
latest_time = pd.read_csv(latest_time_path)
latest_time['date'] = pd.to_datetime(latest_time['date'])
latest_time_max = latest_time['date'].max()
first_time_min = latest_time['date'].min()

st.caption(f"数据时点：{first_time_min} ~ {latest_time_max}")

# #读取IC结果
# st.write(f"数据时点：{first_time_min} ~ {latest_time_max}")
IC_path = get_path("ic_result")
IC_df = pd.read_csv(IC_path)
IC_df = IC_df.rename(columns={'horizon_w': '窗口期（未来几周)', 'factor': '因子', 'type': '因子类型', 'N': '样本数量',
                              'pearson': 'pearson相关系数IC', 'spearman_ic': 'spearman相关系数IC'})
# st.write(IC_df)
# st.dataframe(IC_df)
# print(IC_df)

try:
    best_row = IC_df.sort_values('spearman相关系数IC', ascending=False).iloc[0]
    # best_row = (IC_df.loc[IC_df['因子'].str.endswith('_predret')].sort_values('spearman相关系数IC', ascending=False).iloc[0])
    best_spearman = best_row['spearman相关系数IC']
    best_pearson = best_row['pearson相关系数IC']
    best_factor = best_row['因子']
    best_h = int(best_row['窗口期（未来几周)'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("最佳模型 · Spearman IC", f"{best_spearman:.3f}")
    with col2:
        st.write("最佳模型 · Pearson IC", f"{best_pearson:.3f}")
    with col3:
        st.write(
            "最佳模型（因子 + 窗口）",
            f"{best_factor} · {best_h}周\nSpearman={best_spearman:.3f}",
            help=f"Spearman IC = {best_spearman:.3f}，Pearson IC = {best_pearson:.3f}"
        )

except Exception:
    pass

# 下载IC结果（PNG）
st.download_button('下载IC CSV', data=IC_df.to_csv(index=False).encode('utf-8-sig'), file_name='IC.csv',
                   mime='text/csv')

# —— 避免 ep_path 为空导致 value 被忽略 ——
if "ep_path" not in st.session_state or not st.session_state.get("ep_path"):
    st.session_state["ep_path"] = get_path("div_result_csv2")




    

# 固定读取默认 merged_csv
df0 = read_csv_default("merged_csv")




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
score_candidates = [c for c in _df.columns if
                    c.startswith('score_') or c.startswith('scoreh') or c.startswith('sum_score')]
if not score_candidates:
    # 回退：尝试挑选 f41..f411 做“分数”（虽然不是你滚动回归的总分）
    score_candidates = [c for c in _df.columns if c.startswith('f4')]

# 侧边栏参数
with st.sidebar:
    st.header('最终总分判断牛熊及买卖点·参数')
    score_col = st.selectbox('选择模型（未来1周、未来4周、未来8周、未来12周、未来16周、未来20周）', score_candidates, index=16 if score_candidates else None, key='main_score')
    # index_col = st.selectbox('指数列', index_candidates, index=0 if index_candidates else None, key='main_index')
    # start_date = st.date_input('起始日期（可选）', value=None, key='main_start')  # 若报错，可改成 text_input
    # sigma_k = st.number_input('强弱带宽 K（σ倍数）', min_value=0.1, max_value=3.0, value=1.0, step=0.1, key='main_k')
    # show_bands = st.checkbox('显示均值与 ±σ 带', value=True, key='main_bands')
    use_quadrant = st.checkbox('启用强买强卖信号', value=True, key='main_quad')

    # === 新增：分数线右移（周） ===
    import re


    def _infer_h_from_name(col: str) -> Union[int, None]:
        m = re.search(r'h(\d+)', col)  # 例如 score_h16_predret -> 16
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None


    enable_shift = st.checkbox('将分数线向右平移（按周）', value=True, key='main_shift_enable')
    if enable_shift:
        default_h = _infer_h_from_name(score_col) or 0
        shift_weeks = st.number_input(
            '右移周数（>0 向右）',
            min_value=0, max_value=260, value=default_h, step=1,
            key='main_shift_weeks'
        )
    else:
        shift_weeks = 0

# 过滤起始日期
# if start_date:
#     _df = _df[_df['date'] >= pd.Timestamp(start_date)]

# 数值转换
# _df[index_col] = _df[index_col].map(to_num)
_df[score_col] = _df[score_col].map(to_num)

# 计算统计量与信号
mu, sigma = zstats(_df[score_col])
# if use_quadrant:
#     sig_df = find_signals(_df, score_col, index_col, mu, sigma, k=sigma_k, slope_win=28, slope_th=0.0)
# else:
#     sig_df = pd.DataFrame(columns=['date', 'score', 'index_px', 'type'])
sig_path = get_path("signal_result")
sig_df = pd.read_csv(sig_path)
sig_df['type'] = np.where(sig_df['final_buy_signal'] == True, '强买', '强卖')
sig_df.rename(columns={'clqn_prc': '指数'}, inplace=True)

sig_df = sig_df.drop(columns=['final_sell_signal', 'final_buy_signal', 'index'])

# ================================ 主图
st.subheader('牛熊因子综合模型输出分数及指数对比')
st.caption(
    '下图展示了牛熊因子模型输出的总分在历史中与指数走势的对比，可以根据IC大小选择不同未来周收益率模型输出分数，一般IC越大模型预测效果越好，此外可以向右平移分数线查看当前预测指数走势，例如若选择预测未来16周指数收益率模型，可向右平移分数线16周，查看当前分数走势')
fig = make_main_figure(_df, score_col, signals=sig_df, shift_weeks=int(shift_weeks))


st.plotly_chart(fig, use_container_width=True)

# 下载主图（PNG）
with st.expander('导出数据'):
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
st.caption(
    '强买强卖信号开发逻辑主要是通过网格搜索参数（未来16周分数斜率计算窗口、斜率阈值、均值下方买入区域大小、均值上方卖出区域大小），结合全量数据买卖次数分别不低于8次，目标买卖胜率分别不低于80%条件，选择最佳前述网格参数，并生成相应强买强卖信号。')

if not sig_df.empty:
    st.dataframe(sig_df.iloc[::-1].reset_index(drop=True))
else:
    st.info('暂无信号。')


# ================== 获取因子堆叠柱状图
# 创建堆叠柱状图
@st.cache_data(show_spinner=False)
def create_stacked_bar_chart(df, columns_to_plot, period_label):
    # 将整个 DataFrame 中的 None 或 NaN 值替换为 0
    df = df.fillna(0)

    df_filtered = df[['date'] + columns_to_plot]

    # 定义因子名称到颜色的映射
    # color_map = {
    # 'f41_contrib_h16': 'blue',
    # 'f42_contrib_h16': 'green',
    # 'f43_contrib_h16': 'red',
    # 'f45_contrib_h16': 'purple',
    # 'f49_contrib_h16': 'orange',
    # 'f411_contrib_h16': 'pink'
    # }

    # 创建正值的堆叠柱状图
    fig = px.bar(df_filtered, x='date', y=columns_to_plot,
                 title=f"预测未来{period_label}周模型的因子贡献",
                 labels={'date': '日期'},
                 # template='plotly_dark',
                 # template='presentation',
                 # color_discrete_map=color_map,  # 使用自定义颜色映射
                 barmode='group')  # 堆叠模式

    # 设置透明度
    fig.update_traces(marker=dict(opacity=1))

    # 调整布局
    fig.update_layout(
        xaxis_title='日期',
        yaxis_title='贡献',
        # yaxis=dict(range=[-0.5, 0.5]),  # 根据因子的数值范围调整
        height=800
    )

    return fig



    
df = read_csv_default("factor_decomp_all_h", parse_dates=["date"])


# 选择展示列
# columns_to_plot = [
#     'f41_contrib_h16', 'f42_contrib_h16', 'f43_contrib_h16',
#     'f45_contrib_h16', 'f49_contrib_h16', 'f411_contrib_h16',
#     'intercept_h16', 'score_h16_predret'
# ]
columns_to_plot = [
    'f41_contrib_h16', 'f42_contrib_h16', 'f43_contrib_h16',
    'f45_contrib_h16', 'f49_contrib_h16', 'f411_contrib_h16',
    'intercept_h16'
]
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in ['date'] else x)
df['date'] = pd.to_datetime(df['date'])

# 让用户选择周期长度
st.subheader('查看因子贡献')

period_label = st.selectbox(
    "选择模型（未来1周、未来4周、未来8周、未来12周、未来16周、未来20周）",
    options=[1, 4, 8, 12, 16, 20],
    index=4  # 默认选择16周
)

# 更新因子列名称以适应不同周期
columns_to_plot_for_period = [col.replace('16', str(period_label)) for col in columns_to_plot]
# 强制将所有列转换为数值型，非数值的会被转换为 NaN
df[columns_to_plot_for_period] = df[columns_to_plot_for_period].apply(pd.to_numeric, errors='coerce')

# 删除包含 NaN 的行
df = df.dropna(subset=columns_to_plot_for_period, how='all')
# st.error(f"columns_to_plot_for_period type: {type(columns_to_plot_for_period)}")
st.dataframe(df[['date'] + columns_to_plot_for_period])
# 创建堆叠柱状图
fig = create_stacked_bar_chart(df, columns_to_plot_for_period, period_label)
st.plotly_chart(fig, use_container_width=True, key=f"chart_{period_label}")

# 导出功能
with st.expander('导出数据'):
    st.download_button(
        '下载数据',
        data=df.to_csv(index=False).encode('utf-8-sig'),
        file_name=f'factor_contributions_{period_label}weeks.csv',
        mime='text/csv'
    )

# ================== 单因子图（来自本地导出的PNG/JPG） ==================
st.subheader('单因子分数图')

# with st.expander("以下图形主要展示各因子暴露（因子数值）", expanded=False):
#     st.write("""

# - 各个因子数值中我们定”牛顶“数值为1，”熊底”数值为-1，其他为0


#     """)

st.caption('以下图形主要展示各因子数值，各个因子数值中定”牛顶“数值为1，”熊底”数值为-1，其他为0')

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
            for j, img_file in enumerate(img_files[i:i + per_row]):
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

st.caption(
    '300指数（剔除了金融股）的股息率与十年期国债收益率的比较。股息率是指股票的年度股息除以股价，而十年期国债收益率则是债券投资的回报率。图中展示了这两者的变化趋势。蓝色虚线表示平均值，+1倍和-1倍标准差，图示了股息率与国债收益率的相对收益率情况。(已增加上证综指)')

# with st.sidebar:
#     st.header("1.1 300指数股息率 / 十年国债·参数")
    
#     div_start = st.text_input("起始日(YYYYMMDD，可空)", value="", key="div_start_fixed")
#     div_end = st.text_input("结束日(YYYYMMDD，可空)", value="", key="div_end_fixed")
#     show_bands_11 = st.checkbox("显示均值与±1σ", value=True, key="div_bands_fixed")



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
        df = read_csv_default(
            "div_result_csv",
            required_cols=["trade_date", "weighted_dividend_rate", "sh_close", "nation10_yield", "weighted_dividend_rate_div_nation10"],
            parse_dates=["trade_date"]
        )


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
            try:
                df = df[df["trade_date"] >= pd.to_datetime(div_start, format="%Y%m%d")]
            except:
                st.warning("起始日格式应为 YYYYMMDD，已忽略。")
        if div_end:
            try:
                df = df[df["trade_date"] <= pd.to_datetime(div_end, format="%Y%m%d")]
            except:
                st.warning("结束日格式应为 YYYYMMDD，已忽略。")
        if df.empty:
            st.warning("过滤后无数据。");
            st.stop()

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
      
          
        fig.add_trace(go.Scatter(x=df["trade_date"], y=[mu] * len(df),
                                     mode="lines", name="均值", line=dict(dash="dot"), yaxis="y1"))
        fig.add_trace(go.Scatter(x=df["trade_date"], y=[mu + sd] * len(df),
                                     mode="lines", name="均值+1σ", line=dict(dash="dash"), yaxis="y1"))
        fig.add_trace(go.Scatter(x=df["trade_date"], y=[mu - sd] * len(df),
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
                data=df[["trade_date", "weighted_dividend_rate", "nation10_yield",
                         "weighted_dividend_rate_div_nation10", "sh_close", "ratio"]]
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

st.caption(
    '本图展示了A股的风险溢价（E/P，即市盈率倒数）与十年期国债收益率的时序关系。风险溢价表示股票市场的收益率相对于国债的超额回报，E/P 值越高，表示股票的风险溢价越高。通过计算市盈率倒数和股票的市值加权，得出加权的风险溢价。图中展示了A股风险溢价随时间的变化趋势，以及其相对于十年期国债收益率的波动范围。蓝色虚线表示风险溢价的均值，+1倍和-1倍标准差，图示了股市的波动性和国债收益率的变化关系。')

# 侧边栏参数
with st.sidebar:
    st.header("1.2 全A E/P(市盈率倒数)−十年期国债·参数")
    
    ep_start = st.text_input("起始日(YYYYMMDD，可空)", value="", key="ep_start")
    ep_end = st.text_input("结束日(YYYYMMDD，可空)", value="", key="ep_end")
    ep_clip = st.checkbox("1%/99% 去极值", value=True, key="ep_clip")
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

btn_ep = st.button("生成图表", type="primary", key="ep_btn")
if btn_ep:
    try:
        epdf = read_csv_default(
            "div_result_csv2",
            required_cols=["trade_date", "weighted_ep_10bond"],
            parse_dates=["trade_date"]
        )


        # 列/类型规范化（避免后续筛选/作图出错）
        if "trade_date" not in epdf.columns:
            st.error("CSV 缺少 trade_date 列");
            st.stop()
        if "weighted_ep_10bond" not in epdf.columns:
            st.error("CSV 缺少 weighted_ep_10bond 列");
            st.stop()

        epdf["trade_date"] = pd.to_datetime(epdf["trade_date"], errors="coerce")
        epdf = epdf.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        epdf["weighted_ep_10bond"] = pd.to_numeric(epdf["weighted_ep_10bond"], errors="coerce")

        # 时间过滤
        if ep_start:
            try:
                epdf = epdf[epdf["trade_date"] >= pd.to_datetime(ep_start, format="%Y%m%d")]
            except:
                st.warning("起始日格式应为 YYYYMMDD，已忽略。")
        if ep_end:
            try:
                epdf = epdf[epdf["trade_date"] <= pd.to_datetime(ep_end, format="%Y%m%d")]
            except:
                st.warning("结束日格式应为 YYYYMMDD，已忽略。")

        if epdf.empty:
            st.warning("过滤后无数据。");
            st.stop()

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
            fig.add_trace(go.Scatter(x=epdf["trade_date"], y=[mu] * len(epdf),
                                     mode="lines", name="均值", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=epdf["trade_date"], y=[mu + sd] * len(epdf),
                                     mode="lines", name="均值+1σ", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=epdf["trade_date"], y=[mu - sd] * len(epdf),
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

st.caption(
    '跟踪“大盘vs小盘”等多空组合的收益和净值，在任意周期（日/月/季）上显示谁在持续占优，并用均值±σ告诉你这个风格强弱是否已经走到极端、有没有到该防风格反转/调仓的时点。')

# 侧边栏参数
with st.sidebar:
    st.header("2.1 大小盘轮动·参数")
    freq = st.radio("频率", ["日度", "月度", "季度"], index=1, key="rot_freq")
    view = st.radio("指标", ["收益", "净值(NAV)"], index=0, key="rot_view")
    k_sigma = st.number_input("±σ 带宽（σ倍数，仅收益视图有效）", 0.1, 3.0, 1.0, 0.1, key="rot_k")
    show_band = st.checkbox("显示均值与±σ（仅收益视图）", value=True, key="rot_band")
    date_start = st.text_input("起始日(YYYYMMDD，可空)", value="", key="rot_start")
    date_end = st.text_input("结束日(YYYYMMDD，可空)", value="", key="rot_end")



@st.cache_data(show_spinner=False)
def _load_df(path: str, idx_name: str):
    df = pd.read_csv(path)
    # 兼容：索引列可能已经叫 date/month/quarter，也可能就没明示
    # 尝试识别第1列为日期索引
    # 若文件有名为 idx_name 的列，用它；否则找第1列
    if idx_name in df.columns:
        dt = pd.to_datetime(df[idx_name], errors="coerce")
    else:
        dt = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df.insert(0, "dt", dt)
    df = df.dropna(subset=["dt"]).drop(columns=[c for c in ["date", "month", "quarter"] if c in df.columns],
                                       errors="ignore")
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
            out = out[out.index <= pd.to_datetime(end_str, format="%Y%m%d")]
        except:
            pass
    return out


def _mean_std(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s2.empty:
        return 0.0, 1.0
    return float(s2.mean()), float(s2.std(ddof=1)) or 1.0



# 读取基础数据（一定要用 _load_df 做索引列清洗）
daily_df   = _load_df(str(resolve_first_existing(get_path("rot_daily_csv"))),   idx_name="date")
month_df   = _load_df(str(resolve_first_existing(get_path("rot_month_csv"))),   idx_name="month")
quarter_df = _load_df(str(resolve_first_existing(get_path("rot_quarter_csv"))), idx_name="quarter")



# 确定可选组合
source_df = {"日度": daily_df, "月度": month_df, "季度": quarter_df}[freq]
if (source_df is None) or source_df.empty:
    # 回退：若只给了日度，而你选了月/季，则从日度聚合
    if (freq in ["月度", "季度"]) and (daily_df is not None) and (not daily_df.empty):
        agg_rule = "M" if freq == "月度" else "Q"
        ret_agg, nav_agg = _agg_compound(daily_df, agg_rule)
        source_df = pd.concat([ret_agg, nav_agg], axis=1)
    else:
        st.warning(f"未检测到 {freq} 数据且无法回退。请检查对应CSV是否存在。")
        st.stop()

# 拿出“收益列名”（非 _NAV）和“净值列名”（_NAV）
spread_cols = [c for c in source_df.columns if not c.endswith("_NAV")]
nav_cols = [c for c in source_df.columns if c.endswith("_NAV")]

default_choices = spread_cols[:2] if view == "收益" else nav_cols[:2]
choices = st.multiselect(
    f"选择要展示的{'收益组合' if view == '收益' else '净值组合'}（可多选）",
    options=(spread_cols if view == "收益" else nav_cols),
    default=default_choices,
    key="rot_chosen"
)

if not choices:
    st.info("请选择至少一个组合列。")
    st.stop()

# 时间过滤
plot_df = _clip_by_date(source_df[choices], date_start, date_end)
if plot_df.empty:
    st.warning("时间过滤后无数据。");
    st.stop()

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
    fig.add_trace(go.Scatter(x=plot_df.index, y=[mu] * len(plot_df), mode="lines",
                             name="均值", line=dict(dash="dot"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=[mu + k_sigma * sd] * len(plot_df), mode="lines",
                             name=f"+{k_sigma}σ", line=dict(dash="dash"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=[mu - k_sigma * sd] * len(plot_df), mode="lines",
                             name=f"-{k_sigma}σ", line=dict(dash="dash"), yaxis="y1"))

fig.update_layout(
    template="plotly_dark",
    height=560,
    legend=dict(orientation="h", x=0, y=1.12),
    margin=dict(l=60, r=70, t=40, b=40),
    xaxis=dict(title="日期"),
    yaxis=dict(title=("收益" if view == "收益" else "净值"), side="left"),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("下载当前视图数据"):
    st.download_button("下载CSV", data=plot_df.reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"大小盘_{freq}_{'收益' if view == '收益' else '净值'}.csv", mime="text/csv")

# ======================= 2.2 行业拥挤度=======================
import pandas as pd
from PIL import Image

# st.markdown("---")
# st.subheader("2.2 行业拥挤度")

# @st.cache_data(show_spinner=False)

# --- 侧边栏：2.2 时间区间参数 ---
with st.sidebar:
    st.header("2.2图1行业成交额占比·参数")
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

import plotly.graph_objects as go


def _parse_ymd(s: str):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False)
def load_industry_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 你的日期列叫 index（字符串/数字都可能），这里统一转 datetime
    if "index" not in df.columns:
        raise ValueError(f"industry_daily_all 缺少 index 列，实际列：{list(df.columns)[:10]}...")
    df["index"] = pd.to_datetime(df["index"], errors="coerce")
    df = df.dropna(subset=["index"]).sort_values("index").drop_duplicates("index", keep="last")

    # 数值列统一转数值
    for c in df.columns:
        if c != "index":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def calc_industry_share_bar(df_ind: pd.DataFrame, start_str: str, end_str: str):
    s_dt = _parse_ymd(start_str)
    e_dt = _parse_ymd(end_str)

    d = df_ind.copy()
    if s_dt is not None:
        d = d[d["index"] >= s_dt]
    if e_dt is not None:
        d = d[d["index"] <= e_dt]

    if d.empty:
        return pd.DataFrame(columns=["industry", "amt_sum", "share"]), None

    # 选出所有行业的 *_amt 列（排除 total_amt / tmt_amt）
    amt_cols = [c for c in d.columns if c.endswith("_amt") and c not in ["total_amt"]]
    if not amt_cols:
        raise ValueError("未找到任何 *_amt 行业列（已排除 total_amt")

    total = float(d["total_amt"].sum()) if "total_amt" in d.columns else float(d[amt_cols].sum().sum())
    if total == 0 or np.isnan(total):
        total = 1.0

    amt_sum = d[amt_cols].sum(axis=0).sort_values(ascending=False)  # 按行加总
    out = pd.DataFrame({
        "industry": [c[:-4] for c in amt_sum.index],  # 去掉 _amt
        "amt_sum": amt_sum.values,
    })
    out["share"] = out["amt_sum"] / total
    return out, (s_dt, e_dt)


industry_daily_path = get_path("industry_daily_all")
col_left, col_right = st.columns(2)


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
st.subheader("2.2 行业总成交额占比 / 行业拥挤度 / 行业热度榜")

st.markdown("""
<div style="line-height: 1.4; color: #808080;">
本节数据可通过选择侧边栏时间区间、及行业查看统计数据展示。

    图1：行业总成交额占比
    统计整个时间区间内各行业累计成交额在全市场中的比例，用来观察长期资金关注度。占比越高的行业，说明在时间区间里交易最活跃、资金集中度最高。

    图2：行业拥挤度
    计算各行业的每日成交额占比，反映短期市场热度。曲线上升说明资金流入某个板块加快，下降则表示热度减退。

    表3：行业热度榜（最近一周）
    展示时间区间中最近一周各行业的成交额占比、分位数和平均换手率，用于识别“拥挤/冷清”的方向。

</div>
""", unsafe_allow_html=True)

# 线下输出产物路径（保持你原来的路径）
# img_path_industry = get_path("industry_img1")
img_path_tmt = get_path("industry_img2")
xlsx_path_table = get_path("industry_table_xlsx")

col_left, col_right = st.columns(2)

# 图1：行业期间成交占比柱图
with col_left:
    st.markdown("图1：行业成交额占比")
    st.caption(f"区间：{_fmt_range(crowd_start, crowd_end)}")

    try:
        p = resolve_first_existing(industry_daily_path)
        if p is None:
            st.warning(f"找不到 industry_daily_alls 文件：{industry_daily_path}")
        else:
            df_ind = load_industry_daily(str(p))
            bar_df, _range = calc_industry_share_bar(df_ind, crowd_start, crowd_end)

            if bar_df.empty:
                st.info("该区间内无数据。")
            else:
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(
                    x=bar_df["industry"],
                    y=bar_df["share"] * 100.0,
                    name="成交额占比(%)"
                ))
                fig1.update_layout(
                    template="plotly_dark",
                    height=520,
                    margin=dict(l=40, r=20, t=40, b=120),
                    xaxis=dict(title="行业", tickangle=-60),
                    yaxis=dict(title="成交额占比（%）")
                )
                st.plotly_chart(fig1, use_container_width=True)

                # with st.expander("下载图1数据"):
                #     st.download_button(
                #         "下载 CSV（行业区间成交额占比）",
                #         data=bar_df.to_csv(index=False).encode("utf-8-sig"),
                #         file_name="2.2_图1_行业成交额占比_区间汇总.csv",
                #         mime="text/csv"
                #     )
                st.download_button(
                    "下载 CSV（行业时间区间内成交额占比）",
                    data=bar_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="2.2_图1_行业成交额占比_区间汇总.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.warning(f"图1生成失败：{type(e).__name__}: {e}")


# ====== 图2：行业拥挤度（可选行业/区间/MA） ======

def available_industries_from_df(df_ind: pd.DataFrame):
    # 从 *_pct 列提取行业名，排除 tmt_pct（它是组合项）和 index
    pct_cols = [c for c in df_ind.columns if c.endswith("_pct")]
    # pct_cols = [c for c in pct_cols if c not in ["tmt_pct"]]  # 可留可不留
    inds = [c[:-4] for c in pct_cols]  # 去掉 _pct
    return sorted(set(inds))


def calc_multi_industry_lines(df_ind: pd.DataFrame, industries: list, start_str: str, end_str: str, win: int = 5):
    s_dt = _parse_ymd(start_str)
    e_dt = _parse_ymd(end_str)

    d = df_ind.copy()
    if s_dt is not None:
        d = d[d["index"] >= s_dt]
    if e_dt is not None:
        d = d[d["index"] <= e_dt]
    if d.empty:
        return pd.DataFrame()

    # 选出这些行业对应的 pct 列
    pct_cols = []
    for ind in industries:
        c = f"{ind}_pct"
        if c in d.columns:
            pct_cols.append(c)

    if not pct_cols:
        raise ValueError("所选行业在 df_ind 中找不到对应的 *_pct 列")

    out = d[["index"] + pct_cols].copy()

    # 每条线分别做 MA
    for c in pct_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c + f"_MA{win}"] = out[c].rolling(win, min_periods=1).mean()

    return out


# --- 侧边栏：2.2 时间区间参数 ---
# ===== 1. 选择行业 & MA窗口 =====
all_inds = sorted({c[:-4] for c in df_ind.columns if c.endswith("_pct")})
default_inds = [x for x in ["电子", "通信", "传媒", "计算机"] if x in all_inds]
with st.sidebar:
    st.header("2.2图2行业拥挤度曲线·参数")
    crowd_start2 = st.text_input(
        "起始日(YYYYMMDD，可空)",
        value="",
        key="crowd_start2"
    )
    crowd_end2 = st.text_input(
        "结束日(YYYYMMDD，可空)",
        value="",
        key="crowd_end2"
    )
    sel_inds = st.multiselect(
        "选择行业（每个行业一条线）",
        options=all_inds,
        default=default_inds,
        key="chart2_inds_no_sum"
    )

# --- 图2绘制 ---
with col_right:
    st.markdown("图2：行业拥挤度")
    st.caption(f"区间：{_fmt_range(crowd_start2, crowd_end2)}")

    if not default_inds:
        default_inds = all_inds[:3]

    # ma_win = st.number_input(
    #     "MA窗口",
    #     min_value=1, max_value=60, value=5, step=1,
    #     key="chart2_ma_no_sum"
    # )

    if not sel_inds:
        st.info("请至少选择一个行业")
    else:
        # ===== 2. 时间区间过滤 =====
        d = df_ind.copy()
        s_dt = _parse_ymd(crowd_start2)
        e_dt = _parse_ymd(crowd_end2)

        if s_dt is not None:
            d = d[d["index"] >= s_dt]
        if e_dt is not None:
            d = d[d["index"] <= e_dt]

        if d.empty:
            st.info("该区间内无数据")
        else:
            # ===== 3. 画图：每个行业一条线（各自 MA） =====
            fig2 = go.Figure()

            for ind in sel_inds:
                col = f"{ind}_pct"
                if col not in d.columns:
                    continue

                y = pd.to_numeric(d[col], errors="coerce")
                # y_ma = y.rolling(int(ma_win), min_periods=1).mean()

                fig2.add_trace(go.Scatter(
                    x=d["index"],
                    # y=y_ma,
                    y=y,
                    mode="lines",
                    name=ind
                ))

            fig2.update_layout(
                template="plotly_white",  # 和你截图风格一致
                height=520,
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis=dict(title="日期"),
                # yaxis=dict(title=f"成交额占比 MA{ma_win}（%）"),
                yaxis=dict(title=f"成交额占比（%）"),
                legend=dict(orientation="h", x=0, y=1.12)
            )

            st.plotly_chart(fig2, use_container_width=True)

            st.download_button(
                "下载 CSV（行业时间区间内拥挤度）",
                data=d.to_csv(index=False).encode("utf-8-sig"),
                file_name="2.2_图2_行业时间区间内拥挤度.csv",
                mime="text/csv"
            )

# 表3：行业热度榜
import pandas as pd
import streamlit as st

# 假设这个是数据路径
industry_data_path = get_path("industry_weekly_all")


def load_industry_data(path):
    """读取并加载行业数据"""
    df = pd.read_csv(path)
    df['week'] = pd.to_datetime(df['week'], errors='coerce')
    return df.dropna(subset=['week'])


# 加载数据
df_industry = load_industry_data(industry_data_path)

# 侧边栏：时间区间选择
with st.sidebar:
    st.header("表3：行业热度榜（最近一周）")
    crowd_start3 = st.text_input("起始日(YYYYMMDD，可空)", value="", key="crowd_start3")
    crowd_end3 = st.text_input("结束日(YYYYMMDD，可空)", value="", key="crowd_end3")


# 计算行业热度榜（成交额占比、分位数、换手率）
def calc_industry_heat(df, start_date, end_date):
    # 筛选数据
    if start_date:
        df = df[df['week'] >= pd.to_datetime(start_date, format="%Y%m%d")]
    if end_date:
        df = df[df['week'] <= pd.to_datetime(end_date, format="%Y%m%d")]

    df['amt_pct_rank'] = df.groupby('industry')['amt_pct'].rank(pct=True)
    df['tovr_pct_rank'] = df.groupby('industry')['industry_avg_turnover'].rank(pct=True) * 100
    latest_week = df['week'].max()
    df_latest = df[df['week'] == latest_week].copy()

    return df_latest


# 获取行业热度榜数据
industry_heat_df = calc_industry_heat(df_industry, crowd_start3, crowd_end3)

# 显示行业热度榜
st.subheader("行业热度榜")
st.dataframe(industry_heat_df)

# 导出数据按钮
st.download_button(
    "下载行业热度榜 (Excel)",
    data=industry_heat_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="industry_heat_table.csv",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

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
    
    dt_start_31 = st.text_input("起始日(YYYYMMDD，可空)", value="", key="std31_start")
    dt_end_31 = st.text_input("结束日(YYYYMMDD，可空)", value="", key="std31_end")
    agg_rule_31 = st.selectbox("可选聚合频率", ["不聚合(逐日)", "月度", "季度"], index=0, key="std31_agg")
    smooth_win_31 = st.number_input("平滑窗口（移动平均，期）", 1, 60, 3, 1, key="std31_smooth")



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

    #def _agg_one(g):
        #g = g.set_index("trade_date").sort_index()
        #out = g.resample(res_rule).mean(numeric_only=True).reset_index()
        #for c in ["scope", "level1_industry_name"]:
            #out[c] = g[c].iloc[-1] if not g.empty else None
        #return out[["trade_date", "scope", "level1_industry_name", "turn_daily_std"]]

    def _agg_one(g):
        g = g.set_index("trade_date").sort_index()

     # 只取数值列 resample（最稳）
        num = g[["turn_daily_std"]].apply(pd.to_numeric, errors="coerce")
        out = num.resample(res_rule).mean().reset_index()

    # 补回标签列（每个聚合桶用最后一个标签）
        out["scope"] = g["scope"].iloc[-1] if not g.empty else None
        out["level1_industry_name"] = g["level1_industry_name"].iloc[-1] if not g.empty else None

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

st.markdown("""
<div style="line-height: 1.4; color: #808080;">
本页汇总：时间过滤、按日/月/季聚合、平滑窗口、单位切换（小数/百分比），并把全A与行业统一对比。

</div>
""", unsafe_allow_html=True)

# 读取 CSV
df31_raw = read_csv_default(
    "turn_std_csv",
    required_cols=["trade_date", "scope", "level1_industry_name", "turn_daily_std"],
    parse_dates=["trade_date"]
)
df31_raw["turn_daily_std"] = pd.to_numeric(df31_raw["turn_daily_std"], errors="coerce")


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
        df31.loc[df31["scope"] == "行业", "level1_industry_name"].dropna().unique().tolist()
    )
    has_allA = (df31["scope"] == "全A").any()
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
                g = df31[df31["scope"] == "全A"].sort_values("trade_date")
            else:
                g = df31[(df31["scope"] == "行业") & (df31["level1_industry_name"] == name)].sort_values("trade_date")
            if g.empty:
                continue
            y = _smooth_series(g["turn_std_view"], smooth_win_31)
            fig.add_trace(go.Scatter(x=g["trade_date"], y=y, mode="lines", name=str(name)))
            tmp = g[["trade_date", "turn_std_view"]].copy()
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






