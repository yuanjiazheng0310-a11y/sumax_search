# -*- coding: utf-8 -*-
# @Time : 2026-06-02
# @Author : ZJY
# @File : 版本2
# @Software : PyCharm
# @Description ： 汽配查询平台 (性能优化版：预计算+向量化)

import streamlit as st
import pandas as pd
import os
import base64
import json
import time
from datetime import datetime
from filelock import FileLock

# ==========================================
# 🛠️ 1. 核心模块：IP 访问记录 (线程安全版)
# ==========================================
VISIT_LOG_FILE = "visits.json"
LOCK_FILE = "visits.json.lock"


def get_client_ip():
    """获取访问者的 IP 地址"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.runtime import get_instance
        ctx = get_script_run_ctx()
        if ctx is None:
            return "127.0.0.1"
        session_info = get_instance().get_client(ctx.session_id)
        if session_info is None:
            return "127.0.0.1"
        request = session_info.request
        ip = request.headers.get("X-Real-Ip")
        if not ip:
            ip = request.headers.get("X-Forwarded-For")
        if not ip:
            ip = request.remote_ip
        return ip
    except Exception as e:
        return "127.0.0.1"


def record_visit(ip):
    """记录访问并返回统计信息 (线程安全)"""
    lock = FileLock(LOCK_FILE)
    with lock:
        visit_data = {}
        today_str = datetime.now().strftime("%Y-%m-%d")

        if os.path.exists(VISIT_LOG_FILE):
            try:
                with open(VISIT_LOG_FILE, 'r', encoding='utf-8') as f:
                    visit_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                st.warning("访问记录文件读取异常，正在重建...")
                visit_data = {}

        if ip not in visit_data:
            visit_data[ip] = {"total": 0, "days": {}}

        visit_data[ip]["total"] += 1
        if today_str not in visit_data[ip]["days"]:
            visit_data[ip]["days"][today_str] = 0
        visit_data[ip]["days"][today_str] += 1

        try:
            with open(VISIT_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(visit_data, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"保存访问记录失败: {e}")

        total_count = visit_data[ip]["total"]
        today_count = visit_data[ip]["days"][today_str]
        print(f"🌐 访问日志：IP [{ip}] 今日第 {today_count} 次 | 历史总计 {total_count} 次")
        return total_count, today_count


# ==========================================
# 🖼️ 2. 背景图片配置与处理
# ==========================================
BACKGROUND_IMAGE_FILE = "跑车.jpg"


def get_base64_of_image(file_path):
    """读取图片并转换为 base64 字符串"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"图片读取错误: {e}")
        return None


# ==========================================
# 📊 3. 数据加载与预处理（优化核心）
# ==========================================
@st.cache_data
def load_data():
    """加载原始数据"""
    data = {}
    try:
        file_path_1 = os.path.join(current_dir, "1.parquet")
        file_path_2 = os.path.join(current_dir, "2.parquet")

        if os.path.exists(file_path_1):
            df_kit = pd.read_parquet(file_path_1)
            df_kit.columns = df_kit.columns.str.strip()
            if "套装编号" in df_kit.columns and "配件编号" in df_kit.columns:
                df_kit["套装编号"] = df_kit["套装编号"].astype(str).str.strip()
                df_kit["配件编号"] = df_kit["配件编号"].astype(str).str.strip()
            data["kit"] = df_kit

        if os.path.exists(file_path_2):
            df_part = pd.read_parquet(file_path_2)
            df_part.columns = df_part.columns.str.strip()
            if "零件编号" not in df_part.columns:
                df_part["零件编号"] = ""
            if "OEM编号" not in df_part.columns:
                df_part["OEM编号"] = ""
            df_part["OEM编号"] = df_part["OEM编号"].astype(str).str.strip()
            df_part["零件编号"] = df_part["零件编号"].astype(str).str.strip()
            df_part = df_part[df_part["OEM编号"] != "nan"]
            df_part = df_part[df_part["OEM编号"] != ""]
            data["part"] = df_part
        return data
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return data


@st.cache_data
def preprocess_part_data(df_part):
    """预处理零件数据，预计算所有需要的结构（性能优化关键）"""
    if df_part.empty:
        return {}, pd.DataFrame()

    df_part = df_part.copy()

    # 1. 添加OEM长度列
    df_part["oem_len"] = df_part["OEM编号"].str.len()

    # 2. 按长度分组（用于快速筛选）
    len_groups = {}
    for length in df_part["oem_len"].unique():
        len_groups[length] = df_part[df_part["oem_len"] == length]

    # 3. 预构建精确匹配字典
    oem_to_parts = df_part.groupby("OEM编号")["零件编号"].apply(list).to_dict()

    # 4. 预构建OEM长度映射
    oem_len_map = df_part.set_index("OEM编号")["oem_len"].to_dict()

    return {
        "len_groups": len_groups,
        "oem_to_parts": oem_to_parts,
        "oem_len_map": oem_len_map,
        "df_part": df_part
    }, df_part


def clean_oem(oem_str):
    """清理OEM编号"""
    return oem_str.replace("-", "").replace(" ", "").replace(".", "")


def check_limit(text, limit=100):
    """检查输入行数限制"""
    lines = [x for x in text.strip().split("\n") if x.strip()]
    return len(lines)


def format_search_results(results_df, success_msg=None):
    """统一的结果展示格式"""
    if results_df.empty:
        st.warning("未找到任何结果。")
        return False
    else:
        if success_msg:
            st.success(success_msg)
        else:
            st.success(f"找到 {len(results_df)} 条结果！")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        return True


# ==========================================
# 🚀 4. 页面主程序
# ==========================================
# --- 页面配置 ---
st.set_page_config(
    page_title="汽配套装平台_查询模块(更新时间：26年08_26)",
    page_icon="📦",
    layout="wide"
)

# --- 注入 CSS 样式 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, BACKGROUND_IMAGE_FILE)
img_base64 = get_base64_of_image(img_path)

if img_base64:
    bg_css = f"""
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    """
else:
    bg_css = """
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2b55 100%);
    }
    """

transparent_css = f"""
<style>
{bg_css}
.stApp {{ background-color: transparent; }}
[data-testid="stSidebar"] {{ background-color: rgba(0, 0, 0, 0.6); backdrop-filter: blur(5px); }}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {{
    color: #888888 !important;
    opacity: 1;
}}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background-color: #FFFFFF !important;
    color: #000000 !important;
}}
.stButton > button {{
    background-color: #FF4B4B;
    color: white;
    border-radius: 5px;
}}
.stTextArea textarea {{
    resize: vertical !important;
    min-height: 100px;
    max-height: 500px;
}}
</style>
"""
st.markdown(transparent_css, unsafe_allow_html=True)

# --- 加载数据 ---
data_sources = load_data()
df_kit = data_sources.get("kit", pd.DataFrame())
df_part = data_sources.get("part", pd.DataFrame())

# --- 预处理零件数据（性能优化）---
precomputed_data, df_part_processed = preprocess_part_data(df_part)
len_groups = precomputed_data.get("len_groups", {})
oem_to_parts = precomputed_data.get("oem_to_parts", {})
oem_len_map = precomputed_data.get("oem_len_map", {})

# ==========================================
# 📝 5. 侧边栏
# ==========================================
with st.sidebar:
    st.header("📊 SUMAX查询平台📊")
    st.header("📦注意：为了流畅使用/服务器需要重启和刷新缓存-请在早上9点后使用")
    ip_placeholder = st.empty()
    today_placeholder = st.empty()
    total_placeholder = st.empty()
    st.markdown("---")

    if not df_kit.empty:
        unique_kit_count = df_kit.iloc[:, 0].nunique()
        st.metric("总套装记录数📦", unique_kit_count)

    if not df_part.empty:
        unique_part_count = df_part.iloc[:, 0].nunique()
        st.metric("含有oem产品总数📦", unique_part_count)

    st.markdown("---")
    st.info("💡 提示1：输入框现已支持鼠标拖拽调整高度。")
    st.info("💡 提示2：智能识别 OEM 编号，自动忽略空格、横杠及点号。")
    st.warning("⚠️ 限制：单次查询最多支持 **100 条** 数据/oem查询限制1000行。")

# ==========================================
# 🔍 6. 主界面：套装查询
# ==========================================
st.subheader("🔧 套装查询 ")
col1_kit, col2_kit = st.columns([3, 1])

with col1_kit:
    search_text_kit = st.text_area(
        "🔍 输入配件编号",
        placeholder="支持批量查询，请换行分隔：\nTC1445\nTG1891",
        key="kit_input",
        label_visibility="visible",
    )

with col2_kit:
    mode_kit = st.radio(
        "匹配模式:",
        ["精确匹配 (严格)", "包含匹配 (宽泛)"],
        horizontal=False,
        key="kit_mode"
    )

if st.button("📦 查询套装", type="primary"):
    current_ip = get_client_ip()
    total_visits, today_visits = record_visit(current_ip)

    with st.sidebar:
        today_placeholder.success(f"📅 今日访问：第 **{today_visits}** 次")

    if not df_kit.empty and search_text_kit.strip():
        line_count = check_limit(search_text_kit, 100)
        if line_count > 100:
            st.error(f"❌ 输入数据过多！当前 {line_count} 行，单次查询不得超过 100 条。")
            st.stop()

        query_parts = [x.strip() for x in search_text_kit.strip().split("\n") if x.strip()]
        query_set = set(query_parts)
        results = []
        grouped = df_kit.groupby("套装编号")["配件编号"].apply(set).reset_index()
        grouped.columns = ["套装编号", "kit_parts_set"]

        if "包含匹配" in mode_kit:
            for _, row in grouped.iterrows():
                kit_parts = row["kit_parts_set"]
                if query_set.issubset(kit_parts):
                    results.append({
                        "套装编号": row["套装编号"],
                        "匹配零件数": len(query_set),
                        "套装总零件数": len(kit_parts),
                    })
        else:
            for _, row in grouped.iterrows():
                kit_parts = row["kit_parts_set"]
                if kit_parts == query_set:
                    results.append({
                        "套装编号": row["套装编号"],
                        "匹配零件数": len(query_set),
                    })

        if results:
            st.success(f"找到 {len(results)} 个符合条件的套装！")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("未找到匹配的套装。")
    elif df_kit.empty:
        st.error("套装数据未加载，请检查 Parquet 文件。")
    else:
        st.warning("请输入配件编号。")

st.markdown("---")

# ==========================================
# 🔎 7. 主界面：OEM 零件查询（优化版）
# ==========================================
st.subheader("🔍 OEM 零件查询 ")
col1_oem, col2_oem = st.columns([3, 1])

with col1_oem:
    search_text_oem = st.text_area(
        "🔍 输入 OEM 编号",
        placeholder="支持批量查询，请换行分隔：\n38810\n926003",
        key="oem_input",
        label_visibility="visible",
    )

with col2_oem:
    mode_oem = st.radio(
        "匹配模式:",
        ["精确匹配", "宽泛匹配 (包含+长度差≤1)"],
        horizontal=False,
        key="oem_mode",
        help="宽泛匹配规则：\n1. 字符串互相包含\n2. 长度差不能超过 1 位"
    )

if st.button("⚙️ 查询零件", type="primary"):
    # 记录查询开始时间（性能监控）
    start_time = time.time()

    # 1. 获取 IP 并记录
    current_ip = get_client_ip()
    total_visits, today_visits = record_visit(current_ip)

    with st.sidebar:
        today_placeholder.success(f"📅 今日访问：第 **{today_visits}** 次")

    # 2. 执行查询逻辑
    if not df_part.empty and search_text_oem.strip():
        line_count = check_limit(search_text_oem, 1000)
        if line_count > 1000:
            st.error(f"❌ 输入数据过多！当前 {line_count} 行，单次查询不得超过 1000 条。")
            st.stop()

        # 清理查询数据（使用列表推导式）
        cleaned_queries = [
            clean_oem(line.strip())
            for line in search_text_oem.strip().split("\n")
            if line.strip()
        ]
        cleaned_queries = [q for q in cleaned_queries if q]  # 过滤空字符串

        if mode_oem == "精确匹配":
            # ========== 精确匹配模式：合并单元格展示 ==========
            final_results = []
            for idx, q_oem in enumerate(cleaned_queries, 1):
                matches = oem_to_parts.get(q_oem, [])
                if matches:
                    unique_matches = list(set(matches))
                    final_results.append({
                        "序号": idx,
                        "查询OEM": q_oem,
                        "零件OEM": q_oem,
                        "结果零件编号": ",".join(unique_matches),
                        "匹配模式": "精准匹配"
                    })
                else:
                    final_results.append({
                        "序号": idx,
                        "查询OEM": q_oem,
                        "零件OEM": q_oem,
                        "结果零件编号": "未找到",
                        "匹配模式": "无"
                    })

            if final_results:
                df_results = pd.DataFrame(final_results)
                format_search_results(df_results,
                                      f"找到 {len([r for r in final_results if r['匹配模式'] != '无'])} 个有结果的查询")
            else:
                st.warning("未找到任何结果。")

        else:
            # ========== 宽泛匹配模式：每个匹配结果单独一行（优化版）==========
            results = []
            idx = 1

            for q_oem in cleaned_queries:
                q_len = len(q_oem)

                # 优化：只检查长度差<=1的长度组
                candidate_dfs = []
                for length in [q_len - 1, q_len, q_len + 1]:
                    if length in len_groups:
                        candidate_dfs.append(len_groups[length])

                if not candidate_dfs:
                    # 没有候选数据
                    results.append({
                        "序号": idx,
                        "查询OEM": q_oem,
                        "零件OEM": "未找到",
                        "结果零件编号": "未找到",
                        "匹配模式": "无",
                    })
                    idx += 1
                    continue

                # 合并候选数据
                candidates = pd.concat(candidate_dfs, ignore_index=True)

                # 分离精确匹配和模糊匹配
                exact_mask = candidates["OEM编号"] == q_oem
                exact_matches = candidates[exact_mask]
                fuzzy_candidates = candidates[~exact_mask]

                temp_results = []

                # 精确匹配结果
                if not exact_matches.empty:
                    for _, row in exact_matches.iterrows():
                        temp_results.append({
                            "OEM编号": row["OEM编号"],
                            "零件编号": row["零件编号"],
                            "匹配模式": "精准匹配"
                        })

                # 模糊匹配结果（使用向量化操作）
                if not fuzzy_candidates.empty:
                    # str.contains 是向量化操作
                    contains_db = fuzzy_candidates["OEM编号"].str.contains(q_oem, regex=False, na=False)
                    # 反向包含判断
                    # contains_q = fuzzy_candidates["OEM编号"].apply(lambda x: q_oem in x)
                    contains_q = fuzzy_candidates["OEM编号"].apply(lambda x: x in q_oem)
                    fuzzy_mask = contains_db | contains_q
                    fuzzy_matches = fuzzy_candidates[fuzzy_mask]

                    if not fuzzy_matches.empty:
                        for _, row in fuzzy_matches.iterrows():
                            temp_results.append({
                                "OEM编号": row["OEM编号"],
                                "零件编号": row["零件编号"],
                                "匹配模式": "模糊匹配"
                            })

                # 去重并添加到结果
                if temp_results:
                    seen = set()
                    for item in temp_results:
                        key = (item["OEM编号"], item["零件编号"])
                        if key not in seen:
                            results.append({
                                "序号": idx,
                                "查询OEM": q_oem,
                                "零件OEM": item["OEM编号"],
                                "结果零件编号": item["零件编号"],
                                "匹配模式": item["匹配模式"],
                            })
                            seen.add(key)
                    idx += 1
                else:
                    results.append({
                        "序号": idx,
                        "查询OEM": q_oem,
                        "零件OEM": "未找到",
                        "结果零件编号": "未找到",
                        "匹配模式": "无",
                    })
                    idx += 1

            if results:
                df_results = pd.DataFrame(results)
                format_search_results(df_results)
            else:
                st.warning("未找到任何结果。")

        # 性能监控输出（开发调试用）
        elapsed_time = time.time() - start_time
        st.caption(f"⚡ 查询耗时：{elapsed_time:.3f} 秒")

    elif df_part.empty:
        st.error("零件数据未加载，请检查 Parquet 文件。")
    else:
        st.warning("请输入 OEM 编号。")

# ==========================================
# 🔄 8. 文本处理工具箱
# ==========================================
st.subheader("🧰 文本处理工具箱")
tab_multi_to_single, tab_single_to_multi = st.tabs(["🔄 多行转单行", "⬇️ 单行转多行"])

# --- 选项卡 1：多行转单行 ---
with tab_multi_to_single:
    st.markdown("将多行文本合并为一行，使用英文逗号 `,` 作为分隔符。")
    col_input_1, col_output_1 = st.columns([3, 1])
    with col_input_1:
        multi_line_input = st.text_area(
            "📝 输入多行数据",
            placeholder="支持批量输入，请换行分隔：\nTC1445\nTG1891\n(最多100行)",
            key="multi_line_input",
            height=150
        )
    if st.button("➡️ 转换为单行", type="secondary", key="btn_multi_to_single"):
        raw_text = multi_line_input.strip()
        if raw_text:
            all_lines = raw_text.split('\n')
            total_raw_lines = len(all_lines)
            if total_raw_lines > 100:
                st.error(f"❌ 输入行数过多！当前 {total_raw_lines} 行，单次转换不得超过 100 行。")
            else:
                valid_lines = [line.strip() for line in all_lines if line.strip()]
                if valid_lines:
                    result_line = ','.join(valid_lines)
                    st.text_area("✅ 转换结果", value=result_line, key="result_line_tab1", height=50)
                    st.success(f"转换成功！共处理 {len(valid_lines)} 个有效数据。")
                else:
                    st.warning("输入内容为空或没有有效数据行。")
        else:
            st.warning("请输入一些数据。")

# --- 选项卡 2：单行转多行 ---
with tab_single_to_multi:
    st.markdown("将用英文逗号 `,` 分隔的一行文本，拆分为多行显示。")
    col_input_2, col_output_2 = st.columns([3, 1])
    with col_input_2:
        single_line_input = st.text_area(
            "🔤 输入单行数据",
            placeholder="请在此输入用逗号分隔的数据，例如：\n38810,926003,TG1891",
            key="single_line_input",
            height=100
        )
    if st.button("⬇️ 转换为多行", type="secondary", key="btn_single_to_multi"):
        raw_text = single_line_input.strip()
        if raw_text:
            potential_parts = [part.strip() for part in raw_text.split(',')]
            total_parts = len(potential_parts)
            if total_parts > 1000:
                st.error(f"❌ 分割段数过多！当前检测到 {total_parts} 段，单次转换不得超过 1000 段。")
            else:
                valid_parts = [part for part in potential_parts if part]
                if valid_parts:
                    result_multi_line = '\n'.join(valid_parts)
                    st.text_area("✅ 转换结果", value=result_multi_line, key="result_multi_line_tab2", height=200)
                    st.success(f"转换成功！共拆分出 {len(valid_parts)} 行有效数据。")
                else:
                    st.warning("输入内容为空或没有有效数据段。")
        else:
            st.warning("请输入一些数据。")

# 数据预览
with st.expander("查看原始数据预览"):
    if not df_part.empty:
        st.write("**零件 OEM 数据 (2.parquet):**")
        st.dataframe(df_part.head(5))