# =========================
# EPDSLL 风险预测临床应用 - Streamlit（部署版）
# 适配：SVM TopK9 + 约登阈值风险分层 + SHAP 力图解释
# 结构：资源文件放在仓库根目录（与你当前仓库一致）
#   - svm_topk9_deploy_res.joblib
#   - final_top9_vars.json（可选：兜底）
# =========================

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt


# -------------------------
# 1) 页面基础配置
# -------------------------
st.set_page_config(
    page_title="孕晚期风险预测工具",
    page_icon="🏥",
    layout="wide"
)
st.title("🏥 孕晚期 风险预测与临床解释工具")
st.markdown("### 基于 SVM TopK9 + 约登阈值风险分层 + SHAP 特征解释")
st.markdown("**输入以下 9 项特征，自动生成预测结果及特征贡献度分析**")
st.divider()


# -------------------------
# 2) 资源文件路径（与你仓库一致：根目录）
# -------------------------
RESOURCE_FILE = "svm_topk9_deploy_res.joblib"
TOP9_JSON = "final_top9_vars.json"

resource_path = os.path.join(os.path.dirname(__file__), RESOURCE_FILE)
top9_json_path = os.path.join(os.path.dirname(__file__), TOP9_JSON)


# -------------------------
# 3) 加载资源（缓存，避免每次刷新重复加载）
# -------------------------
@st.cache_resource(show_spinner=False)
def load_deploy_resource(path: str):
    return joblib.load(path)

def safe_load_top9_vars(json_path: str):
    """final_top9_vars.json 作为兜底（deploy_res 没有 final_top9_vars 时用）"""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if isinstance(arr, list) and len(arr) > 0:
            return arr
    except Exception:
        pass
    return None


try:
    deploy_res = load_deploy_resource(resource_path)
except Exception as e:
    st.error("❌ 部署资源加载失败")
    st.code(f"路径：{resource_path}\n错误：{repr(e)}")
    st.info(
        "请检查：\n"
        "1）仓库根目录是否存在 svm_topk9_deploy_res.joblib\n"
        "2）文件名是否完全一致（大小写也要一致）\n"
        "3）是否缺少 requirements.txt 导致依赖未安装"
    )
    st.stop()

# 必须字段（final_top9_vars 允许用 json 兜底）
required_keys_min = ["best_model", "youden_threshold", "shap_background", "feature_type_info", "model_metrics"]
missing_min = [k for k in required_keys_min if k not in deploy_res]
if missing_min:
    st.error(f"❌ deploy_res 缺少关键字段：{missing_min}")
    st.info(f"当前 deploy_res keys：{list(deploy_res.keys())}")
    st.stop()

model = deploy_res["best_model"]                        # Pipeline：prep + clf（通常）
youden_thr = float(deploy_res["youden_threshold"])
shap_bg = deploy_res["shap_background"]
feat_type = deploy_res.get("feature_type_info", {})
model_metrics = deploy_res.get("model_metrics", {})

FINAL_TOP9_VARS = deploy_res.get("final_top9_vars", None)
if FINAL_TOP9_VARS is None:
    FINAL_TOP9_VARS = safe_load_top9_vars(top9_json_path)

if not FINAL_TOP9_VARS:
    st.error("❌ 无法获取 final_top9_vars（deploy_res中没有，且 final_top9_vars.json 也读取失败）")
    st.stop()

FINAL_TOP9_VARS = list(FINAL_TOP9_VARS)


# -------------------------
# 4) 侧边栏输入
# -------------------------
st.sidebar.header("📋 请输入特征")

# 前端显示 -> 训练编码（必须与你训练时一致）
edu2num = {"小学及以下": 1, "初中": 2, "高中/大专": 3, "本科及以上": 4}
pg2num = {"计划内": 0, "计划外": 1}
reac2num = {
    "无反应": 1,
    "正常妊娠反应（恶心呕吐）": 2,
    "不良妊娠反应（感冒、出血、严重恶心呕吐就医）": 3
}
hmi2num = {"10000以下": 1, "10001-20000": 2, "20000以上": 3}

# 你当前 top9 的输入项（按你之前界面写的）
input_data = {}

input_data["EPDSA"] = st.sidebar.number_input("EPDSA（孕早期EPDS分数）", min_value=0.0, max_value=30.0, value=0.0, step=0.1)
input_data["Insomnia"] = st.sidebar.number_input("Insomnia（失眠分数）", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
input_data["Anxiety"] = st.sidebar.number_input("Anxiety（焦虑分数）", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
input_data["GA"] = st.sidebar.number_input("GA（孕周）", min_value=0.0, max_value=42.0, value=0.0, step=0.1)
input_data["Capital"] = st.sidebar.number_input("Capital（社会资本分数）", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

input_data["reactions"] = st.sidebar.selectbox("reactions（妊娠反应）", list(reac2num.keys()), index=0)
input_data["Educational"] = st.sidebar.selectbox("Educational（学历等级）", list(edu2num.keys()), index=0)
input_data["PG"] = st.sidebar.selectbox("PG（妊娠计划）", list(pg2num.keys()), index=0)
input_data["HMI"] = st.sidebar.selectbox("HMI（家庭月收入）", list(hmi2num.keys()), index=0)

predict_btn = st.sidebar.button("🚀 开始预测", type="primary")


# -------------------------
# 5) shap_background 统一成原始9变量 DataFrame（优先解释整个 pipeline）
# -------------------------
def to_bg_dataframe(shap_bg_obj, columns):
    """
    尝试把背景数据变成 shape=(n, 9) 的 DataFrame，列名=FINAL_TOP9_VARS
    """
    if isinstance(shap_bg_obj, pd.DataFrame):
        if all(c in shap_bg_obj.columns for c in columns):
            return shap_bg_obj[columns].copy()
        if shap_bg_obj.shape[1] == len(columns):
            df = shap_bg_obj.copy()
            df.columns = columns
            return df

    arr = np.asarray(shap_bg_obj)
    if arr.ndim == 2 and arr.shape[1] == len(columns):
        return pd.DataFrame(arr, columns=columns)

    return None


# -------------------------
# 6) KernelExplainer 构建（缓存）
# -------------------------
@st.cache_resource(show_spinner=False)
def build_kernel_explainer_pipeline(_model, bg_df: pd.DataFrame):
    """解释整个 Pipeline：输入=原始9变量 DataFrame"""
    def predict_fn(df):
        return _model.predict_proba(df)[:, 1]
    return shap.KernelExplainer(predict_fn, bg_df)

@st.cache_resource(show_spinner=False)
def build_kernel_explainer_clf(_clf, bg_matrix: np.ndarray):
    """备用：解释 clf：输入=预处理后的矩阵"""
    def predict_fn(x):
        return _clf.predict_proba(x)[:, 1]
    return shap.KernelExplainer(predict_fn, bg_matrix)


# -------------------------
# 7) 开始预测
# -------------------------
if predict_btn:
    # 原始展示（用户可读）
    show_df = pd.DataFrame([input_data])
    st.subheader("📊 输入特征核对（原始显示）")
    st.dataframe(show_df, use_container_width=True)

    # 映射成训练编码
    input_df = pd.DataFrame([input_data])
    input_df["Educational"] = input_df["Educational"].map(edu2num)
    input_df["PG"] = input_df["PG"].map(pg2num)
    input_df["reactions"] = input_df["reactions"].map(reac2num)
    input_df["HMI"] = input_df["HMI"].map(hmi2num)

    # 强制列顺序 = 训练顺序（非常关键）
    input_df = input_df.reindex(columns=FINAL_TOP9_VARS)

    with st.spinner("正在预测并生成解释..."):
        # 预测概率
        pred_prob = float(model.predict_proba(input_df)[0, 1])

        # 风险分层
        if pred_prob >= youden_thr:
            risk_level = "🔴 高风险"
            clinical_suggestion = "临床建议：高风险，建议进一步评估、密切监测并及时干预。"
        else:
            risk_level = "🟢 低风险"
            clinical_suggestion = "临床建议：低风险，建议常规随访。"

        # -------------------------
        # SHAP 解释（优先：解释整个 Pipeline）
        # -------------------------
        shap_value_1d = None
        base_val = None
        features_for_plot = None
        feature_names_for_plot = None

        bg_df = to_bg_dataframe(shap_bg, FINAL_TOP9_VARS)

        if bg_df is not None:
            explainer = build_kernel_explainer_pipeline(model, bg_df)
            sv = explainer.shap_values(input_df, nsamples=100)
            sv = sv[0] if isinstance(sv, list) else sv

            shap_value_1d = np.array(sv).reshape(-1)
            base_val = float(np.array(explainer.expected_value).reshape(-1)[0])

            features_for_plot = input_df.iloc[0].values.reshape(-1)
            feature_names_for_plot = FINAL_TOP9_VARS

        else:
            # 备用：如果 shap_bg 不是(?,9)，很可能是“预处理后矩阵”
            preprocessor = getattr(model, "named_steps", {}).get("prep", None)
            clf = getattr(model, "named_steps", {}).get("clf", None)

            if preprocessor is None or clf is None:
                st.warning("⚠️ SHAP 解释不可用：模型未包含 'prep'/'clf' 结构，或背景数据形态不匹配。")
            else:
                X_in = preprocessor.transform(input_df)
                X_in = X_in.toarray() if hasattr(X_in, "toarray") else np.asarray(X_in)

                bg_mat = np.asarray(shap_bg)
                if bg_mat.ndim == 1:
                    bg_mat = bg_mat.reshape(1, -1)

                explainer = build_kernel_explainer_clf(clf, bg_mat)
                sv = explainer.shap_values(X_in, nsamples=100)
                sv = sv[0] if isinstance(sv, list) else sv

                shap_value_1d = np.array(sv).reshape(-1)
                base_val = float(np.array(explainer.expected_value).reshape(-1)[0])

                features_for_plot = np.array(X_in[0]).reshape(-1)

                # 尝试给预处理后的列命名
                try:
                    names = list(preprocessor.get_feature_names_out())
                    names = [n.split("__", 1)[1] if "__" in n else n for n in names]
                except Exception:
                    names = [f"f{i}" for i in range(features_for_plot.shape[0])]
                feature_names_for_plot = names

    # -------------------------
    # 8) 展示预测结果
    # -------------------------
    st.subheader("🎯 预测结果")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("EPDSLL 阳性概率", f"{pred_prob:.2%}", delta=f"约登阈值：{youden_thr:.2%}")

    with col2:
        auc_val = model_metrics.get("auc", model_metrics.get("AUC", None))
        auc_str = f"{float(auc_val):.3f}" if auc_val is not None else "N/A"
        st.metric("风险等级", risk_level, delta=f"模型AUC：{auc_str}")

    st.info(clinical_suggestion)
    st.divider()

    # -------------------------
    # 9) SHAP 力图
    # -------------------------
    st.subheader("🔍 特征贡献度解释（SHAP 力图）")
    st.markdown(
        "- **红色**：推动预测更偏向「EPDSLL阳性（高风险）」\n"
        "- **蓝色**：推动预测更偏向「EPDSLL阴性（低风险）」\n"
        "- **条越长**：影响越大"
    )

    if shap_value_1d is None or base_val is None:
        st.warning("当前无法生成 SHAP 力图（背景数据形态不匹配或模型结构不支持）。")
    else:
        fig = plt.figure(figsize=(12, 5))
        shap.force_plot(
            base_value=base_val,
            shap_values=shap_value_1d,
            features=features_for_plot,
            feature_names=feature_names_for_plot,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# -------------------------
# 10) 底部说明
# -------------------------
st.divider()
with st.expander("ℹ️ 工具说明", expanded=False):
    tpr = model_metrics.get("tpr_at_thr", model_metrics.get("TPR_at_thr_train", None))
    tnr = model_metrics.get("tnr_at_thr", model_metrics.get("TNR_at_thr_train", None))
    acc = model_metrics.get("accuracy", model_metrics.get("Accuracy", None))
    auc_val = model_metrics.get("auc", model_metrics.get("AUC", None))

    def fmt_pct(x):
        return f"{float(x):.2%}" if x is not None else "N/A"

    def fmt_float(x):
        return f"{float(x):.3f}" if x is not None else "N/A"

    st.markdown(
        f"1. 模型基础：SVM TopK9 特征构建；\n"
        f"2. 风险阈值：**{youden_thr:.2%}**（约登指数确定）；\n"
        f"3. 阈值平衡：灵敏度（TPR）={fmt_pct(tpr)}，特异性（TNR）={fmt_pct(tnr)}；\n"
        f"4. 模型性能：AUC={fmt_float(auc_val)}，准确率={fmt_float(acc)}；\n"
        f"5. 解释逻辑：SHAP 力图展示单个样本的特征贡献（仅供临床参考）。"
    )
