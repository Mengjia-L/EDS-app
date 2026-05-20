# -*- coding: utf-8 -*-
"""
Streamlit 应用：老年人认知障碍风险智能筛查评估系统

运行方式：
    streamlit run 智能筛查系统_问卷页面优化版.py

本版本在双模型 + 标准化 + 自定义题设基础上，重点优化：
1. 将问卷改为按模块纵向排列的页面；
2. 连续变量 NACCAGEB / EDUC / SMOKYRS 使用填空输入；
3. 其余离散变量使用单选题，并展示每个编码值的含义；
4. 尽量保持原有模型预测、标准化、SHAP、常模图等功能不变。
"""

import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler


FINAL_CLASS_ORDER = ["NC", "MCI", "AD", "nAD"]
COG_CLASS_ORDER = ["NC", "MCI", "DE"]
ADD_CLASS_ORDER = ["AD", "nAD"]

COG_LABEL_MAP = {
    0: "NC",
    1: "MCI",
    2: "DE",
    "0": "NC",
    "1": "MCI",
    "2": "DE",
    "NC": "NC",
    "MCI": "MCI",
    "DE": "DE",
    "DEM": "DE",
    "Dementia": "DE",
    "dementia": "DE",
}

ADD_LABEL_MAP = {
    0: "nAD",
    1: "AD",
    "0": "nAD",
    "1": "AD",
    "AD": "AD",
    "nAD": "nAD",
    "nADD": "nAD",
    "NAD": "nAD",
}

FINAL_LABEL_MAP = {
    "NC": "NC",
    "MCI": "MCI",
    "AD": "AD",
    "nAD": "nAD",
    "nADD": "nAD",
    "NAD": "nAD",
    0: "NC",
    1: "MCI",
    2: "AD",
    3: "nAD",
    "0": "NC",
    "1": "MCI",
    "2": "AD",
    "3": "nAD",
}

LABEL_DISPLAY_MAP = {
    "NC": "无认知障碍",
    "MCI": "轻度认知障碍",
    "DE": "疑似阿尔兹海默症痴呆",
    "AD": "阿尔兹海默症型痴呆",
    "nAD": "非阿尔兹海默症型痴呆",
}

EXCLUDED_COLUMNS = {"NACCID", "COG", "ADD"}
CONTINUOUS_FEATURES = {"NACCAGEB", "EDUC", "SMOKYRS"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COG_MODEL_PATH = os.path.join(BASE_DIR, "catboost_cog_model.cbm")
DEFAULT_ADD_MODEL_PATH = os.path.join(BASE_DIR, "catboost_add_model.cbm")
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "NACC数据_哑变量还原_特征筛选.xlsx")
DEFAULT_FEATURE_SOURCE_PATH = os.path.join(BASE_DIR, "NACC数据_哑变量还原_特征筛选.xlsx")
DEFAULT_SCALER_SOURCE_PATH = os.path.join(BASE_DIR, "X_train(μ,σ).xlsx")

REQUIRED_FEATURES: List[str] = [
    "SEX", "NACCAGEB", "EDUC", "SMOKYRS", "HYPERTEN", "HYPERCHO", "CBSTROKE",
    "TBI", "PD", "DEP2YRS", "TRAVEL", "REMDATES", "EVENTS", "TAXES", "MEALPREP", "STOVE",
    "GAMES", "PACKSPER", "TOBAC100", "DEPOTHR"
]

FEATURE_NAME_CONFIG: Dict[str, str] = {
    "NACCAGEB": "基线年龄",
    "EDUC": "受教育年限",
    "SMOKYRS": "吸烟年数",
    "EVENTS": "时事追踪能力",
    "GAMES": "游戏/爱好活动能力",
    "MEALPREP": "备餐能力",
    "REMDATES": "记忆日期能力",
    "STOVE": "使用炉灶能力",
    "TAXES": "税务/资料处理能力",
    "TRAVEL": "出行能力",
    "CBSTROKE": "卒中病史",
    "PD": "帕金森病史",
    "TBI": "脑外伤史",
    "HYPERTEN": "高血压",
    "HYPERCHO": "高胆固醇",
    "SEX": "性别",
    "DEP2YRS": "近两年抑郁",
    "PACKSPER": "平均每日吸烟量",
    "TOBAC100": "一生是否累计吸烟超过100支",
    "DEPOTHR": "两年前是否有抑郁发作",
}

FEATURE_QUESTION_CONFIG: Dict[str, str] = {
    "SEX": "请选择您的性别",
    "NACCAGEB": "您的年龄是？",
    "EDUC": "您的受教育水平是？(没有受过正式教育：0年; 小学毕业：6年; 初中毕业：9年; 高中或中专毕业：12年; 大专或高职毕业：15年; 大学及其以上毕业：≥16年)",
    "SMOKYRS": "您累计吸烟多少年？",
    "HYPERTEN": "您是否有高血压病史？",
    "HYPERCHO": "您是否有高胆固醇血症病史？",
    "CBSTROKE": "您是否有脑卒中病史？",
    "TBI": "您是否有脑外伤史？",
    "PD": "您是否有帕金森病病史？",
    "DEP2YRS": "过去两年内，您是否出现过活动性抑郁症状或被诊断为抑郁？",
    "DEPOTHR": "两年以前，您是否出现过抑郁发作？",
    "TOBAC100": "您一生中是否累计吸烟超过 100 支？",
    "PACKSPER": "在吸烟时期，您平均每天吸烟多少包？",
    "EVENTS": "过去4周内，您在关注和了解各类时事动态方面，是否存在困难或需要帮助？",
    "GAMES": "过去4周内，您在参与棋牌、技巧类游戏或个人兴趣活动时，是否存在困难或需要帮助？",
    "MEALPREP": "过去4周内，您在准备营养均衡的一日三餐时，是否存在困难或需要帮助？",
    "REMDATES": "过去4周内，您在记住预约、节假日、服药等日期相关事项时，是否存在困难或需要帮助？",
    "STOVE": "过去4周内，您在烧水、煮饭、冲咖啡、关闭炉灶等操作时，是否存在困难或需要帮助？",
    "TAXES": "过去4周内，您在整理税务文件、业务资料或重要票据时，是否存在困难或需要帮助？",
    "TRAVEL": "过去4周内，您在独自外出、驾驶或乘坐公共交通时，是否存在困难或需要帮助？",
}

FEATURE_INPUT_CONFIG: Dict[str, Dict[str, Any]] = {
    "NACCAGEB": {"min": 18, "max": 120, "default": 70, "step": 1},
    "EDUC": {"min": 0, "max": 31, "default": 9, "step": 1},
    "SMOKYRS": {"min": 0, "max": 100, "default": 0, "step": 1},
}

QUESTION_MODULES: List[Dict[str, Any]] = [
    {
        "key": "demographic",
        "title": "一、人口学与基本信息",
        "description": "请先填写基本人口学信息，用于完成基线资料录入。",
        "features": ["SEX", "NACCAGEB", "EDUC"],
    },
    {
        "key": "medical",
        "title": "二、既往病史调查",
        "description": "以下问题关注您既往的慢病、神经系统病史和情绪病史。",
        "features": ["HYPERTEN", "HYPERCHO", "CBSTROKE", "TBI", "PD", "DEP2YRS", "DEPOTHR"],
    },
    {
        "key": "habit",
        "title": "三、生活习惯调查",
        "description": "以下问题关注吸烟相关信息。",
        "features": ["TOBAC100", "SMOKYRS", "PACKSPER"],
    },
    {
        "key": "function",
        "title": "四、日常功能与生活能力",
        "description": "请根据最近 4 周的真实情况作答。",
        "features": ["EVENTS", "REMDATES", "TAXES", "MEALPREP", "STOVE", "GAMES", "TRAVEL"],
    },
]

OPTION_MEANINGS: Dict[str, Dict[Any, str]] = {
    "SEX": {1: "男", 2: "女"},
    "HYPERTEN": {0: "无 / 从未有", 1: "近期或当前存在", 2: "既往有，但目前不活动或已缓解"},
    "HYPERCHO": {0: "无 / 从未有", 1: "近期或当前存在", 2: "既往有，但目前不活动或已缓解"},
    "CBSTROKE": {0: "无 / 从未有", 1: "近期或当前存在", 2: "既往有，但目前不活动或已缓解"},
    "PD": {0: "无", 1: "有"},
    "TBI": {0: "无", 1: "单次脑外伤", 2: "反复/多次脑外伤"},
    "DEP2YRS": {0: "否", 1: "是"},
    "DEPOTHR": {0: "否", 1: "是"},
    "TOBAC100": {0: "否", 1: "是"},
    "PACKSPER": {
        0: "无吸烟 / 无报告吸烟",
        1: "1 支到不足半包/天",
        2: "半包到不足 1 包/天",
        3: "1 包到 1.5 包/天",
        4: "1.5 包到 2 包/天",
        5: "超过 2 包/天",
    },
    "EVENTS": {0: "正常，可独立完成", 1: "有困难，但仍可自行完成", 2: "需要协助", 3: "完全依赖他人"},
    "REMDATES": {0: "正常，可独立完成", 1: "有困难，但仍可自行完成", 2: "需要协助", 3: "完全依赖他人"},
    "TAXES": {0: "正常，可独立完成", 1: "有困难，但仍可自行完成", 2: "需要协助", 3: "完全依赖他人"},
    "MEALPREP": {0: "正常，可独立完成", 1: "有困难，但仍可自行完成", 2: "需要协助", 3: "完全依赖他人"},
    "STOVE": {0: "正常，可独立完成", 1: "有困难，但仍可自行完成", 2: "需要协助", 3: "完全依赖他人"},
    "GAMES": {0: "正常，可独立完成", 1: "有困难，但仍可自行完成", 2: "需要协助", 3: "完全依赖他人"},
    "TRAVEL": {0: "正常，可独立完成", 1: "有困难，但仍可自行完成", 2: "需要协助", 3: "完全依赖他人"},
}

MODULE_ICONS = {
    "demographic": "👤",
    "medical": "🩺",
    "habit": "🚬",
    "function": "🧭",
}

FEATURE_CN: Dict[str, str] = {}


@st.cache_resource(show_spinner=False)
def load_model_and_explainer(model_path: str, model_mtime: float) -> Tuple[CatBoostClassifier, shap.TreeExplainer]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"未找到模型文件：{model_path}。请将训练好的 CatBoost 模型放到该路径，或在侧边栏修改路径。"
        )

    model = CatBoostClassifier()
    model.load_model(model_path)
    explainer = shap.TreeExplainer(model)
    return model, explainer


@st.cache_data(show_spinner=False)
def load_reference_data(data_path: str, data_mtime: float) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"未找到历史数据文件：{data_path}。请提供包含特征列与标签列的 CSV / Excel 文件。"
        )

    ext = os.path.splitext(data_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    return df


@st.cache_data(show_spinner=False)
def load_feature_source(feature_source_path: str, feature_source_mtime: float) -> pd.DataFrame:
    if not os.path.exists(feature_source_path):
        raise FileNotFoundError(
            f"未找到特征参考文件：{feature_source_path}。请将 xlsx/csv 文件放到该路径，或在侧边栏修改路径。"
        )

    ext = os.path.splitext(feature_source_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(feature_source_path)
    else:
        df = pd.read_csv(feature_source_path)

    return df


@st.cache_data(show_spinner=False)
def load_scaler_source(scaler_source_path: str, scaler_source_mtime: float) -> pd.DataFrame:
    if not os.path.exists(scaler_source_path):
        raise FileNotFoundError(
            f"未找到标准化训练数据文件：{scaler_source_path}。请提供原始 X_train 对应的 CSV / Excel 文件。"
        )

    ext = os.path.splitext(scaler_source_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(scaler_source_path)
    else:
        df = pd.read_csv(scaler_source_path)
    return df


def inject_questionnaire_css() -> None:
    st.markdown(
        """
        <style>
        .survey-shell {
            background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
            border: 1px solid #e3eefc;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 18px;
        }
        .module-card {
            background: #ffffff;
            border: 1px solid #e9eef5;
            border-left: 6px solid #4f8dfd;
            border-radius: 16px;
            padding: 18px 18px 12px 18px;
            margin: 16px 0 8px 0;
            box-shadow: 0 4px 18px rgba(21, 61, 123, 0.06);
        }
        .module-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: #183b66;
            margin-bottom: 4px;
        }
        .module-desc {
            font-size: 0.95rem;
            color: #5d6f87;
            margin-bottom: 2px;
        }
        .question-chip {
            display: inline-block;
            font-size: 0.82rem;
            color: #3b5b84;
            background: #eef4ff;
            border: 1px solid #d8e6ff;
            border-radius: 999px;
            padding: 4px 10px;
            margin-top: 6px;
            margin-bottom: 10px;
        }
        .option-note {
            background: #f9fbfe;
            border: 1px dashed #d9e2ef;
            border-radius: 12px;
            padding: 10px 12px;
            color: #4d6079;
            font-size: 0.92rem;
            line-height: 1.6;
            margin-top: 2px;
            margin-bottom: 14px;
        }
        .question-divider {
            border-top: 1px solid #edf2f8;
            margin: 10px 0 18px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_label_for_task(label: Any, task: str) -> str:
    if task == "COG":
        return COG_LABEL_MAP.get(label, str(label))
    if task == "ADD":
        return ADD_LABEL_MAP.get(label, str(label))
    return FINAL_LABEL_MAP.get(label, str(label))


def normalize_final_label(label: Any) -> str:
    return normalize_label_for_task(label, task="FINAL")


def display_label(label: Any, task: Optional[str] = None) -> str:
    if label is None:
        return ""

    candidates: List[str] = []
    if task is not None:
        candidates.append(normalize_label_for_task(label, task))
    candidates.extend([
        normalize_final_label(label),
        normalize_label_for_task(label, "COG"),
        normalize_label_for_task(label, "ADD"),
        str(label),
    ])

    for candidate in candidates:
        if candidate in LABEL_DISPLAY_MAP:
            return LABEL_DISPLAY_MAP[candidate]
    return str(label)


def get_file_mtime(file_path: str) -> float:
    return os.path.getmtime(file_path) if os.path.exists(file_path) else 0.0


def feature_display_name(feature: str) -> str:
    return FEATURE_CN.get(feature, feature)


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_numeric_step(min_value: float, max_value: float, int_like: bool) -> float:
    if int_like:
        return 1.0

    span = float(max_value) - float(min_value)
    if span >= 100:
        return 1.0
    if span >= 10:
        return 0.1
    return 0.01


def resolve_feature_list(feature_df: pd.DataFrame) -> List[str]:
    raw_feature_cols = [col for col in feature_df.columns if col not in EXCLUDED_COLUMNS]

    if not REQUIRED_FEATURES:
        return raw_feature_cols

    missing_features = [feature for feature in REQUIRED_FEATURES if feature not in raw_feature_cols]
    if missing_features:
        raise ValueError(
            "以下 REQUIRED_FEATURES 在特征文件中不存在：" + "、".join(missing_features)
        )

    return [feature for feature in REQUIRED_FEATURES if feature in raw_feature_cols]


def infer_feature_metadata(
    feature_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    feature_cols = feature_cols or [col for col in feature_df.columns if col not in EXCLUDED_COLUMNS]
    metadata: Dict[str, Dict[str, Any]] = {}

    for col in feature_cols:
        numeric_series = to_numeric_series(feature_df[col]).dropna()

        if numeric_series.empty:
            metadata[col] = {
                "feature": col,
                "is_discrete": col not in CONTINUOUS_FEATURES,
                "int_like": col not in CONTINUOUS_FEATURES,
                "options": [],
                "min": 0.0,
                "max": 1.0,
                "default": 0.0,
                "nunique": 0,
                "dtype": str(feature_df[col].dtype),
                "step": 0.1,
            }
            continue

        raw_unique = sorted(pd.unique(numeric_series))
        int_like = bool(np.allclose(numeric_series, np.round(numeric_series), equal_nan=True))
        unique_values = [int(round(v)) if int_like else float(v) for v in raw_unique]
        nunique = len(unique_values)

        if col in CONTINUOUS_FEATURES:
            is_discrete = False
        else:
            is_discrete = True
            int_like = True
            unique_values = [int(round(v)) for v in raw_unique]

        if is_discrete:
            default_value = numeric_series.mode().iloc[0] if len(numeric_series) > 0 else 0
            default_value = int(round(default_value))
            min_value = int(min(unique_values))
            max_value = int(max(unique_values))
        else:
            default_value = float(numeric_series.median())
            min_value = float(numeric_series.min())
            max_value = float(numeric_series.max())

        metadata[col] = {
            "feature": col,
            "is_discrete": is_discrete,
            "int_like": int_like,
            "options": unique_values if is_discrete else [],
            "min": min_value,
            "max": max_value,
            "default": default_value,
            "nunique": nunique,
            "dtype": str(feature_df[col].dtype),
            "step": infer_numeric_step(min_value, max_value, int_like),
        }

    return feature_cols, metadata


def get_continuous_feature_input_spec(feature: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    custom_spec = FEATURE_INPUT_CONFIG.get(feature, {})
    min_value = custom_spec.get("min", meta["min"])
    max_value = custom_spec.get("max", meta["max"])
    default_value = custom_spec.get("default", meta["default"])
    step_value = custom_spec.get("step", meta["step"])

    if min_value > max_value:
        min_value, max_value = max_value, min_value

    if default_value < min_value:
        default_value = min_value
    if default_value > max_value:
        default_value = max_value

    return {"min": min_value, "max": max_value, "default": default_value, "step": step_value}


def format_range_text(min_value: Any, max_value: Any) -> str:
    if isinstance(min_value, (int, np.integer)) and isinstance(max_value, (int, np.integer)):
        return f"{int(min_value)}-{int(max_value)}"

    if float(min_value).is_integer() and float(max_value).is_integer():
        return f"{int(round(float(min_value)))}-{int(round(float(max_value)))}"

    return f"{float(min_value):.4f}-{float(max_value):.4f}"


def get_discrete_option_meanings(feature: str, options: List[Any]) -> Dict[Any, str]:
    configured = OPTION_MEANINGS.get(feature, {})
    resolved: Dict[Any, str] = {}
    for opt in options:
        key = int(round(opt)) if isinstance(opt, (float, np.floating)) and float(opt).is_integer() else opt
        resolved[key] = configured.get(key, f"编码值 {key}")
    return resolved


def format_option_note(feature: str, options: List[Any]) -> str:
    option_meanings = get_discrete_option_meanings(feature, options)
    parts = [f"<b>{opt}</b> = {meaning}" for opt, meaning in option_meanings.items()]
    return "；".join(parts)


def build_question_prompt(feature: str, meta: Dict[str, Any]) -> str:
    question_text = FEATURE_QUESTION_CONFIG.get(feature, f"请输入{feature_display_name(feature)}")
    if meta["is_discrete"]:
        return question_text
    spec = get_continuous_feature_input_spec(feature, meta)
    return f"{question_text}（填写范围：{format_range_text(spec['min'], spec['max'])}）"


def align_input_features_to_model(model: CatBoostClassifier, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    model_features = list(getattr(model, "feature_names_", []))

    if not model_features:
        return input_df.copy(), []

    missing_features = [col for col in model_features if col not in input_df.columns]
    extra_features = [col for col in input_df.columns if col not in model_features]

    if missing_features:
        raise ValueError(
            "当前输入缺少模型所需特征：" + ", ".join(missing_features) + "。请检查字段名是否与模型训练字段一致。"
        )

    aligned_df = input_df[model_features].copy()
    return aligned_df, extra_features


def get_task_class_order(task: str) -> List[str]:
    if task == "COG":
        return COG_CLASS_ORDER
    if task == "ADD":
        return ADD_CLASS_ORDER
    return FINAL_CLASS_ORDER


def get_class_display_names(model: CatBoostClassifier, proba_vector: np.ndarray, task: str) -> Tuple[List[Any], List[str]]:
    task_order = get_task_class_order(task)
    raw_classes = list(getattr(model, "classes_", task_order))

    if len(raw_classes) != len(proba_vector):
        raw_classes = task_order[: len(proba_vector)]

    display_classes = [normalize_label_for_task(c, task) for c in raw_classes]
    return raw_classes, display_classes


def build_probability_df(model: CatBoostClassifier, probabilities: np.ndarray, task: str) -> pd.DataFrame:
    raw_classes, display_classes = get_class_display_names(model, probabilities, task)

    df = pd.DataFrame(
        {"Raw_Class": raw_classes, "Class": display_classes, "Probability": probabilities}
    )

    class_rank = {cls: idx for idx, cls in enumerate(get_task_class_order(task))}
    df["Rank"] = df["Class"].map(class_rank).fillna(999)
    df = df.sort_values("Rank").drop(columns="Rank").reset_index(drop=True)
    return df


def extract_predicted_class_index(prob_df: pd.DataFrame) -> int:
    return int(prob_df["Probability"].values.argmax())


def get_model_class_index_for_shap(
    model: CatBoostClassifier,
    predicted_display_class: str,
    probabilities: np.ndarray,
    task: str,
) -> int:
    _, display_classes = get_class_display_names(model, probabilities, task)

    for idx, display_name in enumerate(display_classes):
        if display_name == predicted_display_class:
            return idx

    return 0


def build_shap_explanation(
    explainer: shap.TreeExplainer,
    model_input_df: pd.DataFrame,
    class_index: int,
    display_input_df: Optional[pd.DataFrame] = None,
) -> Tuple[shap.Explanation, pd.DataFrame]:
    shap_values_raw = explainer.shap_values(model_input_df)
    expected_value_raw = explainer.expected_value

    if isinstance(shap_values_raw, list):
        selected_shap_values = np.asarray(shap_values_raw[class_index])[0]
    else:
        shap_array = np.asarray(shap_values_raw)

        if shap_array.ndim == 3:
            if shap_array.shape[0] == model_input_df.shape[0] and shap_array.shape[1] == model_input_df.shape[1]:
                selected_shap_values = shap_array[0, :, class_index]
            elif shap_array.shape[0] > 1 and shap_array.shape[1] == model_input_df.shape[0]:
                selected_shap_values = shap_array[class_index, 0, :]
            elif shap_array.shape[0] == model_input_df.shape[0] and shap_array.shape[2] == model_input_df.shape[1]:
                selected_shap_values = shap_array[0, class_index, :]
            else:
                raise ValueError(
                    f"暂不支持的 SHAP 数组形状：{shap_array.shape}。请根据当前 shap 版本调整 build_shap_explanation 函数。"
                )
        elif shap_array.ndim == 2:
            selected_shap_values = shap_array[0]
        else:
            raise ValueError(f"暂不支持的 SHAP 输出维度：{shap_array.ndim}。")

    expected_value_array = np.asarray(expected_value_raw)
    if expected_value_array.ndim == 0:
        base_value = float(expected_value_array)
    else:
        expected_flat = expected_value_array.reshape(-1)
        if class_index < len(expected_flat):
            base_value = float(expected_flat[class_index])
        else:
            base_value = float(expected_flat[0])

    visible_df = display_input_df.copy() if display_input_df is not None else model_input_df.copy()

    explanation = shap.Explanation(
        values=np.asarray(selected_shap_values, dtype=float),
        base_values=base_value,
        data=visible_df.iloc[0].values,
        feature_names=model_input_df.columns.tolist(),
    )

    shap_df = pd.DataFrame(
        {"Feature": model_input_df.columns, "Value": visible_df.iloc[0].values, "SHAP": explanation.values}
    )
    shap_df["Abs_SHAP"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values("Abs_SHAP", ascending=False).reset_index(drop=True)

    return explanation, shap_df


def plot_probability_bar(prob_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    plot_df = prob_df.copy()
    plot_df["ClassDisplay"] = plot_df["Class"].map(display_label)
    sns.barplot(data=plot_df, x="ClassDisplay", y="Probability", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("类别")
    ax.set_ylabel("概率")
    ax.set_ylim(0, 1)

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.1%}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    return fig


def plot_shap_waterfall(explanation: shap.Explanation):
    feature_names = explanation.feature_names or [f"Feature {idx + 1}" for idx in range(len(explanation.values))]
    shap_df = pd.DataFrame({"Feature": feature_names, "SHAP": np.asarray(explanation.values, dtype=float)})
    shap_df["Abs_SHAP"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values("Abs_SHAP", ascending=False).head(min(10, len(shap_df)))
    shap_df = shap_df.sort_values("SHAP", ascending=True)

    colors = np.where(shap_df["SHAP"] >= 0, "#d95f5f", "#4c72b0")
    fig_height = max(4.8, 0.55 * len(shap_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    bars = ax.barh(shap_df["Feature"], shap_df["SHAP"], color=colors, alpha=0.92)
    ax.axvline(0, color="#6c757d", linewidth=1.0)
    ax.set_title("主要特征贡献")
    ax.set_xlabel("SHAP 值")
    ax.set_ylabel("")

    max_abs_value = float(shap_df["Abs_SHAP"].max()) if not shap_df.empty else 0.0
    text_offset = max_abs_value * 0.03 if max_abs_value > 0 else 0.02
    for bar, shap_value in zip(bars, shap_df["SHAP"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        if shap_value >= 0:
            ax.text(x + text_offset, y, f"{shap_value:.3f}", va="center", ha="left", fontsize=9)
        else:
            ax.text(x - text_offset, y, f"{shap_value:.3f}", va="center", ha="right", fontsize=9)

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()
    return fig


def derive_four_class_label(row: pd.Series) -> str:
    cog = row.get("COG", np.nan)
    add = row.get("ADD", np.nan)

    if pd.isna(cog):
        return np.nan

    try:
        cog = int(cog)
    except Exception:
        return np.nan

    if cog == 0:
        return "NC"
    if cog == 1:
        return "MCI"
    if cog == 2:
        if pd.isna(add):
            return np.nan
        try:
            add = int(add)
        except Exception:
            return np.nan
        return "AD" if add == 1 else "nAD"

    return np.nan


def summarize_continuous_distribution(
    df: pd.DataFrame,
    continuous_features: List[str],
) -> pd.DataFrame:
    available_continuous = [f for f in continuous_features if f in df.columns]
    if not available_continuous:
        return pd.DataFrame(columns=["Feature", "mu_train", "sigma_train"])

    numeric_df = df[available_continuous].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(
        {
            "Feature": available_continuous,
            "mu_train": numeric_df.mean(axis=0, skipna=True).values,
            "sigma_train": numeric_df.std(axis=0, skipna=True, ddof=0).values,
        }
    )


def prepare_reference_plot_data(
    reference_df: pd.DataFrame,
    scaler: Optional[StandardScaler],
    continuous_features: List[str],
) -> Tuple[pd.DataFrame, str]:
    plot_df = reference_df.copy()
    available_continuous = [f for f in continuous_features if f in plot_df.columns]

    if scaler is None or not available_continuous:
        return plot_df, "raw"

    reference_stats_df = summarize_continuous_distribution(plot_df, available_continuous)
    if detect_prestandardized_data(reference_stats_df):
        return plot_df, "standardized_existing"

    numeric_continuous = plot_df[available_continuous].apply(pd.to_numeric, errors="coerce")
    plot_df.loc[:, available_continuous] = scaler.transform(numeric_continuous)
    return plot_df, "standardized_transformed"


def prepare_reference_mean_data(reference_df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = reference_df.copy()

    if "Label" in df.columns:
        df["Label"] = df["Label"].apply(normalize_final_label)
    elif {"COG", "ADD"}.issubset(df.columns):
        df["Label"] = df.apply(derive_four_class_label, axis=1)
    else:
        raise ValueError("历史数据中既没有 Label 列，也无法通过 COG + ADD 推导四分类标签。")

    available_features = [f for f in feature_list if f in df.columns]
    if not available_features:
        raise ValueError("历史数据中未找到任何可用于常模对比的特征列。")

    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Label"])

    mean_df = (
        df.groupby("Label")[available_features]
        .mean(numeric_only=True)
        .reindex(FINAL_CLASS_ORDER)
    )

    return mean_df, available_features


def plot_reference_comparison(mean_df: pd.DataFrame, input_df: pd.DataFrame, available_features: List[str]):
    plot_df = mean_df.reset_index().melt(
        id_vars="Label",
        var_name="Feature",
        value_name="Mean_Value",
    )

    plot_df["Feature"] = pd.Categorical(plot_df["Feature"], categories=available_features, ordered=True)
    plot_df["LabelDisplay"] = plot_df["Label"].map(display_label)
    plot_df = plot_df.sort_values(["Feature", "Label"])

    fig, ax = plt.subplots(figsize=(max(12, len(available_features) * 0.45), 6))
    sns.lineplot(
        data=plot_df,
        x="Feature",
        y="Mean_Value",
        hue="LabelDisplay",
        marker="o",
        ax=ax,
        sort=False,
    )

    user_values = input_df.iloc[0][available_features].astype(float).values
    ax.scatter(
        x=np.arange(len(available_features)),
        y=user_values,
        color="red",
        marker="*",
        s=180,
        label="当前用户",
        zorder=5,
    )

    ax.set_xticks(np.arange(len(available_features)))
    ax.set_xticklabels([feature_display_name(f) for f in available_features], rotation=60, ha="right")
    ax.set_title("常模对比图：四类均值 vs 当前用户")
    ax.set_xlabel("特征")
    ax.set_ylabel("数值")
    ax.legend()
    plt.tight_layout()
    return fig


def format_top_shap_table(top_shap_df: pd.DataFrame, predicted_class: str) -> pd.DataFrame:
    display_df = top_shap_df.copy()
    predicted_display = display_label(predicted_class)
    display_df["特征"] = display_df["Feature"].map(feature_display_name)
    display_df["当前值"] = display_df["Value"]
    display_df["SHAP值"] = display_df["SHAP"].round(4)
    display_df["影响方向"] = np.where(
        display_df["SHAP"] >= 0,
        f"推动 -> {predicted_display}",
        f"远离 -> {predicted_display}",
    )
    return display_df[["特征", "当前值", "SHAP值", "影响方向"]]


def feature_advice(feature: str, value: float, shap_value: float, predicted_class: str) -> str:
    direction_text = "推动了当前评估结果" if shap_value >= 0 else "在一定程度上抵消了当前评估结果倾向"
    return f"{feature_display_name(feature)}（当前值 {value}）{direction_text}，建议结合临床评估综合解释。"


def generate_recommendations(predicted_class: str, confidence: float, top_shap_df: pd.DataFrame) -> str:
    diagnosis_name = display_label(predicted_class)
    intro_map = {
        "NC": "本次评估结果提示：当前更接近无认知障碍状态，建议保持规律作息、运动和定期随访。",
        "MCI": "本次评估结果提示：当前更接近轻度认知障碍，建议结合门诊评估与随访复查进一步确认。",
        "AD": "本次评估结果提示：当前更接近阿尔兹海默症型痴呆，建议尽快到相关专科进一步检查。",
        "nAD": "本次评估结果提示：当前更接近非阿尔兹海默症型痴呆，建议结合临床表现进一步鉴别诊断。",
        "DE": "本次评估结果提示：当前存在痴呆风险，建议尽快进行进一步检查。",
    }

    lines = [
        intro_map.get(predicted_class, f"本次评估结果为：{diagnosis_name}。"),
        f"当前评估概率约为 {confidence:.1%}。",
        "",
        "结合 SHAP 解释，以下 3 个因素对本次判断影响最大：",
    ]

    for idx, row in enumerate(top_shap_df.head(3).itertuples(index=False), start=1):
        advice_text = feature_advice(
            feature=row.Feature,
            value=row.Value,
            shap_value=row.SHAP,
            predicted_class=predicted_class,
        )
        lines.append(f"{idx}. {feature_display_name(row.Feature)}：{advice_text}")

    lines.extend(
        [
            "",
            "温馨提示：本系统仅用于智能辅助筛查，不能替代医生面诊、正式神经心理测评或影像学诊断。",
            "如近期认知、行为或日常功能下降明显，请尽快就医。",
        ]
    )

    return "\n".join(lines)


def build_user_input_df(user_inputs: Dict[str, Any], feature_list: List[str]) -> pd.DataFrame:
    return pd.DataFrame([user_inputs], columns=feature_list)


def fit_continuous_standardizer(
    scaler_source_df: pd.DataFrame,
    feature_list: List[str],
    feature_meta: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[StandardScaler], List[str], pd.DataFrame]:
    continuous_features = [f for f in feature_list if not feature_meta[f]["is_discrete"]]

    if not continuous_features:
        empty_stats = pd.DataFrame(columns=["Feature", "mu_train", "sigma_train"])
        return None, [], empty_stats

    X_train = scaler_source_df[continuous_features].apply(pd.to_numeric, errors="coerce")

    scaler = StandardScaler()
    scaler.fit(X_train)

    stats_df = pd.DataFrame(
        {"Feature": continuous_features, "mu_train": scaler.mean_, "sigma_train": scaler.scale_}
    )

    return scaler, continuous_features, stats_df


def apply_continuous_standardization(
    input_df: pd.DataFrame,
    scaler: Optional[StandardScaler],
    continuous_features: List[str],
) -> pd.DataFrame:
    if scaler is None or not continuous_features:
        return input_df.copy()

    scaled_df = input_df.copy()
    raw_continuous = scaled_df[continuous_features].apply(pd.to_numeric, errors="coerce")
    scaled_values = scaler.transform(raw_continuous)
    scaled_df.loc[:, continuous_features] = scaled_values
    return scaled_df


def detect_prestandardized_data(scaler_stats_df: pd.DataFrame) -> bool:
    if scaler_stats_df.empty:
        return False

    mu_close = np.isclose(scaler_stats_df["mu_train"].values, 0.0, atol=1e-6)
    sigma_close = np.isclose(scaler_stats_df["sigma_train"].values, 1.0, atol=1e-6)
    ratio = float(np.mean(mu_close & sigma_close))
    return ratio >= 0.8


def render_feature_input_form(
    feature_list: List[str],
    feature_meta: Dict[str, Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    inject_questionnaire_css()
    st.markdown("### 筛查问卷输入")
    st.markdown(
        "<div class='survey-shell'>请按照页面顺序逐题作答。连续变量采用填空方式输入，离散变量采用单选方式选择，并在题目下方显示各编码值的含义。</div>",
        unsafe_allow_html=True,
    )

    user_inputs: Dict[str, Any] = {}
    question_index = 1

    with st.form("assessment_form"):
        for module in QUESTION_MODULES:
            module_features = [f for f in module["features"] if f in feature_list]
            if not module_features:
                continue

            icon = MODULE_ICONS.get(module["key"], "📝")
            st.markdown(
                f"""
                <div class='module-card'>
                    <div class='module-title'>{icon} {module['title']}</div>
                    <div class='module-desc'>{module['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for feature in module_features:
                meta = feature_meta[feature]
                prompt = build_question_prompt(feature, meta)
                st.markdown(
                    f"<div class='question-chip'>第 {question_index} 题 · {feature_display_name(feature)}</div>",
                    unsafe_allow_html=True,
                )

                if meta["is_discrete"]:
                    options = [int(round(v)) if float(v).is_integer() else v for v in meta["options"]]
                    default_value = int(round(meta["default"])) if float(meta["default"]).is_integer() else meta["default"]
                    default_index = options.index(default_value) if default_value in options else 0
                    option_meanings = get_discrete_option_meanings(feature, options)

                    selected = st.radio(
                        prompt,
                        options=options,
                        index=default_index,
                        key=f"input_{feature}",
                        format_func=lambda x, option_meanings=option_meanings: f"{x} · {option_meanings.get(x, f'编码值 {x}')}",
                    )
                    st.markdown(
                        f"<div class='option-note'>{format_option_note(feature, options)}</div>",
                        unsafe_allow_html=True,
                    )
                    user_inputs[feature] = int(round(selected)) if isinstance(selected, (int, float, np.integer, np.floating)) and float(selected).is_integer() else selected
                else:
                    spec = get_continuous_feature_input_spec(feature, meta)
                    if meta["int_like"] or float(spec["step"]).is_integer():
                        value = st.number_input(
                            prompt,
                            min_value=int(round(spec["min"])),
                            max_value=int(round(spec["max"])),
                            value=int(round(spec["default"])),
                            step=max(1, int(round(spec["step"]))),
                            key=f"input_{feature}",
                        )
                        user_inputs[feature] = int(value)
                    else:
                        value = st.number_input(
                            prompt,
                            min_value=float(spec["min"]),
                            max_value=float(spec["max"]),
                            value=float(spec["default"]),
                            step=float(spec["step"]),
                            format="%.4f",
                            key=f"input_{feature}",
                        )
                        user_inputs[feature] = float(value)

                st.markdown("<div class='question-divider'></div>", unsafe_allow_html=True)
                question_index += 1

        submitted = st.form_submit_button("提交问卷并开始评估", use_container_width=True)

    return submitted, user_inputs


def probability_df_to_dict(prob_df: pd.DataFrame) -> Dict[str, float]:
    return {str(row["Class"]): float(row["Probability"]) for _, row in prob_df.iterrows()}


def build_joint_probability_df(cog_prob_df: pd.DataFrame, add_prob_df: pd.DataFrame) -> pd.DataFrame:
    cog_prob = probability_df_to_dict(cog_prob_df)
    add_prob = probability_df_to_dict(add_prob_df)

    p_nc = float(cog_prob.get("NC", 0.0))
    p_mci = float(cog_prob.get("MCI", 0.0))
    p_de = float(cog_prob.get("DE", 0.0))
    p_ad_cond = float(add_prob.get("AD", 0.0))
    p_nad_cond = float(add_prob.get("nAD", 0.0))

    final_df = pd.DataFrame(
        {
            "Class": FINAL_CLASS_ORDER,
            "Probability": [p_nc, p_mci, p_de * p_ad_cond, p_de * p_nad_cond],
            "Source": ["COG", "COG", "COG×ADD", "COG×ADD"],
            "COG_DE_Probability": [np.nan, np.nan, p_de, p_de],
            "ADD_Conditional_Probability": [np.nan, np.nan, p_ad_cond, p_nad_cond],
        }
    )
    return final_df


def get_prediction_row(prob_df: pd.DataFrame) -> pd.Series:
    return prob_df.iloc[extract_predicted_class_index(prob_df)]


def main():
    global FEATURE_CN

    st.set_page_config(
        page_title="老年人认知障碍风险智能筛查评估系统",
        page_icon="🧠",
        layout="wide",
    )

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans"
]
    plt.rcParams["axes.unicode_minus"] = False

    if "editable_recommendation" not in st.session_state:
        st.session_state["editable_recommendation"] = ""

    st.title("老年人认知障碍风险智能筛查评估系统")
    st.markdown("请根据实际情况完成下方问卷，提交后系统将自动生成评估结果、特征解释和个性化建议。")

    st.sidebar.header("文件与参数配置")
    feature_source_path = st.sidebar.text_input("特征参考文件路径（xlsx/csv）", value=DEFAULT_FEATURE_SOURCE_PATH)
    scaler_source_path = st.sidebar.text_input("标准化训练数据路径（原始 X_train）", value=DEFAULT_SCALER_SOURCE_PATH)
    cog_model_path = st.sidebar.text_input("COG 模型路径（NC/MCI/DE）", value=DEFAULT_COG_MODEL_PATH)
    add_model_path = st.sidebar.text_input("ADD 模型路径（AD/nAD）", value=DEFAULT_ADD_MODEL_PATH)
    data_path = st.sidebar.text_input("历史数据路径（可选，CSV/XLSX）", value=DEFAULT_DATA_PATH)

    feature_df = None
    scaler_source_df = None
    feature_list: List[str] = []
    feature_meta: Dict[str, Dict[str, Any]] = {}
    scaler: Optional[StandardScaler] = None
    continuous_features_for_scaling: List[str] = []
    scaler_stats_df = pd.DataFrame()

    try:
        feature_df = load_feature_source(
            feature_source_path=feature_source_path,
            feature_source_mtime=get_file_mtime(feature_source_path),
        )
        feature_list = resolve_feature_list(feature_df)
        feature_list, feature_meta = infer_feature_metadata(feature_df, feature_cols=feature_list)
        FEATURE_CN = {feature: FEATURE_NAME_CONFIG.get(feature, feature) for feature in feature_list}
    except Exception as e:
        st.sidebar.error(f"特征文件加载失败：{e}")
        st.stop()

    try:
        scaler_source_df = load_scaler_source(
            scaler_source_path=scaler_source_path,
            scaler_source_mtime=get_file_mtime(scaler_source_path),
        )
        scaler, continuous_features_for_scaling, scaler_stats_df = fit_continuous_standardizer(
            scaler_source_df=scaler_source_df,
            feature_list=feature_list,
            feature_meta=feature_meta,
        )
    except Exception as e:
        st.sidebar.error(f"标准化训练数据加载失败：{e}")
        st.stop()


    submitted, user_inputs = render_feature_input_form(feature_list, feature_meta)
    user_input_df_raw = build_user_input_df(user_inputs, feature_list)


    cog_model = None
    cog_explainer = None
    add_model = None
    add_explainer = None
    reference_df = None

    try:
        cog_model, cog_explainer = load_model_and_explainer(
            model_path=cog_model_path,
            model_mtime=get_file_mtime(cog_model_path),
        )
    except FileNotFoundError as e:
        st.sidebar.warning(str(e))
    except Exception as e:
        st.sidebar.warning(f"COG 模型加载失败：{e}")

    try:
        add_model, add_explainer = load_model_and_explainer(
            model_path=add_model_path,
            model_mtime=get_file_mtime(add_model_path),
        )
    except FileNotFoundError as e:
        st.sidebar.warning(str(e))
    except Exception as e:
        st.sidebar.warning(f"ADD 模型加载失败：{e}")

    data_path_stripped = data_path.strip()
    if data_path_stripped and os.path.exists(data_path_stripped):
        try:
            reference_df = load_reference_data(
                data_path=data_path_stripped,
                data_mtime=get_file_mtime(data_path_stripped),
            )
        except Exception as e:
            st.sidebar.warning(f"历史数据读取失败：{e}")
            reference_df = feature_df.copy()
    else:
        reference_df = feature_df.copy()

    if submitted:
        st.markdown("---")
        st.subheader("评估结果")

        if cog_model is None or cog_explainer is None:
            st.error("当前无法进行预测：COG 模型未成功加载。")
            return
        if add_model is None or add_explainer is None:
            st.error("当前无法进行预测：ADD 模型未成功加载。")
            return

        try:
            user_input_df_scaled = apply_continuous_standardization(
                input_df=user_input_df_raw,
                scaler=scaler,
                continuous_features=continuous_features_for_scaling,
            )

            aligned_input_df_cog_raw, extra_features_cog = align_input_features_to_model(cog_model, user_input_df_raw)
            aligned_input_df_add_raw, extra_features_add = align_input_features_to_model(add_model, user_input_df_raw)
            aligned_input_df_cog_scaled, _ = align_input_features_to_model(cog_model, user_input_df_scaled)
            aligned_input_df_add_scaled, _ = align_input_features_to_model(add_model, user_input_df_scaled)

            cog_probabilities = cog_model.predict_proba(aligned_input_df_cog_scaled)[0]
            add_probabilities = add_model.predict_proba(aligned_input_df_add_scaled)[0]

            cog_prob_df = build_probability_df(cog_model, cog_probabilities, task="COG")
            add_prob_df = build_probability_df(add_model, add_probabilities, task="ADD")
            final_prob_df = build_joint_probability_df(cog_prob_df, add_prob_df)

            final_pred_row = get_prediction_row(final_prob_df)

            predicted_class = str(final_pred_row["Class"])
            confidence = float(final_pred_row["Probability"])

            if predicted_class in {"NC", "MCI"}:
                shap_model_name = "COG"
                shap_class_index = get_model_class_index_for_shap(cog_model, predicted_class, cog_probabilities, task="COG")
                explanation, shap_df = build_shap_explanation(
                    explainer=cog_explainer,
                    model_input_df=aligned_input_df_cog_scaled,
                    class_index=shap_class_index,
                    display_input_df=aligned_input_df_cog_raw,
                )
            else:
                shap_model_name = "ADD"
                shap_class_index = get_model_class_index_for_shap(add_model, predicted_class, add_probabilities, task="ADD")
                explanation, shap_df = build_shap_explanation(
                    explainer=add_explainer,
                    model_input_df=aligned_input_df_add_scaled,
                    class_index=shap_class_index,
                    display_input_df=aligned_input_df_add_raw,
                )

            predicted_class_display = display_label(predicted_class)

            result_col1, result_col2 = st.columns(2)
            result_col1.metric("评估结果", predicted_class_display)
            result_col2.metric("评估概率", f"{confidence:.2%}")

            if predicted_class == "NC":
                st.success(f"本次评估结果：{predicted_class_display}（评估概率 {confidence:.2%}）")
            elif predicted_class == "MCI":
                st.warning(f"本次评估结果：{predicted_class_display}（评估概率 {confidence:.2%}）")
            else:
                st.error(f"本次评估结果：{predicted_class_display}（评估概率 {confidence:.2%}）")

            st.markdown("### 1）四分类评估概率分布")
            prob_fig = plot_probability_bar(final_prob_df[["Class", "Probability"]], "四分类评估概率分布")
            st.pyplot(prob_fig, use_container_width=True)
            plt.close(prob_fig)

            st.markdown("### 2）个体特征解释（SHAP）")
            shap_fig = plot_shap_waterfall(explanation)
            st.pyplot(shap_fig, use_container_width=True)
            plt.close(shap_fig)

            st.markdown("#### 影响最大的特征")
            top_shap_display_df = format_top_shap_table(shap_df.head(10), predicted_class)
            st.dataframe(top_shap_display_df, use_container_width=True)

            st.markdown("### 3）常模对比图")
            if reference_df is None:
                st.info("由于历史数据未成功加载，已跳过常模对比图。")
            else:
                try:
                    reference_plot_df, reference_plot_scale = prepare_reference_plot_data(
                        reference_df=reference_df,
                        scaler=scaler,
                        continuous_features=continuous_features_for_scaling,
                    )
                    mean_df, available_features = prepare_reference_mean_data(reference_plot_df, feature_list)

                    missing_for_plot = [f for f in feature_list if f not in available_features]
                    if missing_for_plot:
                        st.warning("历史数据缺少以下字段，因此未纳入常模图：" + "、".join([feature_display_name(f) for f in missing_for_plot]))

                    plot_input_df = user_input_df_scaled if reference_plot_scale in {"standardized_existing", "standardized_transformed"} else user_input_df_raw
                    comparison_fig = plot_reference_comparison(
                        mean_df=mean_df,
                        input_df=plot_input_df,
                        available_features=available_features,
                    )
                    st.pyplot(comparison_fig, use_container_width=True)
                    plt.close(comparison_fig)

                except Exception as e:
                    st.warning(f"常模对比图绘制失败：{e}")

            st.markdown("### 4）智能建议（可编辑）")
            recommendation_text = generate_recommendations(
                predicted_class=predicted_class,
                confidence=confidence,
                top_shap_df=shap_df.head(3),
            )

            st.session_state["editable_recommendation"] = recommendation_text

            st.text_area(
                "系统生成的个性化建议（您可以直接修改）：",
                key="editable_recommendation",
                height=260,
            )

            st.markdown("---")
            st.info(
                "免责声明：本系统用于辅助筛查和研究演示，不能替代医生面诊、正式诊断或治疗建议。若存在明显记忆下降、行为改变、语言障碍或生活能力下降，请尽快至神经内科/记忆门诊就诊。"
            )

        except Exception as e:
            st.error(f"评估过程中发生错误：{e}")
            st.exception(e)


if __name__ == "__main__":
    main()
