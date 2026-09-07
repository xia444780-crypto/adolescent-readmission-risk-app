from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import shap
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "final_rf_pipeline.joblib"
METADATA_PATH = APP_DIR / "model" / "model_metadata.json"

FEATURES = [
    "本次住院时长(X12,天)",
    "既往住院次数(X24)",
    "既往自杀未遂史(X26)",
    "首次发病年龄(X18,岁)",
    "出院情况",
    "伴焦虑症状(X15)",
    "医保类型(X9)",
    "既往门诊治疗次数",
    "抑郁程度(X13)",
    "总病程(X25,月)",
]

LABELS = {
    "本次住院时长(X12,天)": "本次住院时长",
    "既往住院次数(X24)": "既往住院次数",
    "既往自杀未遂史(X26)": "既往自杀未遂史",
    "首次发病年龄(X18,岁)": "首次发病年龄",
    "出院情况": "出院情况",
    "伴焦虑症状(X15)": "伴焦虑症状",
    "医保类型(X9)": "医保类型",
    "既往门诊治疗次数": "既往门诊治疗次数",
    "抑郁程度(X13)": "抑郁程度",
    "总病程(X25,月)": "总病程",
}

VALUE_LABELS = {
    "既往住院次数(X24)": {0: "0次", 1: "1次", 2: "2次", 3: "≥3次"},
    "既往自杀未遂史(X26)": {0: "无", 1: "有"},
    "出院情况": {1: "好转", 2: "未愈"},
    "伴焦虑症状(X15)": {0: "无", 1: "轻度", 2: "中度", 3: "重度"},
    "医保类型(X9)": {3: "新农合", 4: "自费", 5: "其他社会保险"},
    "既往门诊治疗次数": {0: "0次", 1: "1次", 2: "2次", 3: "≥3次"},
    "抑郁程度(X13)": {1: "轻度", 2: "中度", 3: "重度"},
}

UNITS = {
    "本次住院时长(X12,天)": "天",
    "首次发病年龄(X18,岁)": "岁",
    "总病程(X25,月)": "个月",
}


st.set_page_config(
    page_title="青少年再入院风险评估",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
    :root {
        --navy: #163A5F;
        --teal: #167D8D;
        --teal-dark: #0C6070;
        --coral: #C94C4C;
        --ink: #243442;
        --muted: #657786;
        --mist: #F3F7FA;
        --line: #DCE6ED;
        --white: #FFFFFF;
    }
    .stApp {
        background:
            radial-gradient(circle at 92% 5%, rgba(22,125,141,.10), transparent 24rem),
            linear-gradient(180deg, #F7FAFC 0%, #EDF3F7 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    .block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--navy); letter-spacing: -.02em; }
    .hero {
        background: linear-gradient(120deg, #123653 0%, #176B79 100%);
        border-radius: 24px;
        padding: 30px 34px;
        color: white;
        box-shadow: 0 18px 45px rgba(22,58,95,.18);
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        right: -75px;
        top: -120px;
        border: 1px solid rgba(255,255,255,.20);
        border-radius: 50%;
        box-shadow: 0 0 0 34px rgba(255,255,255,.04), 0 0 0 70px rgba(255,255,255,.03);
    }
    .eyebrow { font-size: .78rem; letter-spacing: .18em; opacity: .78; font-weight: 700; }
    .hero-title { font-size: 2rem; font-weight: 760; margin: 7px 0 5px; line-height: 1.28; }
    .hero-subtitle { font-size: .96rem; opacity: .84; margin: 0; }
    .section-kicker { color: var(--teal-dark); font-weight: 800; font-size: .78rem; letter-spacing: .12em; }
    .section-title { color: var(--navy); font-size: 1.25rem; font-weight: 760; margin: 3px 0 16px; }
    div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,.96);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 22px 24px;
        box-shadow: 0 10px 30px rgba(28,62,86,.08);
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="select"] > div {
        border-radius: 11px !important;
        border-color: #C9D8E2 !important;
        background: #FBFDFE !important;
    }
    div[data-testid="stFormSubmitButton"] button {
        min-height: 48px;
        border: 0;
        border-radius: 13px;
        background: linear-gradient(100deg, #163A5F, #167D8D);
        color: white;
        font-weight: 760;
        box-shadow: 0 8px 20px rgba(22,125,141,.20);
    }
    div[data-testid="stFormSubmitButton"] button:hover { color: white; border: 0; filter: brightness(1.06); }
    .identity-line { height: 1px; background: linear-gradient(90deg, var(--teal), transparent); margin: 7px 0 18px; }
    div[data-testid="stVerticalBlockBorderWrapper"] { margin-top: 22px; padding: 7px; }
    .patient-chip {
        display: inline-block;
        background: #EAF2F6;
        color: #35586D;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: .86rem;
        margin-bottom: 13px;
    }
    .risk-label { color: var(--muted); font-size: .86rem; font-weight: 650; }
    .risk-number { color: var(--navy); font-size: 3.35rem; line-height: 1; font-weight: 820; letter-spacing: -.06em; }
    .risk-number span { font-size: 1.3rem; letter-spacing: 0; }
    .risk-state-high, .risk-state-low {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 8px 13px;
        font-weight: 760;
        font-size: .88rem;
        margin-top: 13px;
    }
    .risk-state-high { background: #F9E8E8; color: #A53434; }
    .risk-state-low { background: #E4F2F0; color: #17665D; }
    .risk-track-wrap { margin-top: 22px; }
    .risk-track { height: 13px; border-radius: 999px; background: linear-gradient(90deg, #B8DDD8 0%, #7CB6BC 45%, #E4A0A0 100%); position: relative; }
    .risk-marker { position: absolute; top: -7px; width: 4px; height: 27px; background: #122E49; border-radius: 4px; transform: translateX(-2px); }
    .threshold-marker { position: absolute; top: -4px; width: 2px; height: 21px; background: rgba(31,48,60,.56); transform: translateX(-1px); }
    .risk-scale { display: flex; justify-content: space-between; color: var(--muted); font-size: .73rem; margin-top: 8px; }
    .threshold-caption { color: #536A78; font-size: .78rem; margin-top: 3px; }
    .factor-card {
        background: #F8FBFD;
        border: 1px solid var(--line);
        border-left: 4px solid var(--coral);
        border-radius: 13px;
        padding: 11px 13px;
        margin: 8px 0;
        min-height: 63px;
    }
    .factor-card.protective { border-left-color: var(--teal); }
    .factor-name { font-size: .86rem; font-weight: 760; color: var(--ink); }
    .factor-value { font-size: .78rem; color: var(--muted); margin-top: 3px; }
    .factor-delta-up { color: #A53434; font-weight: 760; }
    .factor-delta-down { color: #17665D; font-weight: 760; }
    .small-heading { color: var(--navy); font-size: 1rem; font-weight: 780; margin: 8px 0 4px; }
    .result-heading { color: var(--navy); font-size: 1.55rem; font-weight: 800; margin: 2px 0 4px; }
    .result-subtitle { color: var(--muted); font-size: .9rem; margin-bottom: 16px; }
    .advice-card {
        background: #F8FBFD;
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        border-radius: 13px;
        padding: 12px 14px;
        margin: 9px 0;
    }
    .advice-card.priority { border-left-color: var(--coral); background: #FFF9F8; }
    .advice-title { color: var(--navy); font-size: .9rem; font-weight: 800; }
    .advice-body { color: #526875; font-size: .8rem; line-height: 1.6; margin-top: 4px; }
    .clinical-note {
        background: #EEF5F7;
        color: #486371;
        border-radius: 11px;
        padding: 10px 12px;
        font-size: .76rem;
        line-height: 1.55;
        margin-top: 12px;
    }
    div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button {
        min-height: 44px;
        border-radius: 12px;
        font-weight: 740;
    }
    .footer-note { color: #70808C; text-align: center; font-size: .75rem; margin-top: 30px; }
    @media (max-width: 760px) {
        .block-container { padding: 1rem .8rem 2rem; }
        .hero { padding: 24px 20px; border-radius: 18px; }
        .hero-title { font-size: 1.55rem; }
        div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] { padding: 17px; border-radius: 16px; }
        .risk-number { font-size: 2.8rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    preprocessor = pipeline.named_steps["preprocess"]
    forest = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(forest)
    transformed_names = list(preprocessor.get_feature_names_out())
    groups = {
        feature: [
            index
            for index, name in enumerate(transformed_names)
            if name == feature or name.startswith(f"{feature}_")
        ]
        for feature in FEATURES
    }
    if any(not indices for indices in groups.values()):
        raise RuntimeError("模型预处理字段与应用字段不一致。")
    return pipeline, metadata, preprocessor, forest, explainer, groups


def display_value(feature: str, value: int | float) -> str:
    if feature in VALUE_LABELS:
        return VALUE_LABELS[feature][int(value)]
    if feature in UNITS:
        numeric = int(value) if float(value).is_integer() else float(value)
        return f"{numeric}{UNITS[feature]}"
    return str(value)


def calculate_prediction(model_input: dict[str, int | float]) -> dict:
    pipeline, metadata, preprocessor, forest, explainer, groups = load_artifacts()
    frame = pd.DataFrame([model_input], columns=FEATURES)
    probability = float(pipeline.predict_proba(frame)[0, 1])
    transformed = np.asarray(preprocessor.transform(frame), dtype=float)
    explanation = explainer(transformed, check_additivity=False)
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        positive_values = values[0, :, 1]
        base_probability = float(base_values[0, 1])
    else:
        positive_values = values[0]
        base_probability = float(np.ravel(base_values)[0])
    contributions = np.asarray(
        [positive_values[groups[feature]].sum() for feature in FEATURES], dtype=float
    )
    reconstructed = base_probability + float(contributions.sum())
    if abs(reconstructed - probability) > 1e-6:
        raise RuntimeError("个体解释与模型预测概率不一致。")
    return {
        "probability": probability,
        "threshold": float(metadata["threshold"]),
        "base_probability": base_probability,
        "contributions": contributions,
        "model_input": model_input,
    }


def make_shap_figure(result: dict):
    values = result["contributions"]
    model_input = result["model_input"]
    explanation = shap.Explanation(
        values=values,
        base_values=result["base_probability"],
        data=np.asarray([display_value(feature, model_input[feature]) for feature in FEATURES]),
        feature_names=[LABELS[feature] for feature in FEATURES],
    )
    font_candidates = []
    for font_path in (
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_candidates.append(font_manager.FontProperties(fname=str(font_path)).get_name())

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_candidates + [
                "WenQuanYi Zen Hei",
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    figure = plt.figure(figsize=(7.0, 5.1), facecolor="white")
    shap.plots.waterfall(explanation, max_display=10, show=False)
    axis = plt.gca()
    axis.set_xlabel("对一年内非计划再入院预测概率的贡献")
    axis.grid(axis="x", color="#DCE6ED", linewidth=0.6, alpha=0.85)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    return figure


def factor_rows(result: dict, positive: bool, limit: int = 3) -> list[tuple[str, str, float]]:
    values = result["contributions"]
    indices = [i for i, value in enumerate(values) if (value > 0 if positive else value < 0)]
    indices.sort(key=lambda i: abs(values[i]), reverse=True)
    rows = []
    for index in indices[:limit]:
        feature = FEATURES[index]
        rows.append(
            (
                LABELS[feature],
                display_value(feature, result["model_input"][feature]),
                float(values[index]),
            )
        )
    return rows


def factor_card(name: str, value: str, contribution: float, positive: bool) -> str:
    class_name = "factor-card" if positive else "factor-card protective"
    delta_class = "factor-delta-up" if positive else "factor-delta-down"
    direction = "增加" if positive else "降低"
    return (
        f'<div class="{class_name}">'
        f'<div class="factor-name">{html.escape(name)}</div>'
        f'<div class="factor-value">{html.escape(value)} · '
        f'<span class="{delta_class}">{direction} {abs(contribution) * 100:.1f} 个百分点</span></div>'
        "</div>"
    )


def make_recommendations(result: dict) -> list[tuple[str, str, bool]]:
    """Create clinician-facing review prompts, not treatment orders."""
    data = result["model_input"]
    contribution = dict(zip(FEATURES, result["contributions"]))
    high_risk = result["probability"] >= result["threshold"]
    items: list[tuple[str, str, bool]] = []

    if high_risk:
        items.append((
            "加强风险复核与随访衔接",
            "模型结果高于分类阈值。建议由主管医护人员结合当前精神症状、自伤或自杀风险、家庭支持及治疗依从性进行复核，明确出院后的责任人员、首次联系时间和异常情况处置路径。",
            True,
        ))
    else:
        items.append((
            "维持规范随访并动态复核",
            "本次预测低于分类阈值，但不能排除临床风险。建议按院内规范完成出院评估与随访安排；症状、家庭环境或安全状态变化时应重新评估。",
            False,
        ))

    if int(data["既往自杀未遂史(X26)"]) == 1:
        items.append((
            "优先完成自伤与自杀安全评估",
            "患者有既往自杀未遂史。建议直接询问当前自伤/自杀想法、计划与可获得手段，制定可执行的安全计划并与监护人沟通环境安全；如存在即刻危险，立即启动院内急诊或危机处置流程。",
            True,
        ))

    symptom_flags = []
    if int(data["抑郁程度(X13)"]) >= 2:
        symptom_flags.append(display_value("抑郁程度(X13)", data["抑郁程度(X13)"]))
    if int(data["伴焦虑症状(X15)"]) >= 2:
        symptom_flags.append(f"焦虑{display_value('伴焦虑症状(X15)', data['伴焦虑症状(X15)'])}")
    if int(data["出院情况"]) == 2:
        symptom_flags.append("出院时未愈")
    if symptom_flags:
        items.append((
            "复核症状控制与出院准备度",
            f"当前记录提示：{'、'.join(symptom_flags)}。建议复核残留症状、共病、用药与心理治疗衔接、家庭照护能力及复诊可及性，由临床团队决定是否需要强化监测或调整随访计划。",
            True,
        ))

    if int(data["既往住院次数(X24)"]) >= 1:
        items.append((
            "针对既往再住院经历制定预警方案",
            "回顾既往住院前的早期预警信号、诱因与就医障碍，将可识别的预警表现、联系人和快速返院路径写入出院计划，并向患者及监护人说明。",
            contribution["既往住院次数(X24)"] > 0,
        ))

    if int(data["既往门诊治疗次数"]) <= 1 or int(data["医保类型(X9)"]) == 4:
        items.append((
            "评估连续照护与就医可及性",
            "建议核实预约落实、交通与费用负担、监护人陪诊、药物获得及失访风险，必要时联动门诊、社区或社会工作资源。医保信息仅用于提示资源可及性评估，不代表因果关系。",
            contribution["既往门诊治疗次数"] > 0 or contribution["医保类型(X9)"] > 0,
        ))

    items.append((
        "开展患者与家庭早期预警教育",
        "共同识别睡眠、情绪、行为、拒学或社会退缩等个体化复发信号，明确何时联系随访团队、何时紧急就医，并记录可获得的支持人员与联系方式。",
        False,
    ))
    return items[:5]


def advice_card(title: str, body: str, priority: bool) -> str:
    class_name = "advice-card priority" if priority else "advice-card"
    return (
        f'<div class="{class_name}"><div class="advice-title">{html.escape(title)}</div>'
        f'<div class="advice-body">{html.escape(body)}</div></div>'
    )


def report_html(patient_name: str, admission_number: str, result: dict, figure) -> bytes:
    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plot_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    probability = result["probability"]
    threshold = result["threshold"]
    risk_text = "较高风险" if probability >= threshold else "较低风险"
    rows = "".join(
        f"<tr><td>{html.escape(LABELS[feature])}</td><td>{html.escape(display_value(feature, result['model_input'][feature]))}</td></tr>"
        for feature in FEATURES
    )
    advice_rows = "".join(
        f"<li><strong>{html.escape(title)}</strong><br>{html.escape(body)}</li>"
        for title, body, _ in make_recommendations(result)
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>再入院风险评估报告</title>
<style>
body{{font-family:'Microsoft YaHei',Arial,sans-serif;color:#243442;max-width:920px;margin:36px auto;line-height:1.65}}
h1{{color:#163A5F;border-bottom:3px solid #167D8D;padding-bottom:12px}} .meta{{display:flex;gap:36px}}
.risk{{background:#F3F7FA;border-left:5px solid #C94C4C;padding:18px 22px;margin:22px 0;border-radius:8px}}
.risk strong{{font-size:32px;color:#163A5F}} table{{width:100%;border-collapse:collapse;margin:18px 0}}
th,td{{border-bottom:1px solid #DCE6ED;padding:9px;text-align:left}} img{{width:100%;margin-top:15px}}
.advice li{{margin:12px 0;padding-left:4px}} .advice strong{{color:#163A5F}}
.note{{font-size:12px;color:#657786;margin-top:24px}} @media print{{body{{margin:12mm}}}}
</style></head><body>
<h1>青少年抑郁症患者一年内非计划再入院风险评估报告</h1>
<div class="meta"><div>姓名：{html.escape(patient_name or '未填写')}</div><div>住院号：{html.escape(admission_number or '未填写')}</div></div>
<div class="risk">预测风险：<strong>{probability * 100:.1f}%</strong><br>模型判定：{risk_text}（分类阈值 {threshold * 100:.1f}%）</div>
<h2>预测信息</h2><table><tr><th>变量</th><th>患者取值</th></tr>{rows}</table>
<h2>医护关注建议</h2><ol class="advice">{advice_rows}</ol>
<h2>个体SHAP解释</h2><img src="data:image/png;base64,{plot_data}" alt="个体SHAP解释图">
<div class="note">本报告为科研模型辅助评估结果，不替代临床判断、诊断或治疗决策。SHAP值表示各变量对本次预测结果的贡献，不代表因果关系。</div>
</body></html>"""
    return document.encode("utf-8")


if "page" not in st.session_state:
    st.session_state["page"] = "input"

is_result_page = st.session_state["page"] == "result" and "prediction_result" in st.session_state
hero_subtitle = (
    "查看本次评估结果、重点关注因素与医护措施建议"
    if is_result_page
    else "输入患者临床信息，获得个体化风险概率与可解释性结果"
)
st.markdown(
    f"""
<div class="hero">
  <div class="eyebrow">CLINICAL RISK ESTIMATION</div>
  <div class="hero-title">青少年抑郁症患者一年内非计划再入院风险评估</div>
  <p class="hero-subtitle">{hero_subtitle}</p>
</div>
""",
    unsafe_allow_html=True,
)


if not is_result_page:
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown('<div class="section-kicker">PATIENT</div><div class="section-title">患者档案</div>', unsafe_allow_html=True)
        identity_left, identity_right = st.columns(2, gap="large")
        with identity_left:
            patient_name = st.text_input("姓名", placeholder="请输入患者姓名")
        with identity_right:
            admission_number = st.text_input("住院号", placeholder="请输入住院号")

        st.markdown('<div class="identity-line"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">PREDICTORS</div><div class="section-title">预测信息</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="small-heading">病程与既往史</div>', unsafe_allow_html=True)
            onset_age = st.number_input("首次发病年龄（岁）", min_value=8, max_value=18, value=15, step=1)
            total_course = st.number_input("总病程（月）", min_value=1, max_value=72, value=13, step=1)
            prior_admissions_label = st.selectbox("既往住院次数", ["0次", "1次", "2次", "≥3次"])
            prior_outpatient_label = st.selectbox("既往门诊治疗次数", ["0次", "1次", "2次", "≥3次"], index=1)
            suicide_label = st.radio("既往自杀未遂史", ["无", "有"], horizontal=True)
        with right:
            st.markdown('<div class="small-heading">本次住院与临床情况</div>', unsafe_allow_html=True)
            stay_length = st.number_input("本次住院时长（天）", min_value=3, max_value=52, value=19, step=1)
            discharge_label = st.selectbox("出院情况", ["好转", "未愈"])
            depression_label = st.selectbox("抑郁程度", ["轻度", "中度", "重度"], index=1)
            anxiety_label = st.selectbox("伴焦虑症状", ["无", "轻度", "中度", "重度"], index=1)
            insurance_label = st.selectbox("医保类型", ["新农合", "自费", "其他社会保险"])

        submitted = st.form_submit_button("开始风险评估", width="stretch")

    if submitted:
        inverse = lambda mapping, label: next(code for code, text in mapping.items() if text == label)
        model_input = {
            "本次住院时长(X12,天)": int(stay_length),
            "既往住院次数(X24)": inverse(VALUE_LABELS["既往住院次数(X24)"], prior_admissions_label),
            "既往自杀未遂史(X26)": inverse(VALUE_LABELS["既往自杀未遂史(X26)"], suicide_label),
            "首次发病年龄(X18,岁)": int(onset_age),
            "出院情况": inverse(VALUE_LABELS["出院情况"], discharge_label),
            "伴焦虑症状(X15)": inverse(VALUE_LABELS["伴焦虑症状(X15)"], anxiety_label),
            "医保类型(X9)": inverse(VALUE_LABELS["医保类型(X9)"], insurance_label),
            "既往门诊治疗次数": inverse(VALUE_LABELS["既往门诊治疗次数"], prior_outpatient_label),
            "抑郁程度(X13)": inverse(VALUE_LABELS["抑郁程度(X13)"], depression_label),
            "总病程(X25,月)": int(total_course),
        }
        st.session_state["prediction_result"] = calculate_prediction(model_input)
        st.session_state["patient_name"] = patient_name
        st.session_state["admission_number"] = admission_number
        st.session_state["page"] = "result"
        st.rerun()


if is_result_page:
    result = st.session_state["prediction_result"]
    probability = result["probability"]
    threshold = result["threshold"]
    high_risk = probability >= threshold
    patient_name = st.session_state.get("patient_name", "")
    admission_number = st.session_state.get("admission_number", "")
    patient_caption = " · ".join(part for part in [patient_name, admission_number] if part) or "当前评估"

    result_box = st.container(border=True)
    result_box.markdown('<div class="section-kicker">ASSESSMENT RESULT</div><div class="result-heading">本次风险评估结果</div><div class="result-subtitle">请结合临床评估、病程变化与家庭支持情况综合判断</div>', unsafe_allow_html=True)
    overview, explanation_column = result_box.columns([1.0, 1.45], gap="large")
    with overview:
        st.markdown(f'<div class="patient-chip">{html.escape(patient_caption)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="risk-label">一年内非计划再入院预测风险</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="risk-number">{probability * 100:.1f}<span>%</span></div>', unsafe_allow_html=True)
        state_class = "risk-state-high" if high_risk else "risk-state-low"
        state_text = "模型判定 · 较高风险" if high_risk else "模型判定 · 较低风险"
        st.markdown(f'<div class="{state_class}">{state_text}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="risk-track-wrap">
  <div class="risk-track">
    <div class="threshold-marker" style="left:{threshold * 100:.2f}%"></div>
    <div class="risk-marker" style="left:{probability * 100:.2f}%"></div>
  </div>
  <div class="risk-scale"><span>0%</span><span>50%</span><span>100%</span></div>
  <div class="threshold-caption">分类阈值 {threshold * 100:.1f}% · 当前风险 {probability * 100:.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with explanation_column:
        increase, decrease = st.columns(2, gap="medium")
        with increase:
            st.markdown('<div class="small-heading">主要风险升高因素</div>', unsafe_allow_html=True)
            rows = factor_rows(result, positive=True)
            if rows:
                for row in rows:
                    st.markdown(factor_card(*row, positive=True), unsafe_allow_html=True)
            else:
                st.caption("本次预测未出现明显的风险升高贡献。")
        with decrease:
            st.markdown('<div class="small-heading">主要风险降低因素</div>', unsafe_allow_html=True)
            rows = factor_rows(result, positive=False)
            if rows:
                for row in rows:
                    st.markdown(factor_card(*row, positive=False), unsafe_allow_html=True)
            else:
                st.caption("本次预测未出现明显的风险降低贡献。")

    result_box.markdown("---")
    advice_column, shap_column = result_box.columns([1.25, 1.0], gap="large")
    with advice_column:
        st.markdown('<div class="section-kicker">CLINICAL ACTION</div><div class="section-title">医护关注与措施建议</div>', unsafe_allow_html=True)
        for title, body, priority in make_recommendations(result):
            st.markdown(advice_card(title, body, priority), unsafe_allow_html=True)
        st.markdown('<div class="clinical-note">建议内容用于提示需要复核的临床环节，不构成自动医嘱。具体处置应由有资质的医护团队结合实时风险评估及院内制度决定。</div>', unsafe_allow_html=True)
    with shap_column:
        st.markdown('<div class="section-kicker">INTERPRETATION</div><div class="section-title">个体SHAP解释</div>', unsafe_allow_html=True)
        st.caption("显示10个变量对本次预测的方向与贡献；贡献不等于因果作用。")
        shap_figure = make_shap_figure(result)
        st.pyplot(shap_figure, width="stretch")

    report = report_html(patient_name, admission_number, result, shap_figure)
    result_box.markdown("---")
    download_left, restart, download_right = result_box.columns([1, 1, 1.6], gap="medium")
    with download_left:
        st.download_button(
            "下载评估报告",
            data=report,
            file_name=f"再入院风险评估_{admission_number or '未编号'}.html",
            mime="text/html",
            width="stretch",
        )
    with restart:
        if st.button("重新评估", width="stretch", key="restart_assessment"):
            for key in ["prediction_result", "patient_name", "admission_number"]:
                st.session_state.pop(key, None)
            st.session_state["page"] = "input"
            st.rerun()
    with download_right:
        st.caption("下载后可使用浏览器打开并打印或另存为PDF。")


st.markdown(
    '<div class="footer-note">研究原型 v1.0 · 预测结果用于辅助风险评估，不替代临床判断</div>',
    unsafe_allow_html=True,
)
