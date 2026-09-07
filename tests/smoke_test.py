from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import pandas as pd


APP_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("readmission_app", APP_DIR / "app.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to import app.py")
APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP)

sample = {
    "本次住院时长(X12,天)": 19,
    "既往住院次数(X24)": 1,
    "既往自杀未遂史(X26)": 0,
    "首次发病年龄(X18,岁)": 15,
    "出院情况": 1,
    "伴焦虑症状(X15)": 1,
    "医保类型(X9)": 3,
    "既往门诊治疗次数": 1,
    "抑郁程度(X13)": 2,
    "总病程(X25,月)": 13,
}

result = APP.calculate_prediction(sample)
pipeline = joblib.load(APP.MODEL_PATH)
expected = float(pipeline.predict_proba(pd.DataFrame([sample], columns=APP.FEATURES))[0, 1])
assert abs(result["probability"] - expected) < 1e-12
assert abs(result["base_probability"] + result["contributions"].sum() - expected) < 1e-6
assert len(result["contributions"]) == 10
assert APP.FEATURES == list(sample)

figure = APP.make_shap_figure(result)
assert figure.axes
report = APP.report_html("测试患者", "TEST-001", result, figure).decode("utf-8")
assert "测试患者" in report
assert "TEST-001" in report
assert f"{expected * 100:.1f}%" in report

print(f"SMOKE_TEST_OK probability={expected:.6f} threshold={result['threshold']:.6f}")
