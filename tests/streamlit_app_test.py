from pathlib import Path

from streamlit.testing.v1 import AppTest


app_path = Path(__file__).resolve().parents[1] / "app.py"
app = AppTest.from_file(str(app_path), default_timeout=120)
app.run()
assert not app.exception, app.exception
assert len(app.text_input) == 2
assert len(app.number_input) == 3
assert len(app.selectbox) == 6
assert len(app.radio) == 1

app.text_input[0].set_value("测试患者")
app.text_input[1].set_value("TEST-001")
app.button[0].click()
app.run()
assert not app.exception, app.exception
assert len(app.text_input) == 0, "结果页不应继续显示患者输入表单"
values = [item.value for item in app.markdown]
assert any("0.5" in value for value in values)
assert any("医护关注与措施建议" in value for value in values)
assert any("个体SHAP解释" in value for value in values)
assert any(button.label == "重新评估" for button in app.button)

restart = next(button for button in app.button if button.label == "重新评估")
restart.click()
app.run()
assert not app.exception, app.exception
assert len(app.text_input) == 2, "重新评估后应返回输入页"
assert len(app.number_input) == 3

print("STREAMLIT_APP_TEST_OK")
