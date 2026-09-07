# 青少年抑郁症患者一年内非计划再入院风险评估工具

本目录是用于学位论文展示的 Streamlit 公网演示版本。应用保留“姓名”和“住院号”输入框，供演示时区分不同评估对象；两项信息不进入模型，也不参与 SHAP 计算。应用不连接数据库，不主动持久化用户输入。

## 公网访问

<https://adolescent-readmission-risk-tool.streamlit.app/>

应用在 Streamlit Community Cloud 上以公开模式运行。首次访问或长时间无人使用后的首次唤醒可能需要等待片刻。

## 本地启动

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Community Cloud 部署

1. 将本目录内容放入一个 GitHub 仓库。
2. 登录 <https://share.streamlit.io/> 并连接该仓库。
3. 入口文件选择 `app.py`。
4. Python 版本选择 3.12，与当前已验证的云端运行环境保持一致。
5. 部署后运行 `python tests/smoke_test.py` 对应的等价测试，并分别用电脑和手机浏览器检查输入、结果、SHAP图、报告下载与重新评估。

## 发布范围

公开仓库或部署包仅包含：应用代码、冻结模型、模型元数据、依赖声明和测试代码。不得加入建模数据、外部验证数据、真实患者资料、评估报告或运行日志。

## 模型说明

- 最终算法：随机森林
- 预测变量：10项
- 分类阈值：由 `model/model_metadata.json` 冻结管理
- 输出：预测概率、风险分层、SHAP个体解释和医护关注建议

本工具为科研演示应用，输出用于辅助风险评估，不替代临床判断、诊断或治疗决策。
