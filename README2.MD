# Legal Agent Lab (本地法务合同智能审查系统)

`Legal Agent Lab` 是一套完全运行在本地、高度注重隐私保护与法律法规穿透比对的 **AI 合同审查智能体系统**。

系统基于本地私有化大语言模型（Qwen2.5-7B-Instruct-AWQ），通过自主驱动的 **ReAct (Thought -> Action -> Observation)** 状态机，结合 **权威民法典法规检索**、**企业合规手册核查** 与 **SQLite 企业自编法典红线记忆**，对多模态合同（Word / PDF / 扫描件图片 OCR / TXT）实现自动条款切分、逐条穿透审查并输出结构化法律合规报告。

---

## 🌟 核心特性

- 🔒 **100% 离线与数据合规**：全流程运行于本地环境（WSL2 / Linux），模型权重与企业合同原文绝不上云。
- ⚖️ **三维合规穿透工具链**：
  1. `search_civil_code`：精确提取国家法律法规 PDF 的条文、页码与原文。
  2. `get_company_policy`：核对公司内部合规手册与政策要求。
  3. `get_past_review_rules`：基于 SQLite 的企业自编法典，沉淀特有业务审查偏好与历史红线（如违约金限额、管辖权限制）。
- 📑 **多模态文档加载与条款切分**：
  - 支持 `.docx`、`.pdf`、`.txt` 以及常见格式合同图片（`.png`, `.jpg`, `.jpeg` 等）。
  - 内置本地轻量级 OCR（RapidOCR），离线毫秒级抽取合同扫描件文字。
  - 基于法律文本特征规则，自动将合同精准切分为“第一条、第二条…”语义条款。
- ⚡ **高性能并发网关与打字机流式推送**：
  - 基于 FastAPI 封装异步网关，提供会话隔离与滑动窗口 Token 治理。
  - 基于 SSE (Server-Sent Events) 协议，将 Agent 的思考流、工具调用与审查结果实时推送到 Web 端。
- 🖥️ **法务双栏可视化工作台**：
  - 基于 Streamlit 构建，左栏管理合同原文与条款拆解，右栏动态展示智能体思考轨迹与高/中/低风险红线卡片。

---

## 🏗️ 系统架构与目录结构

```text
legal_agent_lab/
├── configs/
│   └── config.py                 # 全局配置中心 (统一管理路径、端口与超参数)
├── server/
│   └── vllm_launcher.py          # vLLM 本地推理引擎启动器 (FP16 KV Cache / 4096 tokens)
├── core/
│   ├── schemas.py                # Pydantic v2 强类型数据契约
│   ├── prompts.py                # Agent 系统提示词与 ReAct 协议解耦模块
│   └── agent_loop.py             # 核心 ReAct 推理状态机与工具调度引擎
├── services/
│   ├── pdf_kb.py                 # 本地 PDF 知识库服务 (法规与政策按页码抽取)
│   ├── rule_book.py              # 企业自编法典与审查偏好记忆库 (SQLite 持久化)
│   └── doc_loader.py             # 多模态合同解析与条款结构化切分器 (含 OCR 与自测引擎)
├── gateway/
│   ├── session_manager.py        # 多租户会话隔离与滑动窗口上下文管理器
│   └── api_server.py             # FastAPI 异步并发网关 (支持 REST 与 SSE 流式接口)
├── web/
│   └── app_ui.py                 # Streamlit 双栏可视化法务工作台
├── reference_docs/               # 权威民法典法规与公司合规 PDF 目录
├── models/                       # 本地 AWQ 量化模型存放目录
└── data/                         # SQLite 数据库与上传临时文件目录