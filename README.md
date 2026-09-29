# simple-agent

一个最小可运行的 ReAct Agent：模型通过「思考 → 行动 → 观察」的循环自主完成任务，并调用三个内置工具与文件系统交互。

## 核心特性

- **ReAct 循环**：模型用 `<thought>` / `<action>` / `<observation>` 标签驱动，直到给出 `<final_answer>`
- **三个工具**：
  - `read_file`：读取文件内容（自动识别 UTF-8 / GBK，超长自动截断）
  - `write_to_file`：写入文件（自动创建父目录，统一 UTF-8 + LF）
  - `run_terminal_command`：执行终端命令并返回真实输出
- **统一工作目录**：所有相对路径与终端命令都基于命令行传入的工作目录，生成的文件不会影响项目根目录
- **跨平台**：适配 macOS / Linux / Windows 的路径、编码与 shell
- **安全兜底**：写入限制在工作目录内；终端命令需人工确认；超时自动终止进程树

## 运行方法

首先请确保你已经安装了 uv，如果没有的话，请按以下页面的要求安装：

https://docs.astral.sh/uv/guides/install-python/

然后在当前目录下，新建一个叫做 .env 的文件，输入以下内容：

```
DEEPSEEK_API_KEY=xxx
```

xxx 就是你在 DeepSeek 官方平台申请到的 API Key。项目默认直连 DeepSeek 官方 API（base_url 为 https://api.deepseek.com），如需更换模型或服务商，直接修改 agent.py 中的 base_url 和 model 即可。

确保 uv 已经安装成功后，进入到当前文件所在目录，然后执行以下命令即可启动：

```bash
uv run agent.py snake
```

其中 `snake` 是工作目录（可换成任意目录名），**不存在时会自动创建**；Agent 生成的文件都落在其中，不会影响项目根目录，该目录还会被**自动追加到 `.gitignore`**（幂等），避免运行产物被提交。

## 项目结构

```
agent.py            # Agent 主体与三个工具：read_file / write_to_file / run_terminal_command
prompt_template.py  # ReAct 系统提示词模板
pyproject.toml      # 依赖声明（click / openai / python-dotenv）
uv.lock             # 依赖锁定
```