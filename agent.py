import ast
import inspect
import os
import platform
import re
import shutil
import subprocess
from string import Template
from typing import List, Callable, Tuple

import click
from dotenv import load_dotenv
from openai import OpenAI

from prompt_template import react_system_prompt_template


class ReActAgent:
    def __init__(self, tools: List[Callable], model: str):
        self.tools = { func.__name__: func for func in tools }
        self.model = model
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=ReActAgent.get_api_key(),
        )

    def run(self, user_input: str):
        messages = [
            {"role": "system", "content": self.render_system_prompt(react_system_prompt_template)},
            {"role": "user", "content": f"<question>{user_input}</question>"}
        ]

        # 限制最大步数，避免模型陷入无限循环
        for _ in range(MAX_STEPS):

            # 请求模型
            content = self.call_model(messages)

            # 检测 Thought
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
                print(f"\n\n💭 Thought: {thought}")

            # 检测模型是否输出 Final Answer，如果是的话，直接返回
            final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
            if final_answer:
                return final_answer.group(1)

            # 检测 Action
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                # 模型未按格式输出时不崩溃，提示其纠正后重试（受 MAX_STEPS 限制）
                messages.append({"role": "user", "content": "请使用 <action> 调用工具，或在完成时给出 <final_answer>。"})
                continue
            action = action_match.group(1)
            tool_name, args = self.parse_action(action)

            print(f"\n\n🔧 Action: {tool_name}({', '.join(map(str, args))})")
            # 只有终端命令才需要询问用户，其他的工具直接执行
            should_continue = input(f"\n\n是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消。")
                return "操作被用户取消"

            try:
                observation = self.tools[tool_name](*args)
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
            print(f"\n\n🔍 Observation：{observation}")
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})

        return f"已达到最大步数限制（{MAX_STEPS} 步），任务未完成。"

    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """渲染系统提示模板，替换变量"""
        tool_list = self.get_tool_list()
        try:
            names = sorted(os.listdir(WORKSPACE_ROOT))
        except OSError:
            names = []
        # 只列出相对文件名，与提示词中“优先使用相对路径”的指引保持一致
        file_list = ", ".join(names) or "（空目录）"
        return Template(system_prompt_template).substitute(
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            file_list=file_list
        )

    @staticmethod
    def get_api_key() -> str:
        """Load the API key from an environment variable."""
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("未找到 DEEPSEEK_API_KEY 环境变量，请在 .env 文件中设置。")
        return api_key

    def call_model(self, messages):
        print("\n\n正在请求模型，请稍等...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        except Exception as e:
            raise RuntimeError(f"调用模型失败：{e}") from e
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return content

    def parse_action(self, code_str: str) -> Tuple[str, list]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 手动解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        i = 0
        paren_depth = 0
        
        while i < len(args_str):
            char = args_str[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层逗号，结束当前参数
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i-1] != '\\'):
                    in_string = False
                    string_char = None
            
            i += 1
        
        # 添加最后一个参数
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))
        
        return func_name, args
    
    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()
        
        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str
        
        # 尝试使用 ast.literal_eval 解析其他类型
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            return arg_str

    def get_operating_system_name(self):
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }

        return os_map.get(platform.system(), "Unknown")


COMMAND_TIMEOUT = 120        # 终端命令最长执行时间（秒）
MAX_READ_CHARS = 100_000     # read_file 单次返回给模型的最大字符数
MAX_OUTPUT_CHARS = 20_000    # 终端命令输出返回给模型的最大字符数
MAX_STEPS = 30               # 单次任务最大步数，防止模型陷入无限循环

# 统一工作根：相对路径、终端命令与提示词文件列表的唯一基准，main() 会用命令行传入的目录覆盖它
WORKSPACE_ROOT = os.path.abspath(os.getcwd())
# 是否把读写限制在工作根内，防止误写系统目录
RESTRICT_TO_WORKSPACE = True


def set_workspace_root(path: str) -> str:
    """设置统一工作根（通常是命令行传入的工作目录），返回其绝对路径"""
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = os.path.abspath(os.path.expanduser(path))
    return WORKSPACE_ROOT


def _add_to_gitignore(path: str) -> None:
    """把工作目录追加到 agent 所在目录的 .gitignore（幂等，失败不阻断）"""
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        rel = os.path.relpath(path, repo_dir)
        if rel in (".", "..") or rel.startswith(".." + os.sep):
            return  # 仓库外或仓库本身，忽略无意义
        entry = rel.replace(os.sep, "/").rstrip("/") + "/"  # gitignore 用正斜杠，目录以 / 结尾

        gitignore = os.path.join(repo_dir, ".gitignore")
        lines = []
        if os.path.isfile(gitignore):
            with open(gitignore, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        if entry in (line.strip() for line in lines):
            return  # 已存在

        suffix = "" if not lines or lines[-1] == "" else "\n"
        with open(gitignore, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{suffix}{entry}\n")
    except OSError:
        pass


def _truncate(text: str, limit: int, label: str) -> str:
    """把过长文本截断并附加说明，避免撑爆模型上下文"""
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n...（{label}过长，已截断，仅显示前 {limit} 个字符，"
          f"实际共 {len(text)} 个字符）"
    )


def _resolve_path(file_path: str) -> str:
    """解析路径：展开 ~ 与环境变量；相对路径统一基于工作根解析（跨平台）"""
    path = os.path.expanduser(os.path.expandvars(file_path.strip()))
    if not os.path.isabs(path):
        path = os.path.join(WORKSPACE_ROOT, path)
    return os.path.normpath(path)


def _ensure_in_workspace(path: str):
    """校验路径是否位于工作根内；越界返回错误提示，合法返回 None"""
    if not RESTRICT_TO_WORKSPACE:
        return None
    root = os.path.normcase(os.path.abspath(WORKSPACE_ROOT))
    target = os.path.normcase(os.path.abspath(path))
    if target == root or target.startswith(root + os.sep):
        return None
    return f"错误：路径超出工作根范围 -> {path}（工作根：{WORKSPACE_ROOT}）"


def _workspace_cwd() -> str:
    """返回终端命令的默认执行目录（工作根）；工作根不存在时退回进程当前目录"""
    return WORKSPACE_ROOT if os.path.isdir(WORKSPACE_ROOT) else os.getcwd()


def _decode_bytes(data: bytes) -> str:
    """按常见编码依次尝试解码，避免中文文件在 Windows 下乱码"""
    for encoding in ("utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_file(file_path):
    """用于读取文件内容（相对路径基于工作根，超长自动截断）"""
    path = _resolve_path(file_path)
    denied = _ensure_in_workspace(path)
    if denied:
        return denied
    if not os.path.isfile(path):
        return f"错误：文件不存在 -> {path}"
    with open(path, "rb") as f:
        content = _decode_bytes(f.read())
    return _truncate(content, MAX_READ_CHARS, "文件内容")


def write_to_file(file_path, content):
    """将内容写入指定文件（相对路径基于工作根，自动创建父目录）"""
    path = _resolve_path(file_path)
    denied = _ensure_in_workspace(path)
    if denied:
        return denied
    # 自动创建不存在的父目录
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    # 统一以 UTF-8 + LF 写出，保证三平台文本一致
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return f"写入成功 -> {path}（共 {len(content)} 个字符）"


def _build_shell_command(command: str):
    """根据当前操作系统构造 (可执行文件, 参数列表, 是否强制 UTF-8 输出)"""
    system = platform.system()

    if system == "Windows":
        # Windows 优先使用 PowerShell（pwsh 为 PowerShell 7）
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell:
            # 显式把输出编码设为 UTF-8，避免中文输出乱码
            script = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + command
            return powershell, ["-NoProfile", "-NonInteractive", "-Command", script], True
        # 找不到 PowerShell 时退回 cmd.exe，先用 chcp 65001 切到 UTF-8 代码页
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return comspec, ["/c", f"chcp 65001 >nul && {command}"], True

    # macOS / Linux：优先使用 $SHELL（-c 为非交互模式，PATH 由 _build_env 补齐）
    shell = os.environ.get("SHELL", "")
    if not shell or not os.path.isfile(shell):
        shell = shutil.which("zsh") or shutil.which("bash") or "/bin/sh"
    return shell, ["-c", command], True


def _build_env():
    """继承当前环境变量，并补齐各平台常见工具所在的 PATH"""
    env = os.environ.copy()
    if platform.system() == "Windows":
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        python_root = os.path.join(local_appdata, "Programs", "Python")
        extra_paths = [
            os.path.join(os.environ.get("ProgramFiles", ""), "Git", "cmd"),
        ]
        # python.exe 位于版本子目录中（如 Python3xx\），需一并加入 PATH
        if os.path.isdir(python_root):
            extra_paths += [
                os.path.join(python_root, name)
                for name in sorted(os.listdir(python_root))
                if os.path.isdir(os.path.join(python_root, name))
            ]
        extra_paths.append(python_root)
    else:
        extra_paths = [
            "/opt/homebrew/bin", "/opt/homebrew/sbin",   # Apple Silicon Homebrew
            "/usr/local/bin", "/usr/local/sbin",         # Intel Homebrew
        ]

    current = env.get("PATH", "")
    existing = current.split(os.pathsep)
    missing = [p for p in extra_paths if p and p not in existing]
    if missing:
        env["PATH"] = os.pathsep.join(missing + [current])
    return env


def _kill_process_tree(proc):
    """超时后终止整个进程树（Windows 用 taskkill，POSIX 杀进程组）"""
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_terminal_command(command):
    """用于执行终端命令（跨平台：macOS / Linux / Windows），并返回真实输出结果"""
    shell, shell_args, force_utf8 = _build_shell_command(command)

    popen_kwargs = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",   # 遇到无法解码的字节也不抛异常
        env=_build_env(),
        cwd=_workspace_cwd(),  # 命令在工作根下执行，落点可预测
    )
    if force_utf8:
        popen_kwargs["encoding"] = "utf-8"
    if platform.system() == "Windows":
        # 独立进程组，便于超时后连同子进程一起终止
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen([shell] + shell_args, **popen_kwargs)
    except FileNotFoundError as e:
        return f"找不到可执行文件：{e}"

    try:
        stdout, stderr = proc.communicate(timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        # 终止整棵进程树，避免子进程残留
        _kill_process_tree(proc)
        proc.communicate()
        return f"命令执行超时（超过 {COMMAND_TIMEOUT} 秒）"

    stdout = _truncate((stdout or "").strip(), MAX_OUTPUT_CHARS, "标准输出")
    stderr = _truncate((stderr or "").strip(), MAX_OUTPUT_CHARS, "标准错误")

    if proc.returncode == 0:
        # 把真实输出返回给模型，而不是只回复“执行成功”
        return stdout if stdout else "执行成功（无输出）"

    output = [f"命令执行失败（退出码：{proc.returncode}）"]
    if stdout:
        output.append(f"标准输出：\n{stdout}")
    if stderr:
        output.append(f"标准错误：\n{stderr}")
    return "\n".join(output)


@click.command()
@click.argument('workspace_directory')
def main(workspace_directory):
    # 展开 ~ 与环境变量并转为绝对路径
    workspace_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(workspace_directory)))
    # 目录不存在则自动创建
    os.makedirs(workspace_dir, exist_ok=True)
    # 设为统一工作根：相对路径与终端命令都基于它
    set_workspace_root(workspace_dir)
    # 追加到 .gitignore，避免运行产物被提交
    _add_to_gitignore(workspace_dir)

    tools = [read_file, write_to_file, run_terminal_command]
    agent = ReActAgent(tools=tools, model="deepseek-flash")

    task = input("请输入任务：")

    final_answer = agent.run(task)

    print(f"\n\n✅ Final Answer：{final_answer}")

if __name__ == "__main__":
    main()
