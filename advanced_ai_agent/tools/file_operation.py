import ollama
import os
import io
import re
import contextlib
from datetime import datetime

# === 許可フォルダ設定 ===
# ALLOWED_DIR = r"C:\safe_folder"  # LLMが操作できるフォルダ
ALLOWED_DIR = r"./safe_folder"  # LLMが操作できるフォルダ
LOG_FILE = os.path.join(ALLOWED_DIR, "operation_log.txt")

LLM_MODEL = "gemma3:4b"

TOOL_NAME = "operate"
TOOL_ALIAS = {"file_operate": "operate", "operate": "operate"}
TOOL_USING = """
operate:
- Description: Operates on files in a specific folder on the PC.
- Input: Folder status check (Folder_Check) or file operation (Read/Create/Write/Update/Delete) + file name (string) (e.g., "Create-schedule.txt")
- Output: Processing results.
"""

# === ログ関数 ===
def log_action(action: str):
  os.makedirs(ALLOWED_DIR, exist_ok=True)
  with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(f"[{datetime.now()}] {action}\n")

# === パス検証関数 ===
def is_safe_path(path: str) -> bool:
  abs_path = os.path.abspath(path)
  return abs_path.startswith(os.path.abspath(ALLOWED_DIR))

# === 安全なファイル操作クラス ===
class SafeFileOps:
  def create_file(self, path: str, content: str = ""):
    if not is_safe_path(path):
      raise PermissionError("指定フォルダ外へのアクセスは禁止されています。")
    with open(path, "w", encoding="utf-8") as f:
      f.write(content)
    log_action(f"Create: {path}")
    return f"ファイルを作成しました: {path}"

  def read_file(self, path: str) -> str:
    if not is_safe_path(path):
      raise PermissionError("指定フォルダ外へのアクセスは禁止されています。")
    with open(path, "r", encoding="utf-8") as f:
      content = f.read()
    log_action(f"Read: {path}")
    return content

  def write_file(self, path: str, content: str):
    if not is_safe_path(path):
      raise PermissionError("指定フォルダ外へのアクセスは禁止されています。")
    with open(path, "a", encoding="utf-8") as f:
      f.write(content)
    log_action(f"Write: {path}")
    return f"ファイルに追記しました: {path}"

  def delete_file(self, path: str):
    if not is_safe_path(path):
      raise PermissionError("指定フォルダ外へのアクセスは禁止されています。")
    os.remove(path)
    log_action(f"Delete: {path}")
    return f"ファイルを削除しました: {path}"

  def list_files(self, folder: str = ALLOWED_DIR) -> str:
    """フォルダ内のファイル・サブフォルダ一覧を取得"""
    if not is_safe_path(folder):
      raise PermissionError("指定フォルダ外へのアクセスは禁止されています。")
    items = []
    for root, dirs, files in os.walk(folder):
      rel_root = os.path.relpath(root, ALLOWED_DIR)
      for d in dirs:
        items.append(f"DIR: {os.path.join(rel_root, d)}")
      for f in files:
        items.append(f"FILE: {os.path.join(rel_root, f)}")
    log_action(f"List: {folder}")
    return "\n".join(items)


# === ファイル操作クラス ===
class FileOperateion:
  @staticmethod
  def name():
    return TOOL_NAME
  
  @staticmethod
  def alias():
    return TOOL_ALIAS

  @staticmethod
  def using():
    return TOOL_USING
  
  # === コード安全チェック ===
  def check_safe_code(code: str):
    forbidden = ["__import__", "eval(", "exec(", "os.system", "subprocess", "open("]
    for word in forbidden:
      if word in code:
        raise ValueError(f"危険なコードが検出されました: {word}")
    return code


  # === メイン実行関数 ===
  def llm_file_executor(user_request: str, model_name: str = "llama3"):
    """
    LLMが指定フォルダ内で安全にファイル操作を実行。
    フォルダ構造と日付情報をもとに予定表などのタスクを処理可能。
    """

    # --- 現在の日時 ---
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    # --- 現在のフォルダ構造を取得 ---
    file_ops = SafeFileOps()
    folder_structure = file_ops.list_files()

    # --- LLMに渡すプロンプト ---
    query = f"""
  あなたは安全なローカルファイル操作AIです。
  現在の日時: {date_str} {time_str}
  操作可能フォルダ: {ALLOWED_DIR}

  フォルダ内の現在のファイル構成:
  {folder_structure}

  使用できるメソッド:
  - file_ops.create_file(path, content)
  - file_ops.read_file(path)
  - file_ops.write_file(path, content)
  - file_ops.delete_file(path)
  - file_ops.list_files(folder)

  制約:
  - 指定フォルダ外にはアクセス禁止
  - 出力はPythonコードとして ```python ... ``` で囲んで出力
  - 実行結果は print または result に入れること

  ユーザーの依頼:
  {user_request}
  """

    # --- ① LLMがコードを生成 ---
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": query}])
    content = response["message"]["content"]

    # --- ② コード抽出 ---
    code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
    if not code_match:
      raise ValueError("Pythonコードが出力されませんでした:\n" + content)
    code = code_match.group(1).strip()
    check_safe_code(code)

    print("=== 生成されたコード ===\n", code)

    # --- ③ execを安全に実行し出力をキャプチャ ---
    safe_globals = {
      "__builtins__": {"print": print, "math": __import__("math")},
      "file_ops": file_ops,
    }
    local_vars = {}
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
      exec(code, safe_globals, local_vars)

    exec_output = output_buffer.getvalue()
    result = local_vars.get("result")

    # --- ④ 結果報告をLLMに生成させる ---
    summary_prompt = f"""
  次の操作の結果を日本語で簡潔に報告してください。

  日時: {date_str} {time_str}
  ユーザーの依頼: {user_request}
  実行コード出力:
  {exec_output}
  result変数の内容:
  {result}
  """

    summary = ollama.chat(model=model_name, messages=[{"role": "user", "content": summary_prompt}])
    summary_text = summary["message"]["content"]

    log_action(f"LLM実行完了: {user_request}")
    return summary_text

# === 使用例 ===
if __name__ == "__main__":
  ops = FileOperateion()
  response = ops.llm_file_executor("今日の15時に会議があることを予定表に書き込んでください。")
  print("\n=== LLMの報告 ===\n", response)
