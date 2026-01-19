import ollama
import re

LLM_MODEL = "gemma3:4b"

def check_safe_code(code: str) -> str:
  """
  LLMが生成したコードに対して安全チェックを行う。
  - import は math のみ許可
  - 危険な関数・モジュール呼び出しを禁止
  """
  forbidden_patterns = [
    "open(",
    "os.",
    "sys.",
    "subprocess",
    "shutil",
    "socket",
    "eval(",
    "exec(",
    "compile(",
    "__import__",
    "globals(",
    "locals(",
    "input(",
  ]

  # 行ごとに import を検査（math のみ許可）
  lines = code.splitlines()
  safe_code_lines = []
  for line in lines:
    if line.strip().startswith("import "):
      if line.strip() != "import math":
        raise ValueError(f"許可されていない import が検出されました: {line.strip()}")
    safe_code_lines.append(line)
  code = "\n".join(safe_code_lines)

  # 禁止パターン検出
  for pattern in forbidden_patterns:
    if pattern in code:
      raise ValueError(f"危険なコードが検出されました: {pattern}")

  return code

# コードインタープリターで計算を行う
def code_calc(expression:str, num=0, agent=None):

  # --- ① LLMへのプロンプト ---
  query = f"""
次の式をPythonで正確に計算してください。
計算対象: {expression}

条件:
- 必要なら import math は使用してよいです。
- math 以外のライブラリは import してはいけません。
- 計算結果は変数 result に代入してください。
- 実行可能なPythonコードを ```python ... ``` の形式で出力してください。
- Pythonコード以外の余計な説明は不要です。
"""
# no_think

  # --- ② LLM呼び出し ---
  if agent is not None:
    content = agent.generate(query, sys_use=False)
  else:
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": query}])
    content = response["message"]["content"]

  # --- ③ コードブロック抽出 ---
  code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
  if not code_match:
    return "計算に失敗しました。"
    # raise ValueError("Pythonコードが出力されませんでした:\n" + content)
    
  code = code_match.group(1).strip()

  print("=== LLMが生成したコード ===")
  print(code)

  # --- ④ 安全チェック ---
  code = check_safe_code(code)

  # --- ⑤ 制限付き環境でコード実行 ---
  safe_globals = {
    "__builtins__": {
      "abs": abs,
      "round": round,
      "pow": pow,
      "range": range,
      "math": __import__("math"),
      "print": print,
    }
  }
  local_vars = {}
  exec(code, safe_globals, local_vars)

  # --- ⑥ 結果取得 ---
  result = local_vars.get("result", None)
  if result is None:
    raise ValueError("result 変数が見つかりません。LLM出力を確認してください。")

  # print("\n=== 計算結果 ===")
  # print(result)
  return f"計算結果: {result}"


# --- 使用例 ---
if __name__ == "__main__":
  code_calc("1234567890 * 9876543210")
