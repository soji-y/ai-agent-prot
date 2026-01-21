import json
from transformers import AutoTokenizer, AutoModelForCausalLM
# from simple_llm import SimpleLLM
from simple_llm_ollama import SimpleLLMOllama
from datetime import datetime

AGENT_SYS_PROMPT = """
You can use the following tools:

1. search:
  - Description: Search the web for up-to-date information.
  - Input: search queries (string) (example: "today news weather")
  - Output: A short summary of search results.

2. calc:
  - Description: Perform basic mathematical calculations.
  - Input: An arithmetic expression (e.g., "2 + 2 * 3")
  - Output: The numeric result of the calculation.

3. answer:
  - Description: Provide the final answer to the user.
  - Input: The final response text to return to the user.

Always output your reasoning and selected action in JSON format:
{
  "thought": "...",
  "action": "search" | "calc" | "answer",
  "action_input": "..."
}
"""

# class Agent(SimpleLLM):
class Agent(SimpleLLMOllama):

  # ------------------
  # 初期化
  # ------------------
  def __init__(self, model_id_path, system_prompt, tools=None, host='http://127.0.0.1:11434'):
    super().__init__(model_id_path, system_prompt, host)

    # システムプロンプトにツールの説明を追加
    self.system_prompt += AGENT_SYS_PROMPT

    # 外部から渡されたツールを設定
    self.tools = tools.copy() if tools else {}
    
  # ------------------
  # テキストからJSONオブジェクトを抽出してパース
  # LLMは余計な説明を付けることがあるため、テキスト全体から {...} を取り出し
  # ------------------
  def _extract_json_from_text(self, text):
    start = text.find("{")
    if start == -1:
      return None
    depth = 0
    for i in range(start, len(text)):
      if text[i] == "{":
        depth += 1
      elif text[i] == "}":
        depth -= 1
        if depth == 0:
          return text[start:i+1]
    return None

  # ------------------
  # LLM の返した自由テキストを JSON として解釈
  # ------------------  
  def parse_response(self, text):
    """
    LLM の返した自由テキストを JSON として解釈し、
    {'thought':..., 'action':..., 'action_input':...} を返す。
    フォールバック: JSON にできなければ最終的に 'action': 'respond' で生テキストを返す。
    """
    # JSON 部分を抽出してパース
    json_text = self._extract_json_from_text(text)
    if json_text:
      try:
        obj = json.loads(json_text)
        # 正常にパースできたら必要キーを取り出す（存在しないキーは空文字で埋める）
        return {
          "thought": obj.get("thought", "").strip(),
          "action": obj.get("action", "").strip(),
          "action_input": obj.get("action_input", "").strip(),
          "raw": text
        }
      except Exception as e:
        # JSON 部分が不正な場合はフォールバック
        print(f"Error: {e}")
        pass

    # フォールバック：JSONが取れなければ、LLMの出力全体を action_input とする
    return {
      "thought": "",
      "action": "respond",            # 直接応答を期待する形にフォールバック
      "action_input": text.strip(),
      "raw": text
    }

  # ------------------
  # ルーター（意図解析 + 検証） -- **明示的に分離**
  # ------------------
  def route_action(self, parsed):
    """
    parsed = {'thought':..., 'action':..., 'action_input':..., 'raw':...}
    を受け取り、実行すべきアクション種別とパラメータを返す。

    戻り値例:
      {"type": "web_search", "input": "〜"}
      {"type": "calc", "input": "2+2"}
      {"type": "respond", "input": "最終回答"}
      {"type": "unknown", "input": "...", "reason": "..."}
    """
    action = (parsed.get("action") or "").lower().strip()
    action_input = parsed.get("action_input", "")

    # 別名対応（教育用の親切仕様）
    alias_map = {
      "search": "web_search",
      "web": "web_search",
      "calculate": "calc",
      "answer": "respond",
      "final": "respond"
    }
    action = alias_map.get(action, action)

    # 未指定や空文字は不明扱い
    if not action:
      return {"type": "unknown", "input": action_input, "reason": "action が空です。"}

    # respond（最終回答）はツールを呼ばずそのまま返す
    if action in ("respond", "answer"):
      return {"type": "respond", "input": action_input}

    # ツールの存在を確認してから実行する（辞書ベース）
    if action in self.tools:
      return {"type": action, "input": action_input}

    # ツール未登録の場合は unknown を返す（呼び出し元でフォールバック可能）
    return {"type": "unknown", "input": action_input, "reason": f"未登録のアクション: {action}"}

  # ------------------
  # エージェント本体（ユーザーの「目的」を受け、最大10ターンで完了を目指す）
  # ------------------
  def run(self, user_goal, max_steps=10):
    """
    user_goal: ユーザーが与えた目的（文字列）
    戻り値: 最終的な回答文字列（あるいは失敗メッセージ）
    """
    # print(f"🎯 エージェント目標: {user_goal}\n")
    thought_history = ""  # 思考・行動・結果の履歴（エージェント内部メモリ／短期）
    # tool_result = "" # ツール結果
    for step in range(max_steps):
      # LLM に投げるプロンプト（JSON 形式で出力するよう厳格に指示）
      prompt = (
        # f"{self.system_prompt}\n\n"
        "目的達成のために次の形式で JSON を出力してください。\n"
        "```\n"
        "{\n"
        "  \"thought\": \"今考えていること（ユーザーに説明する言葉）\",\n"
        "  \"action\": \"search / calc / answer / ...\",\n"
        "  \"action_input\": \"そのアクションに渡す入力\"\n"
        "}\n"
        "```\n\n"
        "上の形式だけを JSON で返してください（余計な説明は出力しないでください）。"
        f"これまでの内部履歴:\n{thought_history}\n"
        f"現在の目標: {user_goal}\n"
        f"現在の日時: {datetime.now()}"
        # "/no_think"
      )

      raw = self.generate(prompt)
      parsed = self.parse_response(raw)   # JSON 抽出 & パース
      # ここで thought/action/action_input を取り出す
      thought = parsed.get("thought", "")

      # ログ出力
      print(f"\n--- STEP {step+1} ---")
      print(f"🧠 Thought: {thought}")
      # print(f"🔎 LLM が提案した action: {action_hint}")
      # print(f"🔧 action_input: {action_input}")

      if not thought:
        continue
              
      # ルーティング（意図解析・検証）
      routed = self.route_action(parsed)
      typ = routed.get("type")
      inp = routed.get("input")

      # 実行フェーズ
      if typ == "respond":
        # エージェントが「これで完了」と判断した場合
        final_answer = inp or ""
        # print("✅ エージェントが回答を返しました（最終）")
        return final_answer

      elif typ in self.tools:
        # ツール実行（動的に辞書から呼ぶ）
        tool_fn = self.tools[typ]
        print(f" ▶ 実行: {typ} ({inp})")
        tool_result = tool_fn(inp, step, self)
        # 結果を内部履歴に蓄積して次の思考へ渡す
        thought_history += (
          f"\n<STEP {step+1}>\n"
          f"[THOUGHT] {thought}], \n"
          f"[ACTION]: {typ}, \n"
          f"[INPUT]: {inp}, \n"
          f"[RESULT]: {tool_result}\n\n"
        )
        # print(f"📥 結果: {tool_result}")

        # 次ループで LLM がこの結果を踏まえて再思考する
        continue

      else:
        # 未知アクション: フォールバックとして LLM に生テキストを返す、または終了
        reason = routed.get("reason", "unknown")
        print(f"⚠️ 未知のアクションまたは未登録のツール: {reason}")
        # 安全フォールバック：現在の LLM 出力をそのまま最終応答として返す（教育用）
        continue
        # return f"エラー: {reason}\nLLM出力: {parsed.get('raw')}"

    # 最大ステップ到達しても ANSWER/RESPOND が出ない場合
    print("⏹️ 最大ステップに到達しました。タスク完了できませんでした。")
    return "タスクを完了できませんでした（追加情報が必要です）。"


# -----------------------
# 外側の対話ループ（ユーザーとのやり取りはここで管理）
# -----------------------
if __name__ == "__main__":
  model_id = "gpt-oss:20b"
  
  system_prompt = (
    "あなたは優秀な自律型AIエージェントです。"
    "思考過程(THOUGHT)を明示し、必要があれば、"
    "ツールを使って最終的な回答を導いてください。"
  )

  # AIエージェントが使用できるツールを読み込み
  from tools.web_search import web_search
  from tools.code_calc import code_calc
  user_tools = {"web_search": web_search, "calc": code_calc}

  ollama_host = "http://127.0.0.1:11434"
  agent = Agent(model_id, system_prompt, tools=user_tools, host=ollama_host)

  print("🤖 エージェント起動 — 何を頼みますか？（'exit'で終了）")
  while True:
    goal = input("\n🎯 目的を入力: ")
    if goal.strip().lower() in ("終了", "exit", "quit"):
      print("🔚 終了します。")
      break

    print("\n🧩 エージェントが内部思考を開始します...")
    answer = agent.run(goal)
    
    print(f"\n💡 エージェントの最終回答:\n{answer}\n")
    print("="*60)
    

# 【入力例】
# こんにちは。
# 最新の日本のニュースを教えてください。
# 12345678901234567890 * 98765432109876543210 を計算してください。
# > [正解] 1219326311370217952237463801111263526900


