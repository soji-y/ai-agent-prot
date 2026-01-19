import os
import re
import json
from PIL import Image
from datetime import datetime
from collections import deque
from voice.voice import VoiceSoundAivAPI

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# from simple_llm import SimpleLLM
from simple_llm_ollama import SimpleLLMOllama


AGENT_TOOLS_PROMPT_TOP = """
You can use the following tools:
"""

AGENT_TOOLS_PROMPT_BOTTOM = """
Always output your reasoning and selected action in JSON format:
{
  "thought": "...",
  "action": "search" | "calc" | "answer",
  "action_input": "..."
}
"""

ANSWER_USING = """
answer:
 - Description: Provide the final answer to the user.
 - Input: The final response text to return to the user. 
 
"""

VOICE_URL = "http://192.168.1.100:10101"
# 1. search:
#   - Description: Search the web for up-to-date information.
#   - Input: search queries (string) (example: "today news weather")
#   - Output: A short summary of search results.
# 2. calc:
#   - Description: Perform basic mathematical calculations.
#   - Input: An arithmetic expression (e.g., "2 + 2 * 3")
#   - Output: The numeric result of the calculation.
# 3. answer:
#   - Description: Provide the final answer to the user.
#   - Input: The final response text to return to the user.
# Always output your reasoning and selected action in JSON format:
# {
#   "thought": "...",
#   "action": {"type": "search" | "calc" | "answer", "input": "..."}
# }
# """

# 最大会話履歴数
MEMORIES_NUM = 30

# 画像対応モデルか
# VISUAL_USE = True

class AdvancedAgent(SimpleLLMOllama):
  # ------------------
  # 初期化
  # ------------------
  def __init__(self, agents_cfg, def_idx=0, host=None):
    if not agents_cfg:
      return

    self.select_idx = def_idx
    
    # すべてのエージェント設定を保管
    self.agents_cfg = agents_cfg
    
    # 初期のロードのエージェント設定
    def_agent_cfg = agents_cfg[def_idx]
    
    self.load_agent_cfg(def_agent_cfg)

    super().__init__(self.model_id_path, self.sys_prompt, ollama_host=host)

    self.load_tools()
      
    # 会話履歴
    self.memories = deque(maxlen=MEMORIES_NUM)
    
    # モード
    self.mode = "nomal"
    self.game_num = 1 # 後手
  
  # エージェント設定の読み込み
  def load_agent_cfg(self, agent_cfg):
    
    self.name = agent_cfg["name"]
    self.model_id_path = agent_cfg["model_id_path"]
    self.sys_prompt = agent_cfg["sys_prompt"]

    # 以下はなくても良い
    self.max_new_tokens = agent_cfg["max_new_tokens"] if "max_new_tokens" in agent_cfg else None# 生成するトークンの最大数
    self.temperature = agent_cfg["temperature"] if "temperature" in agent_cfg else None# 確率分布の形を調節するパラメータ
    
    cfg_video = agent_cfg["video"] if "video" in agent_cfg else None
    if cfg_video:
      self.video_paths = cfg_video["nomal"]
      self.video_paths_speak = cfg_video["speak"]
      self.video_reverse = cfg_video["reverse"]
      self.video_fade_msec = cfg_video["fade_msec"]
      
    cfg_voice  = agent_cfg["voice"] if "voice" in agent_cfg else None
    if cfg_voice:
      style_id = cfg_voice["style_id"]
      self.voice = VoiceSoundAivAPI(url=VOICE_URL, style_id=style_id, async_flg=True)

    self.tool_list = agent_cfg["tools"] if "tools" in agent_cfg else []

  # ツールの読み込み
  def load_tools(self):
    self.tools = {}
    self.tool_names = []
    self.alias_map = {}
    for tool in self.tool_list:
      if tool == "search": # Web検索
        from tools.web_search import WebSearch
        self.tools["search"] = WebSearch()
      elif tool == "calc": # 計算
        from tools.code_calc import CodeCalc
        self.tools["calc"] = CodeCalc()
      elif tool == "operate": # ファイル操作
        from tools.file_operation import FileOperateion
        self.tools["operate"] = FileOperateion()
      elif tool == "othello": # オセロ
        from tools.othello_play import Othello
        self.tools["othello"] = Othello()
        
    # システムプロンプトにツールの説明を追加
    self.system_prompt += AGENT_TOOLS_PROMPT_TOP

    self.tool_names = []
    # 別名
    self.alias_map = {}
    n = 1
    for tool in self.tools.values():
      self.tool_names.append(tool.name())
      self.system_prompt += f"{n}. {tool.using().strip()}\n"
      for k, v in tool.alias().items():
        self.alias_map[k] = v
      n += 1
    
    # 最終回答  
    self.system_prompt +=  f"{n}. {ANSWER_USING.strip()}\n"
    
    # システムプロンプトにツールの説明を追加
    self.system_prompt += AGENT_TOOLS_PROMPT_BOTTOM

    self.tool_names.append("answer")
    self.alias_map["respond"] = "answer"  
    self.alias_map["answer"] = "answer"    

  # エージェントの変更
  def change(self, idx):
    self.select_idx = idx

    # 初期のロードのエージェント設定
    agent_cfg = self.agents_cfg[idx]
    
    # ひとつ前のモデル
    bef_model_id_path = self.model_id_path
    self.load_agent_cfg(agent_cfg)
    
    model_change = bef_model_id_path != self.model_id_path
    self.change_base(self.model_id_path, self.sys_prompt, model_change=model_change)

    self.load_tools()
      
    # 会話履歴
    self.memories.clear() # = deque(maxlen=MEMORIES_NUM)
  
  
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
  # 安全なパース
  # ------------------
  def _safe_parse_json(self, json_text: str):
    """
    文章中のシングルクオートはそのままにして、
    JSONのキーや構造上のシングルクオートだけをダブルクオートに変換し、
    末尾カンマやスマートクオートも修正して安全にjson.loads()する。
    """
    # 末尾カンマの除去（オブジェクト・配列両対応）
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)

    # スマートクオートの置換
    json_text = json_text.replace("“", '"').replace("”", '"')

    # BOMや先頭/末尾の空白除去
    json_text = json_text.strip("\ufeff\n\r\t ")

    # JSONキーや構造上のシングルクオートをダブルクオートに変換
    # 文字列中のシングルクオートにはマッチしないように簡易正規表現
    # { 'key': ... } → { "key": ... }
    json_text = re.sub(
        r'(?<=\{|,)\s*\'([^\']+)\'\s*:',  # { 'key': か , 'key': のパターン
        r'"\1":',
        json_text
    )

    # JSONとしてロード
    return json.loads(json_text)
  
  # ------------------
  # LLM の返した自由テキストを JSON として解釈
  # ------------------  
  def parse_response(self, text):
    """
    LLM の返した自由テキストを JSON として解釈し、
    {'thought':..., 'action':..., 'action_input':...} を返す。
    フォールバック: JSON にできなければ最終的に 'action': 'answer' で生テキストを返す。
    """
    # ① まず JSON 部分を抽出してパース
    json_text = self._extract_json_from_text(text)
    if json_text:
      try:
        # json_text = re.sub(r',\s*}', '}', json_text)
        # json_text = json_text.replace("'", '"').replace("“", '"').replace("”", '"').strip("\ufeff\n\r\t ")
        # obj = json.loads(json_text)
        obj = self._safe_parse_json(json_text)
        # 正常にパースできたら必要キーを取り出す（存在しないキーは空文字で埋める）
        dic = {
          "thought": obj.get("thought", "").strip(),
          "action": obj.get("action", "").strip(),
          "action_input": obj.get("action_input", "").strip(),
          "raw": text
        }
        return dic
      except json.JSONDecodeError as e:
        print(f"Json parse Error: {e}")
      except Exception as e:
        # JSON 部分が不正な場合はフォールバック
        print(f"Error: {e}")

    # ② フォールバック：JSONが取れなければ、LLMの出力全体を action_input とする
    return {
      "thought": "",
      "action": "answer",            # 直接応答を期待する形にフォールバック
      "action_input": text.strip(),
      "raw": text
    }

  # ------------------
  # ルーター（意図解析 + 検証） -- **明示的に分離**
  # ------------------
  def route_action(self, parsed):
    # """
    # parsed = {'thought':..., 'action':..., 'action_input':..., 'raw':...}
    # を受け取り、実行すべきアクション種別とパラメータを返す。

    # 戻り値例:
    #   {"type": "web_search", "input": "〜"}
    #   {"type": "calc", "input": "2+2"}
    #   {"type": "respond", "input": "最終回答"}
    #   {"type": "unknown", "input": "...", "reason": "..."}
    # """
    
    action = (parsed.get("action") or "").lower().strip()
    action_input = parsed.get("action_input", "")

    # 別名対策
    # alias_map = {
    #   "search": "web_search",
    #   "web": "web_search",
    #   "calculate": "calc",
    #   "answer": "respond",
    #   "final": "respond"
    # }
    action = self.alias_map.get(action, action)

    # 未指定や空文字は不明扱い
    if not action:
      return {"type": "unknown", "input": action_input, "reason": "action が空です。"}

    # respond（最終回答）はツールを呼ばずそのまま返す
    if action in ("respond", "answer"):
      return {"type": "answer", "input": action_input}

    # ツールの存在を確認してから実行する（辞書ベース）
    if action in self.tools:
      return {"type": action, "input": action_input}

    # ツール未登録の場合は unknown を返す（呼び出し元でフォールバック可能）
    return {"type": "unknown", "input": action_input, "reason": f"未登録のアクション: {action}"}

  # モードチェンジ
  def change_mode(self, result, user_name="ユーザー"):
    # result = "<mode:nomal>test"
    match = re.search(r"<mode:(.*?)>", result)
    if match:
      mode_value = match.group(1)
      if mode_value:
        mode_value = mode_value.strip()
        self.mode = mode_value
        result += f"<{self.mode}>に変更しました。\"anser\"から{user_name}に報告してください。"
        print("モード変更:", mode_value)
    else:
      # print("見つかりませんでした")
      pass
    
    return result
  
  # メッセージにリアクションを返すかどうか(音声入力のとき)
  def reaction_check(self, message, user_name):
    prompt = (
      f"以下の{user_name}からのメッセージに対して、回答が必要かどうかを判断してください。"
      "以下のいずれか(Yes or No)を選択してください。\n"
      f"<Yes>: {user_name}からのメッセージがあなたへの「質問」や「指示」、または「応答が必要な内容」の場合。\n"
      f"<No>: {user_name}からのメッセージが「独り言」や「雑音」など、あなたへの話しかけでない場合。\n\n"
      f"メッセージ({user_name}): {message}"
    )
    raw = self.generate(prompt, memories = self.memories, sys_use=False)
    
    if "Yes" in raw:
      print(f"音声入力判定: Yes")
      return True
    else:
      print(f"音声入力判定: No")
      return False
    
  # ------------------
  # エージェント本体（ユーザーの「目的」を受け、最大10ターンで完了を目指す）
  # ------------------
  def run(self, message, max_steps=10, user_name="ユーザー", image_path=None, queue_thought=None, speech_flg=False): #image:Image=None):
    """
    user_message: ユーザーが与えた目的（文字列）
    戻り値: 最終的な回答文字列（あるいは失敗メッセージ）
    """
    set_action = None
    game_Win_flg = False
    if message.startswith("<game>"):
      game_Win_flg = True
      # message = message[len("<game>"):]
      
    if self.mode == "othello":
      result = self.tools["othello"](message, 0)
      # オセロの盤面を文字列で取得
      othello_text = self.tools["othello"].to_text()

      if "<True>" in result:
        turn_str = "黒" if self.game_num == 0 else "白"
        message = f"{result.replace("<True>","").strip()}\n次はあなたの手番({turn_str})です。\"action\": \"othello\"から合法手を指してください。\n{othello_text}"
        # オセロの盤面を文字列で取得
        print(self.tools["othello"].to_text())
        print("合法手: " + ", ".join(self.tools["othello"].get_legal_str_moves()))
          
        set_action = "othello"
      elif "<False>" in result:
        turn_str = "白" if self.game_num == 0 else "黒"
        message = f"{user_name}の手番({turn_str})です。\n{user_name}「{message}」\n{othello_text}\n{result}"

    if not game_Win_flg:
      # 音声入力の場合は、回答が必要かチェック
      if speech_flg:
        if not self.reaction_check(message, user_name):
          return None, None
      
    # print(f"🎯 エージェント目標: {user_message}\n")
    after_action = None # レス後のアクション
    thought_history = ""  # 思考・行動・結果の履歴（エージェント内部メモリ／短期）
    # tool_result = "" # ツール結果
    for step in range(max_steps):
      # LLM に投げるプロンプト（JSON 形式で出力するよう厳格に指示）
      prompt = (
        # f"{self.system_prompt}\n\n"
        f"{user_name}からのメッセージに含まれる、目的達成のために次の形式で JSON を出力してください。\n"
        # "目的達成のために次の形式で JSON を出力してください。\n"
        "```\n"
        "{\n"
        f"  \"thought\": \"今考えていること（{user_name}に説明する言葉）\",\n"
        # "  \"action\": \"search / calc / respond \",\n"
        f"  \"action\": \"{ " / ".join(self.tool_names) if not set_action else set_action}\",\n"
        "  \"action_input\": \"そのアクションに渡す入力、または回答(answerの場合は文字列型の文章)\"\n"
        "}\n"
        "```\n\n"
        "上の形式だけを JSON で返してください（余計な説明は出力しないでください）。"
        # f"メッセージが単純な会話や、これまでの内部履歴がある場合は、\"action\":\"answer\"とし、\"action_input\"に{user_name}への回答を記述してください。\n"
        f"{user_name}からのメッセージ(目標): {message}\n"
        f"入力画像: {os.path.basename(image_path) if image_path else 'なし'}\n"
        f"現在の日時: {datetime.now().strftime("%Y-%m-%d (%A) %H:%M:%S")}\n"
        f"これまでの内部履歴:\n{thought_history}\n"
        # f"以前の会話履歴:\n{self.memories}"
        # "/no_think"
      )

      raw = self.generate(prompt, memories = self.memories, image=image_path)
      parsed = self.parse_response(raw)   # JSON 抽出 & パース
      # ここで thought/action/action_input を取り出す
      thought = parsed.get("thought", "")
      # action_hint = parsed.get("action", "")
      # action_input = parsed.get("action_input", "")
      if queue_thought and thought:
        queue_thought.put(thought)
        
      # ログ出力（勉強会用に可視化）
      print(f"\n--- STEP {step+1} ---")
      print(f"🧠 Thought: {thought}")
      # print(f"🔎 LLM が提案した action: {action_hint}")
      # print(f"🔧 action_input: {action_input}")

      user_message = f"{user_name}: {message}"
      # ルーティング（意図解析・検証）
      routed = self.route_action(parsed)
      typ = routed.get("type")
      inp = routed.get("input")

      # 実行フェーズ
      if typ == "answer":
        # エージェントが「これで完了」と判断した場合
        final_answer = inp or ""

        if self.mode == "othello":
          print(self.tools["othello"].to_text())
          print("合法手: " + ", ".join(self.tools["othello"].get_legal_str_moves()))        
        # if isinstance(final_answer, dict):
        
        # 内部メモリに保存
        self.append_memory(user_message, "user")
        self.append_memory(final_answer, "assistant")

        # print("✅ エージェントが回答を返しました（最終）")
        # if self.voice:
        #   self.voice.create_voice(final_answer)
          
        return final_answer, after_action

      elif typ in self.tools:
        # ツール実行（動的に辞書から呼ぶ）
        tool_fn = self.tools[typ]
        print(f" ▶ 実行: {typ} ({inp})")
        
        if game_Win_flg:
          # ゲームの時は手番を渡す
          num = self.game_num
        else:
          num = step          
        
        tool_result = tool_fn(inp, num, self)
        
        if self.mode == "othello":
          # if "Othello-Start" in inp:
          #   after_action = "<open:othello>"
          # elif "Othello-End" in inp:
          #   after_action = "<close:othello>"                      
          if tool_result.startswith("<True>"):
          
            tool_result = tool_result[len("<True>"):].strip()
            tool_result += f"{user_name}に結果を報告してください。"
            set_action = "answer"
            # オセロの盤面を文字列で取得
            print(self.tools["othello"].to_text())
            print("合法手: " + ", ".join(self.tools["othello"].get_legal_str_moves()))

            if game_Win_flg:
              return None, f"<{self.mode}>"
            
          elif "<False>" in tool_result:
            tool_result = tool_result.replace("<False>", "").strip()
            tool_result += f"オセロの手を打つのに失敗しました。別の手を考えてください。"
        
        # モード変更
        tool_result = self.change_mode(tool_result, user_name)
        
        if self.mode == "othello":
          if "Othello-Start" in inp:
            after_action = "<open:othello>"
          elif "Othello-End" in inp:
            after_action = "<close:othello>"            
          
          
        # 結果を内部履歴に蓄積して次の思考へ渡す
        thought_history += (
          f"\n<STEP {step+1}>\n"
          f"[THOUGHT] {thought}, \n"
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
    return "タスクを完了できませんでした。", None

  # メモリーに追加
  def append_memory(self, message, role="user"):
    self.memories.append({"role": role, "content": message})
    
# -----------------------
# 外側の対話ループ（ユーザーとのやり取りはここで管理）
# -----------------------
if __name__ == "__main__":
  # model_path = "google/gemma-3-4b-it"  # 適宜置き換え
  # model_id = "Qwen/Qwen3-4B-SafeRL"
  # model_id = "Qwen/Qwen3-4B-Instruct-2507-FP8"
  model_id = "Qwen/Qwen3-4B-Instruct-2507"

  system_prompt = (
    "あなたは優秀な自律型AIエージェントです。"
    "思考過程(THOUGHT)を明示し、必要があれば、"
    "ツールを使って最終的な回答を導いてください。"
  )

  # ここで実際の検索関数を注入可能（社内の web_search 関数を渡す想定）
  # from tools.web_search import web_search
  # from tools.code_calc import code_calc
  
  # from tools.web_search import WebSearch
  
  # web = WebSearch()
  
  # web_name = web.name()
  
  # user_tools = {}
  # user_tools["search"] = WebSearch() #web_search
  # # user_tools["calc"] = code_calc
  # agent = AdvancedAgent(model_id, system_prompt, tools=user_tools)

  # 設定されたエージェント(yaml)を読み込み
  agent_cfg = {
    "name": "執事",
    "model_id_path": "Qwen/Qwen3-4B-Instruct-2507",
    "sys_prompt": (
    "あなたは「セバスチャン」という名前の 優秀な執事です。マスターの指示には誠実に答えます。"
    "マスターはあなたのことを「爺」「じい」「ジィ」「執事」などと呼びます。"
    "あなたは高齢の男性執事のため、語尾には「ですじゃ。」「しますぞ。」などの言葉遣いで回答します。"
    ),
    # "max_new_tokens": 768,
    # "temperature": 0.7,
    # "video":{
    #   "nomal": ["./data/Shitsuji_001_nomal_01_27.mp4", "./data/Shitsuji_001_nomal_02_27.mp4"],
    #   "speak": ["./data/Shitsuji_001_speak_01_27.mp4"],
    #   "reverse": True
    # },
    # "voice": {"style_id": 391794336}, # ろてじん（長老ボイス）
    "tools": ["search", "calc"]
  }
  
  agent = AdvancedAgent([agent_cfg])

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
# 最新のニュースを教えてください。
# 12345678901234567890 * 98765432109876543210 を計算してください。
