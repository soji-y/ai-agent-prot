import ollama


# シンプルなローカルLLM(Ollama)
class SimpleLLMOllama:

  # ------------------
  # 初期化
  # ------------------
  def __init__(self, model_id, system_prompt, ollama_host='http://127.0.0.1:11434'):
    self.model_id = model_id
    self.system_prompt = system_prompt # システムプロンプトを設定
    self.ollama_client = ollama.Client(host=ollama_host)

  # ------------------
  # LLMで回答を生成
  # ------------------
  def generate(self, prompt, sys_use=True):

    # 送信メッセージを組み立て
    messages = []
    if sys_use:
      messages.append({"role": "system", "content": self.system_prompt})
    messages.append({"role": "user", "content": prompt})

    # 回答を生成
    output_text = self.ollama_client.chat(model=self.model_id, messages=messages)
    
    # 回答の文字列を取り出し
    content = output_text.get("message", {}).get("content", "")

    return content


# -----------------------
# 起動: 外側の対話ループ（ユーザーとのやり取りはここで管理）
# -----------------------
if __name__ == "__main__":
  model_id = "gemma3:4b"

  system_prompt = (
    "あなたは優秀なアシスタントです。"
    "ユーザーの質問に対する回答を返してください。"
  )

  llm = SimpleLLMOllama(model_id, system_prompt)

  print("🤖 LLM起動 — 何を頼みますか？（'exit'で終了）")
  while True:
    goal = input("\n🎯 目的を入力: ")
    if goal.strip().lower() in ("終了", "exit", "quit"):
      print("🔚 終了します。")
      break

    print("\n🧩 LLMが回答を生成しています...")
    answer = llm.generate(goal)
    
    print(f"\n💡 LLMの最終回答:\n{answer}\n")
    print("="*60)
    
# 【入力例】
# こんにちは。
# 最新の日本のニュースを教えてください。
# 12345678901234567890  98765432109876543210 を計算してください。
# > [正解] 1219326311370217952237463801111263526900
