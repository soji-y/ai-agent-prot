import ollama

# ollama_client = ollama.Client(host='http://192.168.1.100:11434')

DEF_MAX_NEW_TOKENS = 768
DEF_TEMPERATURE = 0.7


# シンプルなローカルLLM(Ollama)
class SimpleLLMOllama:
  # ------------------
  # 初期化
  # ------------------
  def __init__(self, model_id, system_prompt, ollama_host='http://127.0.0.1:11434'):
    self.model_id = model_id
    self.system_prompt = system_prompt # システムプロンプトを設定
    # 画像入力に対応しているか
    self.vision_model = self.is_vision_model()
    self.ollama_client = ollama.Client(host=ollama_host)

  # エージェントの設定を変更
  def change_base(self, model_id, system_prompt, model_change=False):
    self.system_prompt = system_prompt # システムプロンプトを設定
    if model_change:
      self.model_id = model_id
      self.vision_model = self.is_vision_model()
      
    
  # ------------------
  # LLMで回答を生成
  # ------------------
  def generate(self, prompt, sys_use=True, memories=None, image=None):

    # 送信メッセージを組み立て
    messages = []
    if sys_use:
      messages.append({"role": "system", "content": self.system_prompt})

    if memories and len(memories) > 1:
      for i, mem in enumerate(memories):
        messages.append(mem)

    prompt_st = {"role": "user", "content": prompt}
    
    if self.vision_model and image:
      prompt_st["images"] = [image]

    messages.append(prompt_st)

    # if self.ollama_client:
    output_text = self.ollama_client.chat(model=self.model_id, messages=messages)
    # else:
    #   output_text = ollama.chat(model=self.model_id, messages=messages)
          
    return output_text.get("message", {}).get("content", "")

  # ------------------
  # 画像入力に対応しているか
  # ------------------
  def is_vision_model(self) -> bool:
    # 1. 文字列ベースのキーワード判定（単純だが有効なケースを捕らえる）
    low = self.model_id.lower()
    for kw in ("vision", "image", "multimodal", "vl", "gemma3", "llava"):
        if kw in low:
            # “gemma3” を含むモデルで 4b,12b,27b は vision 対応という情報もある
            if kw == "gemma3":
                # ただし 1b バージョンは vision 非対応と言われている例があるので除外
                if ":1b" in low or low.endswith("1b"):
                    continue
            return True

    # 2. モデルメタデータを見て判断
    try:
        info = ollama.show(model_id)
    except Exception as e:
        return False

    # modelfile / parameters 内にキーワードが含まれるか
    for field in ("modelfile", "parameters", "description", "tags"):
        val = info.get(field)
        if isinstance(val, str) and any(kw in val.lower() for kw in ("vision", "image", "multimodal", "vl")):
            return True

    
# -----------------------
# 起動: 外側の対話ループ（ユーザーとのやり取りはここで管理）
# -----------------------
if __name__ == "__main__":
  model_id = "gemma3:4b"
  system_prompt = (
    "あなたは優秀なアシスタントです。"
    "ユーザーの質問に対する回答を返してください。"
  )
  ollama_host = 'http://192.168.1.100:11434'

  llm = SimpleLLMOllama(model_id, system_prompt, ollama_host=ollama_host)
  print(llm.is_vision_model())

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
# 最新のニュースを教えてください。
# 12345678901234567890 * 98765432109876543210 を計算してください。
# > [正解] 1219326311370217952237463801111263526900
# ある商品が定価3,500円から20%引きで販売されています。さらに、会員カードを使うと、割引後の価格から10%が追加で割引されます。この商品を購入するために最終的に支払う金額は「何円」ですか？