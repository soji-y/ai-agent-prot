import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# シンプルなローカルLLM
class SimpleLLM:

  # ------------------
  # 初期化
  # ------------------
  def __init__(self, model_id_path, system_prompt):
    print("🔧 モデルをロード中...")
    
    # トークナイザーをロード
    self.tokenizer = AutoTokenizer.from_pretrained(model_id_path)

    # モデルをロード（型とデバイスは自動設定、必要ならload_in_8bitも検討）
    self.model = AutoModelForCausalLM.from_pretrained(
        model_id_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0",
    )

    # システムプロンプトを設定
    self.system_prompt = system_prompt

    
  # ------------------
  # LLMで回答を生成
  # ------------------
  def generate(self, prompt, sys_use=True):

    # 送信メッセージを組み立て
    messages = []
    if sys_use:
      messages.append({"role": "system", "content": self.system_prompt})
    messages.append({"role": "user", "content": prompt})

    
    # チャットテンプレートを使用した文字列に変換
    temp_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    
    # モデル入力用トークンIDに変換
    input_ids = self.tokenizer([temp_text], return_tensors="pt").to(self.model.device)
    
    # トークン生成
    generated_ids = self.model.generate(
      **input_ids,
      max_new_tokens=512, # 生成するトークンの最大数
      temperature=0.7, # 確率分布の形を調節するパラメータ
    )

    # 入力部分をスライスして出力のみ取得
    output_ids = generated_ids[0][input_ids.input_ids.shape[1]:].tolist()
      
    # トークンIDを文字列に戻す
    content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return content


# -----------------------
# 起動: 外側の対話ループ（ユーザーとのやり取りはここで管理）
# -----------------------
if __name__ == "__main__":
  model_id = "Qwen/Qwen3-4B-Instruct-2507"
  system_prompt = (
    "あなたは優秀なアシスタントです。"
    "ユーザーの質問に対する回答を返してください。"
  )

  llm = SimpleLLM(model_id, system_prompt)

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
# 12345678901234567890 * 98765432109876543210 を計算してください。
# > [正解] 1219326311370217952237463801111263526900
