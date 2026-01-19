import os
import gc
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3VLMoeForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

os.environ["TRITON_DISABLE"] = "1"  # Triton を無効化
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

DEF_MAX_NEW_TOKENS = 768
DEF_TEMPERATURE = 0.7

# モデルを完全開放
def free_transformers_objects(model=None, tokenizer=None, processor=None, other_objs: list = None, verbose=True):
  """
  - model: nn.Module (Transformers model)
  - tokenizer / processor: HF tokenizer/processor objects
  - other_objs: その他削除したいオブジェクトのリスト
  """
  if other_objs is None:
    other_objs = []

  # 1) 可能なら model を CPU に移動（GPU上のパラメータを先に移す）
  try:
    if model is not None:
      # model.half() などよりも確実に GPU -> CPU
      model.to('cpu')
      if verbose:
        print("model moved to CPU")
  except Exception as e:
    if verbose:
      print("model.to('cpu') failed:", e)

  # 2) （オプション）PEFT の場合は専用APIでアンロード（存在すれば）
  try:
    if model is not None and hasattr(model, "merge_and_unload"):
      # LoRA 等をベースにマージしてアダプタをアンロードする（あれば便利）
      model.merge_and_unload()
      if verbose:
        print("Called model.merge_and_unload()")
  except Exception as e:
    if verbose:
      print("merge_and_unload failed or not applicable:", e)

  # 3) 参照を削除
  names = []
  for name, obj in (("model", model), ("tokenizer", tokenizer), ("processor", processor)):
    try:
      if obj is not None:
        del obj
        names.append(name)
    except Exception as e:
      if verbose:
        print(f"del {name} failed:", e)

  for i, o in enumerate(other_objs):
    try:
      del other_objs[i]
    except Exception:
      pass

  # 4) 明示的にガベージコレクション
  gc.collect()
  if verbose:
    print("gc.collect() done")

  # 5) CUDA メモリ開放（キャッシュを空にする）
  if torch.cuda.is_available():
    try:
      torch.cuda.empty_cache()
      # 同期しておくと状況確認が安定する
      torch.cuda.synchronize()
      if verbose:
        print("torch.cuda.empty_cache() + synchronize done")
    except Exception as e:
      if verbose:
        print("CUDA empty_cache/sync failed:", e)

  # 6) 確認用にメモリ使用量を表示（任意）
  try:
    if torch.cuda.is_available():
      idx = torch.cuda.current_device()
      reserved = torch.cuda.memory_reserved(idx)
      allocated = torch.cuda.memory_allocated(idx)
      if verbose:
        print(f"CUDA device {idx} reserved={reserved:,} allocated={allocated:,}")
  except Exception:
    pass
  
# シンプルなローカルLLM
class SimpleLLM:
  # ------------------
  # 初期化
  # ------------------
  def __init__(self, model_id_path, system_prompt):
    self.load_model(model_id_path)

    self.system_prompt = system_prompt # システムプロンプトを設定
    
    if hasattr(self, "max_new_tokens") and self.max_new_tokens:
      self.max_new_tokens = self.max_new_tokens
    else:
      self.max_new_tokens = DEF_MAX_NEW_TOKENS
    
    if hasattr(self, "temperature") and self.temperature:
      self.temperature = self.temperature
    else:
      self.temperature = DEF_TEMPERATURE
      
    # self.max_new_tokens = max_new_tokens # 生成するトークンの最大数
    # self.temperature = temperature # 確率分布の形を調節するパラメータ
  
  # モデルをロード
  def load_model(self, model_id_path):
    print(f"🔧 モデルをロード中... [{model_id_path}]")

    self.model_id_path = model_id_path
    # トークナイザーをロード
    self.tokenizer = AutoTokenizer.from_pretrained(model_id_path)

    if "Qwen3-VL" in model_id_path:
      CausalLM = Qwen3VLMoeForConditionalGeneration    
    elif "Qwen2.5-VL" in model_id_path:
      CausalLM = Qwen2_5_VLForConditionalGeneration
    else:
      CausalLM = AutoModelForCausalLM
    
    # モデルをロード（型とデバイスは自動設定、必要ならload_in_8bitも検討）
    self.model = CausalLM.from_pretrained(
        model_id_path,
        dtype= torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        # load_in_8bit=True      # メモリ節約
    )

    # 画像入力に対応しているか
    self.vision_model = self.is_vision_model()
    
    if self.vision_model:
      self.processor = AutoProcessor.from_pretrained(model_id_path)
    else:
      self.processor = None
      
  # エージェントの設定を変更
  def change_base(self, model_id_path, system_prompt, model_change=False):

    self.system_prompt = system_prompt # システムプロンプトを設定

    if model_change:
      print(f"🔧 モデルを解放中... [{model_id_path}]")
      # モデル読み込み・推論 → 終わったら
      free_transformers_objects(model=self.model, tokenizer=self.tokenizer, processor=self.processor)
      
      self.load_model(model_id_path)
      self.vision_model = self.is_vision_model()
      
    # self.max_new_tokens = max_new_tokens # 生成するトークンの最大数
    # self.temperature = temperature # 確率分布の形を調節するパラメータ
  
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
        # if i == len(memories) - 1:
        #   break
        messages.append(mem)
    
    if self.vision_model and image:
      prompt_st = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
              ]
    else:
      prompt_st = prompt
        
    messages.append({"role": "user", "content": prompt_st})
    
    # messages = [
    #   *([{"role": "system", "content": self.system_prompt}] if sys_use else []),
    #   {"role": "user", "content": prompt}
    # ]
    
    if self.processor:
      # Preparation for inference
      text = self.processor.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
      )
      image_inputs, video_inputs = process_vision_info(messages)
      model_inputs = self.processor(
          text=[text],
          images=image_inputs,
          videos=video_inputs,
          padding=True,
          return_tensors="pt",
      ).to(self.model.device)
      
      # # トークン生成
      # generated_ids = self.model.generate(
      #   **model_inputs, 
      #   max_new_tokens=128
      #   )
      
      # generated_ids_trimmed = [
      #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
      # ]
      # output_text = self.processor.batch_decode(
      #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
      # )[0]
      # print(output_text)

    else:
      # チャットテンプレートを使用
      text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
      )
      
      # モデル入力を作成
      model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
      
    # トークン生成
    generated_ids = self.model.generate(
      **model_inputs,
      max_new_tokens=self.max_new_tokens,
      temperature = self.temperature
    )

    # 入力部分をスライスして出力のみ取得
    output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:].tolist()
      
    # トークン列を文字列に戻す
    output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
      
    return output_text

  # 画像入力に対応しているか
  def is_vision_model(self):
    if not self.model:
      return False

    cfg = getattr(self.model, "config", None)
    if not cfg:
      return False
    return any(hasattr(cfg, key) for key in [
      "vision_config", "mm_vision_tower", "visual_config", "image_token_index"
    ])
    
# -----------------------
# 起動: 外側の対話ループ（ユーザーとのやり取りはここで管理）
# -----------------------
if __name__ == "__main__":
  #model_id = "Qwen/Qwen3-4B-SafeRL"
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
# 最新のニュースを教えてください。
# 12345678901234567890 * 98765432109876543210 を計算してください。
# > [正解] 1219326311370217952237463801111263526900
# ある商品が定価3,500円から20%引きで販売されています。さらに、会員カードを使うと、割引後の価格から10%が追加で割引されます。この商品を購入するために最終的に支払う金額は「何円」ですか？