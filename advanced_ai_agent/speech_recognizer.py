import sys
import torch
import sounddevice as sd
import numpy as np
import queue
import threading
import soundfile as sf
import librosa
from PyQt5.QtWidgets import (
  QApplication, QMainWindow, QTextEdit, QAction, QMenu
)
# pyqtSlot, pyqtSignal, QObject を追加
from PyQt5.QtCore import QMetaObject, Q_ARG, Qt, pyqtSlot, QObject, pyqtSignal

# faster-whisper を使用
from faster_whisper import WhisperModel 
# LLM モデル情報とライブラリ
from transformers import AutoModelForCausalLM, AutoTokenizer 
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio, save_audio, VADIterator, collect_chunks

# — 音声取得 + VAD の強化部 — #
class RealTimeVADRecorder:
  def __init__(self, sample_rate=16000, frame_duration_ms=100, padding_ms=500):
    import queue
    import os
    import urllib.request
    import torch

    self.sample_rate = sample_rate
    self.frame_duration_ms = frame_duration_ms
    self.frame_size = int(sample_rate * frame_duration_ms / 1000)
    self.padding_frames = int(padding_ms / frame_duration_ms)

    # 終了フレーム数
    self.stop_frame = 30

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ VAD running on device: {self.device}")

    # モデルをロード (JIT モード)
    self.model = load_silero_vad(onnx=False)  # 必要なら onnx=True を指定
    self.model.to(self.device).eval()

    # # --- ✅ torch.hub.load を避けて手動ロード ---
    # model_url = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.jit"
    # model_path = "./silero_vad.jit"

    # if not os.path.isfile(model_path):
    #   print("Downloading Silero VAD model...")
    #   urllib.request.urlretrieve(model_url, model_path)

    # self.model = torch.jit.load(model_path, map_location=self.device)
    # self.model.eval()

    # Silero公式から必要ユーティリティを取得
    from silero_vad import get_speech_timestamps, collect_chunks, save_audio, read_audio, VADIterator
    self.get_speech_timestamps = get_speech_timestamps
    self.collect_chunks = collect_chunks
    self.save_audio = save_audio
    self.read_audio = read_audio
    self.VADIterator = VADIterator

    self.audio_queue = queue.Queue()
    self.stream = None
    self.running = False
  
  def audio_callback(self, indata, frames, time_info, status):
      if status:
          print("Audio status:", status)
      mono = indata.mean(axis=1)
      self.audio_queue.put(mono.copy())

  def start(self):
      self.stream = sd.InputStream(
          samplerate=self.sample_rate,
          channels=1,
          callback=self.audio_callback,
          blocksize=self.frame_size,
      )
      self.stream.start()
      self.running = True

  def stop(self):
      self.running = False
      if self.stream and self.stream.active:
          try:
              self.stream.stop()
          except Exception as e:
              print(f"Warning: Stream stop failed: {e}")
          try:
              self.stream.close()
          except Exception as e:
              print(f"Warning: Stream close failed: {e}")
      # キューをクリア
      while not self.audio_queue.empty():
          try:
              self.audio_queue.get_nowait()
          except queue.Empty:
              break

  def read_chunk(self, timeout=None):
      try:
          return self.audio_queue.get(timeout=timeout)
      except queue.Empty:
          return None

  def detect_speech_segment(self):


    ring = []
    voiced = []
    triggered = False
    num_unvoiced = 0
    chunk_buffer = np.zeros(0, dtype=np.float32)

    while True:
      frame = self.read_chunk(timeout=1.0)
      if frame is None:
        # 入力が無い場合、ある程度待っても音声が来なければ終了
        silence_counter += 1
        if silence_counter > 50:  # 約50フレーム分何も入力が無ければ終了
          break
        continue
      silence_counter = 0

      # フレーム結合
      chunk_buffer = np.concatenate([chunk_buffer, frame])

      while len(chunk_buffer) >= 512:
        current_chunk = chunk_buffer[:512]
        chunk_buffer = chunk_buffer[512:]

        audio_tensor = torch.from_numpy(current_chunk).unsqueeze(0).to(self.device)

        # 無音チェック
        if torch.mean(torch.abs(audio_tensor)) < 1e-4:
          is_speech = False
        else:
          try:
            with torch.no_grad():
              is_speech_prob = self.model(audio_tensor, self.sample_rate).item()
            is_speech = is_speech_prob > 0.5
          except ValueError as e:
            if "too short" in str(e):
              continue
            else:
              raise

        # 発話の開始／終了検出
        ring.append(current_chunk)
        if len(ring) > self.padding_frames:
          ring.pop(0)

        if not triggered:
          if is_speech:
            triggered = True
            voiced.extend(ring)
            ring.clear()
        else:
          voiced.append(current_chunk)
          if not is_speech:
            num_unvoiced += 1
            if num_unvoiced >= self.stop_frame:
              break  # 🔸発話終了検出
          else:
            num_unvoiced = 0

      # 🔸発話終了検出でbreakしたら外側も抜ける
      if triggered and num_unvoiced >= self.stop_frame:
        break

    # 🔸最後に音声が溜まっていれば返す
    if voiced:
      audio = np.concatenate(voiced, axis=0)
      return audio
    else:
      return None
    
# — シグナル定義クラス — #
class LlmSignals(QObject):
  """
  LLM/ASR処理スレッドからMainWindowへ通知するためのシグナルを定義
  """
  llm_loaded = pyqtSignal()
  llm_error = pyqtSignal(str) 
  recognized = pyqtSignal(str)
  llm_responded = pyqtSignal(str)

# — 音声認識部 — #
class SpeechRecognizer:
  """
  sounddevice でリアルタイムに音声を取得し、バッファリング後、
  faster-whisper で認識し、結果をシグナルでGUIスレッドに渡す。
  """
  def __init__(self, signals: LlmSignals, model_name="large-v3", sample_rate=16000, device=None):
    self.signals = signals 
    self.sample_rate = sample_rate
    self.stream = None
    
    if device is None:
      device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading faster-whisper model: {model_name} on {device}...")
    self.model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
    
    self.q = queue.Queue()
    self.running = False
    self.silence_limit = 1.5
    self.frame_duration = 0.3 # 0.1

    # ✅ 修正: VADRecorderを初期化
    self.vad_recorder = RealTimeVADRecorder(sample_rate=self.sample_rate)

  def audio_callback(self, indata, frames, time, status):
    """sounddeviceの入力ストリームから呼び出される"""
    if status:
      print("Audio status:", status, file=sys.stderr)
    if not self.running:
      return
            
    if indata.shape[1] > 1:
      indata = indata.mean(axis=1)
    
    self.q.put(indata.copy().astype(np.float32))

  def start(self):
    """音声入力を開始し、処理スレッドを起動"""
    if self.running:
      return
            
    self.running = True

    # ✅ 修正: VADRecorderのストリームを開始
    self.vad_recorder.start() 
    
    # ✅ 修正: VADRecorderから音声を取得する _run スレッドを起動
    threading.Thread(target=self._run, daemon=True).start()
    # threading.Thread(target=self._run).start() # ✅ 修正後 (デーモンを外す)
    
    blocksize = int(self.sample_rate * self.frame_duration)
    self.stream = sd.InputStream(
      channels=1, 
      samplerate=self.sample_rate, 
      callback=self.audio_callback,
      blocksize=blocksize
    )
    self.stream.start()

  def stop(self):
    """音声入力を停止"""
    self.running = False
    if self.stream:
      self.stream.stop()
      self.stream.close()

  def _run(self):
    """音声データをキューから取得し、無音で認識をトリガーする"""
    print("🎙️ 音声入力待機中...")
    # audio_data_list = []
    # silent_frames = 0
    # silence_threshold_frames = int(self.silence_limit / self.frame_duration)

    while self.running:

      # ✅ 修正: VADRecorderに発話の開始と終了を任せ、結合済みの音声データを受け取る
      # タイムアウトは detect_speech_segment 内の read_chunk(timeout=1.0) で処理される
      full_audio = self.vad_recorder.detect_speech_segment() 
      
      if full_audio is not None:
        # 音声データがあれば、認識処理スレッドを起動
        threading.Thread(target=self._process_audio, args=(full_audio,), daemon=True).start()
        
        print("VAD検知スレッドを停止しました。")
    
      # try:
      #   data = self.q.get(timeout=self.frame_duration)
      #   audio_data_list.append(data)
      #   silent_frames = 0
      # except queue.Empty:
      #   silent_frames += 1
        
      #   if silent_frames > silence_threshold_frames and audio_data_list:
      #     full_audio = np.concatenate(audio_data_list, axis=0)
      #     threading.Thread(target=self._process_audio, args=(full_audio,), daemon=True).start()
          
      #     audio_data_list = []
      #     silent_frames = 0

  def _process_audio(self, audio):
    """音声認識の実行と結果のコールバック"""
    print("🧩 音声認識中...")
    
    segments, info = self.model.transcribe(
      audio, 
      language="ja",
      vad_filter=False, 
    )
    
    text = " ".join([segment.text for segment in segments]).strip()

    if text:
      # シグナルを発行
      self.signals.recognized.emit(text) 

# — LLM応答部 — #
class LLMResponder:
  """
  LLMを使って、音声認識結果に返答が必要か判定し、応答を生成
  """
  # ✅ 最新のQwenモデルIDを使用
  def __init__(self, signals: LlmSignals, model_id="Qwen/Qwen3-1.7B", model=None, tokenizer=None):
    self.signals = signals 

    if model:
      self.model = model
      self.tokenizer = tokenizer
    else:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      print(f"Loading LLM model: {model_id} on {self.device}...")
      
      self.tokenizer = AutoTokenizer.from_pretrained(model_id)
      self.model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        trust_remote_code=False 
      )

  def process(self, text):
    """発話が問いかけかどうかを判定し、必要なら応答を生成"""

    # あなたは日本語アシスタントです。

    prompt = f"""
次のユーザーからの発話に対して、質問や指示、または応答が必要な内容であれば、その応答を出力してください。
単なる挨拶や独り言など、応答が不要な場合は「(スルー)」とだけ返してください。

ユーザー: {text}
アシスタント: """
    
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    
    outputs = self.model.generate(
      **inputs, 
      max_new_tokens=150, 
      do_sample=True,
      temperature=0.7,
      pad_token_id=self.tokenizer.eos_token_id,
    )
    
    reply = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    if "(スルー)" in reply or reply.lower().strip() == "(スルー)":
      print()
      return None
      
    return reply.split("アシスタント:")[-1].strip()

# — GUI部 — #
class MainWindow(QMainWindow):
  
  def __init__(self):
    super().__init__()
    self.setWindowTitle("🎧 音声入力対応ローカルAI")
    self.textbox = QTextEdit()
    self.setCentralWidget(self.textbox)
    self.recognizer = None
    self.llm = None
    self.audio_enabled = False
    
    # シグナルオブジェクトをインスタンス化
    self.signals = LlmSignals()
    self._connect_signals()

    # ✅ 修正: メソッド定義が省略されていた init_menu をここで呼び出し
    self.init_menu() 
    
    self.textbox.append("🤖 AIモデルをロード中...")
    threading.Thread(target=self._load_llm, daemon=True).start()
    
  # ✅ 修正: init_menu メソッドを再定義
  def init_menu(self):
    """右クリックメニューの設定"""
    self.menu = QMenu(self)
    self.toggle_action = QAction("音声入力をON", self, checkable=True)
    self.toggle_action.triggered.connect(self.toggle_audio)
    self.toggle_action.setEnabled(False) # ロード完了まで無効
    self.menu.addAction(self.toggle_action)
    
    # ✅ 修正1: テキストボックスにコンテキストメニューポリシーを設定
    self.textbox.setContextMenuPolicy(Qt.CustomContextMenu) 
    # ✅ 修正2: テキストボックスのシグナルに接続
    self.textbox.customContextMenuRequested.connect(self.show_menu)
        
  def show_menu(self, pos):
    """メニュー表示"""
    # self.menu.exec_(self.mapToGlobal(pos))
    """メニュー表示"""
    # mapToGlobal を使って、ウィジェット座標を画面全体座標に変換してメニューを表示
    self.menu.exec_(self.textbox.mapToGlobal(pos))
    
  def _connect_signals(self):
    """シグナルとスロットを接続"""
    self.signals.llm_loaded.connect(self.on_llm_loaded)
    self.signals.llm_error.connect(self.on_llm_error)
    self.signals.recognized.connect(self.on_recognized_gui_thread)
    self.signals.llm_responded.connect(self.display_llm_response)

  def _load_llm(self):
    """LLMのロード (時間のかかる処理)"""
    try:
      # LLMResponder に signals を渡す
      self.llm = LLMResponder(self.signals) 
      self.signals.llm_loaded.emit() 
    except Exception as e:
      # Qwen/Qwen3-1.7B-Instruct は大容量であるため、GPU環境設定やメモリ不足でエラーが出やすい
      error_msg = f"LLMロード中に重大なエラーが発生しました。\n原因: {e}"
      # エラーメッセージとともにシグナルを発行
      self.signals.llm_error.emit(error_msg) 

  @pyqtSlot()
  def on_llm_loaded(self):
    """LLMロード完了後のGUI更新"""
    self.textbox.append("✅ AIモデルのロードが完了しました。")
    self.toggle_action.setEnabled(True) 

  @pyqtSlot(str) 
  def on_llm_error(self, error_message):
    """LLMロード失敗時のGUI更新"""
    self.textbox.append(f"❌ {error_message}")

  def toggle_audio(self):
    """音声入力のON/OFF切り替え"""
    if self.llm is None:
      self.toggle_action.setChecked(False)
      self.textbox.append("⚠️ AIモデルがまだロードされていません。")
      return

    if self.toggle_action.isChecked():
      self.audio_enabled = True
      # SpeechRecognizer に signals を渡す
      self.recognizer = SpeechRecognizer(self.signals) 
      threading.Thread(target=self.recognizer.start, daemon=True).start()
      self.textbox.append("🎙️ 音声入力を開始しました。")
    else:
      self.audio_enabled = False
      if self.recognizer:
        self.recognizer.stop()
        self.recognizer = None
      self.textbox.append("🛑 音声入力を停止しました。")

  @pyqtSlot(str) 
  def on_recognized_gui_thread(self, text):
    """音声認識結果をGUIスレッドで処理"""
    self.textbox.append(f"👤 あなた: {text}")
    threading.Thread(target=self._process_llm, args=(text,), daemon=True).start()

  def _process_llm(self, text):
    """LLMでの応答生成 (時間のかかる処理)"""
    try:
      response = self.llm.process(text) 
      if response:
        self.signals.llm_responded.emit(response)
    except Exception as e:
      print(f"LLM処理エラー: {e}") 

  @pyqtSlot(str) 
  def display_llm_response(self, response):
    """LLMの応答をGUIスレッドで表示"""
    self.textbox.append(f"🤖 AI: {response}")


if __name__ == "__main__":
  app = QApplication(sys.argv)
  win = MainWindow()
  win.resize(600, 400)
  win.show()
  sys.exit(app.exec_())