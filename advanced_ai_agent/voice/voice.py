import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import re
import yaml
import threading
import time
import datetime
import random
import datetime
import threading
import multiprocessing
from io import BytesIO    
from pydub import AudioSegment
from pathlib import Path
# from voicevox_core import AccelerationMode
# from voice.voicevox_core import AccelerationMode
# from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile
# from vokice.voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile
import pytz
import pyaudio
import pyworld as pw
import numpy as np
import librosa
from scipy.io import wavfile
import scipy.ndimage
from scipy.signal import resample

from pydub import AudioSegment
from io import BytesIO
import pyaudio

# コールバックスレッドクラス
class CallbackThread(threading.Thread):
  def __init__(self, target, args=(), callback=None, callback_args=(), daemon=False):
    super().__init__(daemon=daemon)
    self._target = target
    self._args = args
    self._callback = callback
    self._callback_args = callback_args

  def run(self):
    if self._target:
      self._target(*self._args)
    if self._callback:
      self._callback(*self._callback_args)
            
import requests
class VoiceSoundAivAPI:
  
  def __init__(self, url:str="http://127.0.0.1:10101", style_id:int=0, output_dir:str=None, play_flg:bool=True, async_flg:bool=False, change_flg:bool=False, pitch=1.0, formant=1.0):

    self.init_flg=False

    self.url = url # APIのURL
    self.style_id = style_id # スタイルID
    self.output_dir = output_dir # WAV出力フォルダパス
    self.play_flg = play_flg # 再生
    self.async_flg = async_flg # 非同期再生
    self.change_flg = change_flg # ボイスチェンジ有無
    self.pitch = pitch # ボイスチェンジ(Pitch Shift)
    self.formant = formant # ボイスチェンジ(Formant Shift)

    try:
      style_list = self.get_available_styles()
      if not style_id in style_list.keys():
        return
    except Exception as e:
      print(f"Error: {str(e)}")
      return

    self.init_flg=True
    
    # print_str = f"* AivisSpeech-Voice Initialized: style_id[{style_id}]" #, voice_change[{change_flg}]"
    # if change_flg:
    #   print_str += f"(pitch[{pitch}, formant[{formant}])"
    # print(print_str)
         
    # 非同期で音声再生するスレッド 
    self.thread = None
    self.stop_flag = threading.Event()

    # 出力フォルダを作成
    # if output_dir:
    #   os.makedirs(output_dir, exist_ok=True)
    
    # # テスト
    # try:
    #   text = "テスト"
    #   query_url = f"{self.url}/audio_query?text={text}&speaker={self.style_id}"
    #   query = requests.post(query_url).json()
    #   synth_url = f"{self.url}/synthesis?speaker={self.style_id}"
    #   response = requests.post(synth_url, json=query)
    #   wav = response.content
    #   # audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    #   # p = pyaudio.PyAudio()    
    #   # # ストリームを開く
    #   # stream = p.open(format=p.get_format_from_width(audio.sample_width),
    #   #                 channels=audio.channels,
    #   #                 rate=audio.frame_rate,
    #   #                 output=True)
    #   self.init_flg=True
    # except Exception as e:
    #   print(f"Error: {str(e)}")

  def get_available_styles(self):
    speakers = requests.get(f"{self.url}/speakers").json()
    styles = {}
    for sp in speakers:
        for st in sp["styles"]:
            styles[st["id"]] = f"{sp['name']} ({st['name']})"
    return styles
          
  # WAVデータ再生
  def _play_wav_single(self, wav_bytes):
    try:
      bio = BytesIO(wav_bytes)
      bio.seek(0)
      
      audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    
      p = pyaudio.PyAudio()    
      # ストリームを開く
      stream = p.open(format=p.get_format_from_width(audio.sample_width),
                      channels=audio.channels,
                      rate=audio.frame_rate,
                      output=True)

      # 全てのデータを書き込んで再生 (同期再生)
      stream.write(audio.raw_data)

      # ストリームを閉じる
      stream.stop_stream()
      stream.close()

      # PyAudioオブジェクトを終了
      p.terminate()
      
    except Exception as e:
      print(f"Error: {str(e)}")

  # WAVデータ再生(Chunk分割して途中停止可能)
  def _play_wav(self, wav_bytes):
    try:
      audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")

      # 再生用に前後に少し無音を追加
      silence = AudioSegment.silent(duration=1000, frame_rate=audio.frame_rate)  # 300ms
      audio = silence + audio + silence

      p = pyaudio.PyAudio()
      stream = p.open(format=p.get_format_from_width(audio.sample_width),
                      channels=audio.channels,
                      rate=audio.frame_rate,
                      output=True)

      # chunk_size = 1024
      # raw = audio.raw_data
      # for i in range(0, len(raw), chunk_size):
      #   if self.stop_flag.is_set():
      #     break
      #   stream.write(raw[i:i+chunk_size])

      # numpy配列に変換
      samples = np.array(audio.get_array_of_samples())

      # チャンクは「サンプル数」で決める
      chunk_samples = 1024  
      for i in range(0, len(samples), chunk_samples):
        if self.stop_flag.is_set():
          break
        chunk = samples[i:i+chunk_samples]
        stream.write(chunk.tobytes())

      stream.stop_stream()
      stream.close()
      p.terminate()

    except Exception as e:
      print(f"Error: {str(e)}")
      
    # ファイル名に使えない文字（Windows基準）を除去
  
  def _sanitize_filename(self, text:str, max_length:int = 30) -> str:
    # 禁止文字: \ / : * ? " < > | （および制御文字 \x00-\x1f）
    sanitized = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '', text)
    # 先頭から指定文字数だけを取得（全角・半角問わず単純なスライス）
    return sanitized[:max_length]
  
  def task2(self):
    print("ボイス完了")
    
  # ボイス作成
  def create_voice(self, text:str, call_back=None):  
    if not self.init_flg:
      return

    try:
      # 1. audio_query を取得
      query_url = f"{self.url}/audio_query?text={text}&speaker={self.style_id}"
      query = requests.post(query_url).json()

      # 2. synthesis で音声生成
      synth_url = f"{self.url}/synthesis?speaker={self.style_id}"
      response = requests.post(synth_url, json=query)
      wav = response.content
      
      # ボイスチェンジ
      # if self.change_flg:
      #   vc = VoiceChanger()
      #   wav = vc.create_and_play(wav,
      #                           pitch_shift=self.pitch,
      #                           formant_shift=self.formant
      #                           )
      
      # 音声再生
      if self.play_flg:
        if self.async_flg:
          # 先に動いているスレッドを停止させる
          self.stop_flag.set()
          if self.thread and self.thread.is_alive():
            self.thread.join()
          self.stop_flag.clear()
          # # print("PyAudio play async...")
          # self.thread = threading.Thread(target=self._play_wav, args=(wav), daemon=True)
          # call_back = self.task2()
          self.thread = CallbackThread(target=self._play_wav, args=(wav,), callback=call_back, daemon=True)
          self.thread.start()
          # self.thread.join()
        else:
          # print("PyAudio play...")
          self._play_wav(wav)
          
      # ファイル出力
      if self.output_dir:
        # output_path = Path(self.output_dir)
        # output_dir = Path(OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        filesfx = self._sanitize_filename(text, 20)
        wav_path = self.output_dir / (f"{datetime.datetime.now().strftime('%y%m%d%H%M%S')}_{filesfx}.wav")
        mp3_path = wav_path.with_suffix('.mp3')
        
        audio = AudioSegment.from_file(BytesIO(wav), format="wav")
        # 再生用に前後に少し無音を追加
        silence = AudioSegment.silent(duration=1000, frame_rate=audio.frame_rate)  # 300ms
        audio = silence + audio + silence
        
        # ファイルに保存
        # wav_path.write_bytes(wav)
        # audio.export(wav_path, format="wav")
        audio.export(mp3_path, format="mp3")

        print(f"Output Wav file -> [{wav_path}]")
        
    except Exception as e:
      print(f"Error: {str(e)}")
      return None
    
    return wav

# import asyncio
# import soundfile
# from pathlib import Path
# # エンジンのコア機能をインポート
# from voicevox_engine.core.core_initializer import initialize_cores
# from voicevox_engine.tts_pipeline.tts_engine import TTSEngine
# from voicevox_engine.tts_pipeline.model import FrameAudioQuery
# from voicevox_engine.core.core_initializer import CoreManager, initialize_cores
# from voicevox_engine.core.core_wrapper import CoreWrapper
# from voicevox_engine.core.core_adapter import CoreAdapter
# from voicevox_engine.core.core_initializer import CoreManager  # ←こちら

# --- 設定項目 ---
TEXT_TO_SYNTHESIZE = "こんにちは、これはPythonスクリプトからの音声合成テストです。"
SPEAKER_ID = 0  # 話者ID (モデルによって異なります)
STYLE_ID = 1431611904
OUTPUT_WAV_PATH = "output.wav"
# -----------------
# class VoiceSoundAivmx:
#   def __init__(self, style_id:int=0, play_flg=True, output_fig=False, async_flg=False):
#     """
#     AivisSpeech-Engineを直接呼び出して音声合成を行うメイン関数
#     """
#     self.init_flg=False

#     self.style_id = style_id
#     self.play_flg = play_flg
#     self.output_flg = output_fig
#     self.async_flg = async_flg

#     # style_list = self.get_available_styles()
#     # if not style_id in style_list.keys():
#     #   return

#     self.init_flg=True
    
#     # エンジンの初期化
#     # CPUで実行する場合は use_gpu=False に設定
#     # initialize_cores(use_gpu=True, cpu_num_threads=0)
    


#     # コアを初期化
#     core_manager = CoreManager()
#     # .aivmx があるフォルダ
#     core_path = Path(r"C:\Users\Soji\Documents\AI_MyProgram\Project_AI\CHARADEAR\character\voice\aivmx")
#     # core_path = Path(r"C:\Users\Soji\AppData\Roaming\AivisSpeech-Engine")

#     # CoreWrapper を生成（GPU 使用する場合 True に変更）
#     core_wrapper = CoreWrapper(use_gpu=False, core_dir=core_path, cpu_num_threads=16, load_all_models=True)

#     core_manager = initialize_cores(use_gpu=False, cpu_num_threads=16, voicelib_dirs=[core_path], enable_mock=False, load_all_models=True)


#     # CoreAdapter に変換して CoreManager に登録
#     core_manager.register_core(CoreAdapter(core_wrapper), "0.0.0") #, core_wrapper.metas()[0]["version"])

#     # 利用可能なコア一覧を取得
#     for version, core_wrapper in core_manager.items():
#        print("Version:", version, "CoreWrapper:", core_wrapper)

#     # 例として最初のコアを使う場合
#     version, core_wrapper = next(iter(core_manager.items()))

#     # core_wrapper = core_manager["モデル名"]  # 取得できる CoreWrapper

#     # TtsEngineのインスタンスを作成
#     engine = TTSEngine(core_wrapper)

#     print("エンジンが初期化されました。音声合成を開始します...")
    
#     # 1. Accent Phrase (アクセント句) を生成
#     accent_phrases = engine.create_accent_phrases(TEXT_TO_SYNTHESIZE, style_id=STYLE_ID)

#     # 2. Audio Query (音声合成用のクエリ) を生成
#     query = FrameAudioQuery(
#         accent_phrases=accent_phrases,
#         speedScale=1.0,
#         pitchScale=0.0,
#         intonationScale=1.0,
#         volumeScale=1.0,
#         prePhonemeLength=0.1,
#         postPhonemeLength=0.1,
#         outputSamplingRate=24000,
#         outputStereo=False,
#     )

#     # 3. 音声波形 (Wave) を合成
#     wave_bytes = engine.synthesize_wave(query, speaker=SPEAKER_ID)

#     # 4. .wavファイルとして保存
#     output_path = Path(OUTPUT_WAV_PATH)
#     soundfile.write(file=output_path, data=wave_bytes, samplerate=query.outputSamplingRate)
    
#     print(f"音声合成が完了しました: {output_path.resolve()}")

if __name__ == "__main__":
  style_id = 888753760 # Anneli(ノーマル)
  style_id = 888753761 # Anneli(通常)
  style_id = 888753762 # Anneli(テンション高め)
  style_id = 888753764 # Anneli(上機嫌)
  style_id = 888753763 # Anneli(落ち着き)
  style_id = 888753765 # Anneli(怒り・悲しみ)
  
  style_id = 864870016 # Anneli(Whisper/ノーマル)
  style_id = 864870017 # Anneli(Whisper/通常)
  style_id = 864870018 # Anneli(Whisper/テンション高め)
  style_id = 864870020 # Anneli(Whisper/上機嫌)
  style_id = 864870019 # Anneli(Whisper/落ち着き)
  style_id = 864870021 # Anneli(Whisper/怒り・悲しみ)
      
  style_id = 1431611904 # まい
  style_id = 345585728 # るな
  style_id = 604166016 # 中2
  style_id = 376644064 # 桜音
  style_id = 1325133120 # 花音
  
  style_id = 391794336 # ろてじん（長老ボイス）
  style_id = 345585728 # るな
  # style_id = 1937616896 # にせ


  messages = []
  messages.append("マスター、こんにちわ")
  messages.append("""
  今日のニュースについて、以下の通りですじゃ。
  - 政治：自民党と公明党が選挙区での候補擁立を検討。高市、玉木、石破など、主要人物の動きが注目されています。首相指名選挙では基本政策の一致が求められる見方もあります。
  - 国際：ハマスによる人質事件の解釈が開始され、人質の解放が進んでいる。
  - 社会：背中に傷を負った女性がクマに襲われた経験を語り、安全への不安が広がっています。また、1年間の育休を取ったパパが家庭や育児のあり方に ついて新たな気づきを得たことも伝えられています。
  - スポーツ：出雲駅伝で国学院大が2連覇を達成。4区の辻原輝が区間新記録を樹立。大阪万博の閉会式に櫻井翔がサプライズで出演。
  - 経済・ライフ：全国で急増する無料自動販売機の理由が注目され、生活の変化や需要の変化に繋がっています。割り勘や飲酒時の配慮といった日常の課題も議論されています。
  - 地域・文化：「船場吉兆」の次男が6坪の小さな店舗で「もてなしの心」を継いで再出発。名店廃業後の復活を目指す取り組みが話題です。
  - その他：鳥取県で農道のチェーンに大型バイクが接触し、運転手が死亡。高齢男性が孫が運転する車にバックで衝突して重体となった事故も発生。米倉涼子の恋人との関係に異変が発覚するなど、プライベートの問題も報道されていますですぞ。
    """)

  for idx, message in enumerate(messages):

    voc = VoiceSoundAivAPI(url="http://192.168.1.100:10101", style_id=style_id)
    # voc = VoiceSoundAivAPI(style_id=style_id)
    
    start_time = time.time()

    voc.create_voice(message, call_back=lambda: print(f"Time[{idx+1}/{len(messages)}] -> {end_time - start_time:.2f} seconds"))
    
    end_time = time.time()
    print(f"Time[{idx+1}/{len(messages)}] -> {end_time - start_time:.2f} seconds")
    print(message)

    # voc.create_voice()
    # voc.create_voice("マスター、こんにちわ！")
    # asyncio.run(main())
  print("finished!")
    