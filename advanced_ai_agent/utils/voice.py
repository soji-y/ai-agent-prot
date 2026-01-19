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
from voicevox_core import AccelerationMode
# from voice.voicevox_core import AccelerationMode
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile
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

# ONNXRUNTIME_DLL = "./voice/onnxruntime/lib/voicevox_onnxruntime.dll"
# JTALK_DIC = "./voice/dict/open_jtalk_dic_utf_8-1.11"
# OUTPUT_DIR = "./voice/output"
# VOICE_MODEL_DIR = "./voice/vvms"

ONNXRUNTIME_DLL = "./onnxruntime/lib/voicevox_onnxruntime.dll"
JTALK_DIC = "./dict/open_jtalk_dic_utf_8-1.11"
OUTPUT_DIR = "./output"
VOICE_MODEL_DIR = "./vvms"

VOICE_PATH = {
    0: "0.vvm", # 四国めたん | あまあま | 0 |
    1: "0.vvm", # ずんだもん | あまあま | 1 |
    2: "0.vvm", # 四国めたん | ノーマル | 2 |
    3: "0.vvm", # ずんだもん | ノーマル | 3 |
    4: "0.vvm", # 四国めたん | セクシー | 4 |
    5: "0.vvm", # ずんだもん | セクシー | 5 |
    6: "0.vvm", # 四国めたん | ツンツン | 6 |
    7: "0.vvm", # ずんだもん | ツンツン | 7 |
    8: "0.vvm", # 春日部つむぎ | ノーマル | 8 |
    9: "3.vvm", # 波音リツ | ノーマル | 9 |
    10: "0.vvm", # 雨晴はう | ノーマル | 10 |
    11: "4.vvm", # 玄野武宏 | ノーマル | 11 |
    12: "9.vvm", # 白上虎太郎 | ふつう | 12 |
    13: "15.vvm", # 青山龍星 | ノーマル | 13 |
    14: "1.vvm", # 冥鳴ひまり | ノーマル | 14 |
    15: "2.vvm", # 九州そら | あまあま | 15 |
    16: "2.vvm", # 九州そら | ノーマル | 16 |
    17: "2.vvm", # 九州そら | セクシー | 17 |
    18: "2.vvm", # 九州そら | ツンツン | 18 |
    19: "5.vvm", # 九州そら | ささやき | 19 |
    20: "15.vvm", # もち子さん | ノーマル | 20 |
    21: "4.vvm", # 剣崎雌雄 | ノーマル | 21 |
    22: "5.vvm", # ずんだもん | ささやき | 22 |
    23: "8.vvm", # WhiteCUL | ノーマル | 23 |
    24: "8.vvm", # WhiteCUL | たのしい | 24 |
    25: "8.vvm", # WhiteCUL | かなしい | 25 |
    26: "8.vvm", # WhiteCUL | びえーん | 26 |
    27: "7.vvm", # 後鬼 | 人間ver. | 27 |
    28: "7.vvm", # 後鬼 | ぬいぐるみver. | 28 |
    29: "6.vvm", # No.7 | ノーマル | 29 |
    30: "6.vvm", # No.7 | アナウンス | 30 |
    31: "6.vvm", # No.7 | 読み聞かせ | 31 |
    32: "9.vvm", # 白上虎太郎 | わーい | 32 |
    33: "9.vvm", # 白上虎太郎 | びくびく | 33 |
    34: "9.vvm", # 白上虎太郎 | おこ | 34 |
    35: "9.vvm", # 白上虎太郎 | びえーん | 35 |
    36: "5.vvm", # 四国めたん | ささやき | 36 |
    37: "5.vvm", # 四国めたん | ヒソヒソ | 37 |
    38: "5.vvm", # ずんだもん | ヒソヒソ | 38 |
    39: "10.vvm", # 玄野武宏 | 喜び | 39 |
    40: "10.vvm", # 玄野武宏 | ツンギレ | 40 |
    41: "10.vvm", # 玄野武宏 | 悲しみ | 41 |
    42: "10.vvm", # ちび式じい | ノーマル | 42 |
    43: "11.vvm", # 櫻歌ミコ | ノーマル | 43 |
    44: "11.vvm", # 櫻歌ミコ | 第二形態 | 44 |
    45: "11.vvm", # 櫻歌ミコ | ロリ | 45 |
    46: "15.vvm", # 小夜/SAYO | ノーマル | 46 |
    47: "11.vvm", # ナースロボ＿タイプＴ | ノーマル | 47 |
    48: "11.vvm", # ナースロボ＿タイプＴ | 楽々 | 48 |
    49: "11.vvm", # ナースロボ＿タイプＴ | 恐怖 | 49 |
    50: "11.vvm", # ナースロボ＿タイプＴ | 内緒話 | 50 |
    51: "12.vvm", # †聖騎士 紅桜† | ノーマル | 51 |
    52: "12.vvm", # 雀松朱司 | ノーマル | 52 |
    53: "12.vvm", # 麒ヶ島宗麟 | ノーマル | 53 |
    54: "13.vvm", # 春歌ナナ | ノーマル | 54 |
    55: "13.vvm", # 猫使アル | ノーマル | 55 |
    56: "13.vvm", # 猫使アル | おちつき | 56 |
    57: "13.vvm", # 猫使アル | うきうき | 57 |
    58: "13.vvm", # 猫使ビィ | ノーマル | 58 |
    59: "13.vvm", # 猫使ビィ | おちつき | 59 |
    60: "13.vvm", # 猫使ビィ | 人見知り | 60 |
    61: "3.vvm", # 中国うさぎ | ノーマル | 61 |
    62: "3.vvm", # 中国うさぎ | おどろき | 62 |
    63: "3.vvm", # 中国うさぎ | こわがり | 63 |
    64: "3.vvm", # 中国うさぎ | へろへろ | 64 |
    65: "3.vvm", # 波音リツ | クイーン | 65 |
    66: "15.vvm", # もち子さん | セクシー／あん子 | 66 |
    67: "14.vvm", # 栗田まろん | ノーマル | 67 |
    68: "14.vvm", # あいえるたん | ノーマル | 68 |
    69: "14.vvm", # 満別花丸 | ノーマル | 69 |
    70: "14.vvm", # 満別花丸 | 元気 | 70 |
    71: "14.vvm", # 満別花丸 | ささやき | 71 |
    72: "14.vvm", # 満別花丸 | ぶりっ子 | 72 |
    73: "14.vvm", # 満別花丸 | ボーイ | 73 |
    74: "14.vvm", # 琴詠ニア | ノーマル | 74 |
    75: "15.vvm", # ずんだもん | ヘロヘロ | 75 |
    76: "15.vvm", # ずんだもん | なみだめ | 76 |
    77: "15.vvm", # もち子さん | 泣き | 77 |
    78: "15.vvm", # もち子さん | 怒り | 78 |
    79: "15.vvm", # もち子さん | 喜び | 79 |
    80: "15.vvm", # もち子さん | のんびり | 80 |
    81: "15.vvm", # 青山龍星 | 熱血 | 81 |
    82: "15.vvm", # 青山龍星 | 不機嫌 | 82 |
    83: "15.vvm", # 青山龍星 | 喜び | 83 |
    84: "15.vvm", # 青山龍星 | しっとり | 84 |
    85: "15.vvm", # 青山龍星 | かなしみ | 85 |
    86: "15.vvm", # 青山龍星 | 囁き | 86 |
    87: "16.vvm", # 後鬼 | 人間（怒り）ver. | 87 |
    88: "16.vvm", # 後鬼 | 鬼ver. | 88 |
    89: "17.vvm", # Voidoll | ノーマル | 89 |
    90: "18.vvm", # ぞん子 | ノーマル | 90 |
    91: "18.vvm", # ぞん子 | 低血圧 | 91 |
    92: "18.vvm", # ぞん子 | 覚醒 | 92 |
    93: "18.vvm", # ぞん子 | 実況風 | 93 |
    94: "18.vvm", # 中部つるぎ | ノーマル | 94 |
    95: "18.vvm", # 中部つるぎ | 怒り | 95 |
    96: "18.vvm", # 中部つるぎ | ヒソヒソ | 96 |
    97: "18.vvm", # 中部つるぎ | おどおど | 97 |
    98: "18.vvm", # 中部つるぎ | 絶望と敗北 | 98 |
}


# テキストを音声に変換するクラス
class VoiceSound:
  def __init__(self, mode:AccelerationMode = "AUTO", style_id=0, output_dir:str=None, play_flg=True, async_flg=False, change_flg=False, pitch=1.0, formant=1.0):
    
    # 現在のスクリプトの絶対パスを取得
    current_dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

    self.init_flg = False
    self.model_set_flg = False

    self.output_dir = output_dir # WAV出力フラグフォルダパス
    self.play_flg = play_flg # 再生フラグ
    self.async_flg = async_flg # 同期再生フラグ
    self.change_flg = change_flg # ボイスチェンジ
    self.pitch = pitch # ボイスチェンジ(Pitch Shift)
    self.formant = formant # ボイスチェンジ(Formant Shift)
    
    if not (current_dir_path / ONNXRUNTIME_DLL).exists():
      return
    
    onnxruntime_path = current_dir_path / ONNXRUNTIME_DLL
    jtalk_path = current_dir_path / JTALK_DIC
    
    onnxruntime = Onnxruntime.load_once(filename=str(onnxruntime_path))
    self.synthesizer = Synthesizer(
        onnxruntime,
        OpenJtalk(str(jtalk_path)),
        acceleration_mode=mode,
        cpu_num_threads=max(
          multiprocessing.cpu_count(), 2
        ),
    )

    vvm_path = current_dir_path / VOICE_MODEL_DIR / VOICE_PATH[style_id]
    
    if not vvm_path.exists():
      return
    
    # モデルをセット
    self.set_model(vvm_path, style_id)
    
    # 出力フォルダを作成
    # if output_dir:
    #   os.makedirs(output_dir, exist_ok=True)
    
    # 初期化完了
    self.init_flg = True
    print_str = f"* VoiceVox-Voice Initialized: style_id[{style_id}], voice_change[{change_flg}]"
    if change_flg:
      print_str += f"(pitch[{pitch}, formant[{formant}])"
    print(print_str)

  # WAVデータ再生
  def _play_wav(self, wav_bytes):
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

  # 変換できない文字を削除
  def _clean_for_voicevox(self, text:str) -> str:
    # 制御文字を除去（タブや改行は一旦残すなら '\x00-\x08\x0B-\x0C\x0E-\x1F' でも可）
    text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)

    # 絵文字・非BMP文字を除去
    emoji_pattern = re.compile(
      "["
      "\U00010000-\U0010FFFF"  # 非BMP（サロゲートペア文字、絵文字も含む）
      "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    # よくある機種依存・特殊記号を除去（必要に応じて追加）
    special_chars = r'[⓪-⓿㊤-㊿①-⑳⑴-⒇♡♥★☆♪♫※†‡•◆◉◎◯▽▲▼△]'
    text = re.sub(special_chars, '', text)

    return text

  # ファイル名に使えない文字（Windows基準）を除去
  def _sanitize_filename(self, text:str, max_length:int = 30) -> str:
    # 禁止文字: \ / : * ? " < > | （および制御文字 \x00-\x1f）
    sanitized = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '', text)
    
    # 先頭から指定文字数だけを取得（全角・半角問わず単純なスライス）
    return sanitized[:max_length]

  # モデルをセット
  def set_model(self, vvm_path:str="./vvms/0.vvm", style_id:int=0):
    with VoiceModelFile.open(vvm_path) as model:
      self.synthesizer.load_voice_model(model)
    self.style_id = style_id # 再生ID
    self.model_set_flg = True # モデル設定済みフラグ

  # ボイス作成
  def create_voice(self, text:str) -> None:
    if not self.init_flg:
      return
    
    if not self.model_set_flg:
      print(f"Not Set Model... Call 'set_model()'.")
      return

    # 不要な文字を削除
    conv_text = self._clean_for_voicevox(text)
    
    if conv_text != "":      
      audio_query = self.synthesizer.create_audio_query(conv_text, self.style_id)
      wav = self.synthesizer.synthesis(audio_query, self.style_id)

      # ボイスチェンジ
      if self.change_flg:
        vc = VoiceChanger()
        wav = vc.create_and_play(wav,
                               pitch_shift=self.pitch,
                               formant_shift=self.formant
                               )

      if self.output_dir:
        # output_path = Path(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        filesfx = self._sanitize_filename(text, 20)
        # wav_path = output_path / (f"{filename}_{datetime.datetime.now().strftime('%y%m%d%H%M%S')}.wav")
        wav_path = self.output_dir / (f"{datetime.datetime.now().strftime('%y%m%d%H%M%S')}_{filesfx}.wav")
        wav_path.write_bytes(wav)
        print(f"Output Wav file -> [{wav_path}]")

      # 音声再生
      if self.play_flg:
        if self.async_flg:
          # print("PyAudio play async...")
          thread = threading.Thread(target=self._play_wav, args=[wav], daemon=True)
          thread.start()
        else:
          # print("PyAudio play...")
          self._play_wav(wav)
          

# ボイスチェンジャー
class VoiceChanger:
  def __init__(self):
    pass
  
  def change_voice(self, wav_bytes, pitch_shift=1.2, formant_shift=1.0,
                   roughness=0.0, robotize=False, breathiness=0.0, speed=1.0):
    # 読み込み
    audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = audio.frame_rate

    if audio.channels == 2:
      samples = samples.reshape((-1, 2)).mean(axis=1)

    # speed 変更などで float32 になる可能性があるので、ここで統一
    samples = samples.astype(np.float64)

    # 時間伸縮（速度）
    if speed != 1.0:
      samples = librosa.util.fix_length(samples, size=len(samples))  # 長さ調整
      samples = librosa.resample(samples, orig_sr=sr, target_sr=sr)  # 念のため
      samples = librosa.effects.time_stretch(samples, rate=speed)

    # WORLD分解
    f0, time_axis = pw.harvest(samples, sr)
    # 小さく滑らかに
    f0 = scipy.ndimage.gaussian_filter1d(f0, sigma=1)
    
    sp = pw.cheaptrick(samples, f0, time_axis, sr)
    ap = pw.d4c(samples, f0, time_axis, sr)

    # ロボットボイス化
    if robotize:
      f0[:] = np.mean(f0[f0 > 0])  # 非ゼロ部分の平均に固定

    # ピッチ変更
    f0 *= pitch_shift

    # ジッターによる roughness
    if roughness > 0.0:
      jitter = np.random.randn(*f0.shape) * f0 * roughness
      f0 += jitter
      f0 = np.clip(f0, 50, sr // 2)

    # フォルマント変更
    if formant_shift != 1.0:
      new_sp = np.zeros_like(sp)

      # 置き換え
      if formant_shift != 1.0:
        new_sp = np.zeros_like(sp)
        new_len = int(sp.shape[1] * formant_shift)
        for i in range(sp.shape[0]):
          scaled = resample(sp[i], new_len)
          if len(scaled) < sp.shape[1]:
            scaled = np.pad(scaled, (0, sp.shape[1] - len(scaled)), mode='edge')
          else:
            scaled = scaled[:sp.shape[1]]
          new_sp[i] = scaled
        sp = new_sp
    
      # for i in range(sp.shape[0]):
      #   scaled = np.interp(
      #     np.linspace(0, sp.shape[1] - 1, int(sp.shape[1] * formant_shift)),
      #     np.arange(sp.shape[1]),
      #     sp[i]
      #   )
      #   if len(scaled) < sp.shape[1]:
      #     scaled = np.pad(scaled, (0, sp.shape[1] - len(scaled)), mode='edge')
      #   else:
      #     scaled = scaled[:sp.shape[1]]
      #   new_sp[i] = scaled
      # sp = new_sp

    # 息っぽさ（高周波ノイズを加える）
    if breathiness > 0.0:
      noise = np.random.randn(*samples.shape) * np.std(samples) * breathiness
      samples += noise
      f0, time_axis = pw.harvest(samples, sr)  # 再解析
      sp = pw.cheaptrick(samples, f0, time_axis, sr)
      ap = pw.d4c(samples, f0, time_axis, sr)

    # 合成
    # synthesized = pw.synthesize(f0, sp, ap, sr).astype(np.int16)
    synthesized = np.clip(pw.synthesize(f0, sp, ap, sr), -32768, 32767).astype(np.int16)

    out_wav = BytesIO()
    wavfile.write(out_wav, sr, synthesized)
    return out_wav.getvalue()

  def _play_wav(self, wav_bytes):
    audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(audio.sample_width),
                    channels=audio.channels,
                    rate=audio.frame_rate,
                    output=True)
    stream.write(audio.raw_data)
    stream.stop_stream()
    stream.close()
    p.terminate()

  def create_and_play(self, wav_bytes, pitch_shift=1.0, formant_shift=1.0, roughness=0.0, robotize=False, breathiness=0.0, speed=1.0, play_fig=False):

    wav_out = self.change_voice(wav_bytes, pitch_shift, formant_shift,
                                roughness, robotize, breathiness, speed)
    if play_fig:
      self._play_wav(wav_out)
    return wav_out
  

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
    print_str = f"* AivisSpeech-Voice Initialized: style_id[{style_id}], voice_change[{change_flg}]"
    if change_flg:
      print_str += f"(pitch[{pitch}, formant[{formant}])"
    print(print_str)
         
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
        
  # ボイス作成
  def create_voice(self, text:str):  
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
      if self.change_flg:
        vc = VoiceChanger()
        wav = vc.create_and_play(wav,
                                pitch_shift=self.pitch,
                                formant_shift=self.formant
                                )
      
      # 音声再生
      if self.play_flg:
        if self.async_flg:
          # 先に動いているスレッドを停止させる
          self.stop_flag.set()
          if self.thread and self.thread.is_alive():
            self.thread.join()
          self.stop_flag.clear()
          # print("PyAudio play async...")
          self.thread = threading.Thread(target=self._play_wav, args=[wav], daemon=True)
          self.thread.start()
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

import asyncio
import soundfile
from pathlib import Path
# エンジンのコア機能をインポート
from voicevox_engine.core.core_initializer import initialize_cores
from voicevox_engine.tts_pipeline.tts_engine import TTSEngine
from voicevox_engine.tts_pipeline.model import FrameAudioQuery
from voicevox_engine.core.core_initializer import CoreManager, initialize_cores
from voicevox_engine.core.core_wrapper import CoreWrapper
from voicevox_engine.core.core_adapter import CoreAdapter
from voicevox_engine.core.core_initializer import CoreManager  # ←こちら

# --- 設定項目 ---
TEXT_TO_SYNTHESIZE = "こんにちは、これはPythonスクリプトからの音声合成テストです。"
SPEAKER_ID = 0  # 話者ID (モデルによって異なります)
STYLE_ID = 1431611904
OUTPUT_WAV_PATH = "output.wav"
# -----------------
class VoiceSoundAivmx:
  def __init__(self, style_id:int=0, play_flg=True, output_fig=False, async_flg=False):
    """
    AivisSpeech-Engineを直接呼び出して音声合成を行うメイン関数
    """
    self.init_flg=False

    self.style_id = style_id
    self.play_flg = play_flg
    self.output_flg = output_fig
    self.async_flg = async_flg

    # style_list = self.get_available_styles()
    # if not style_id in style_list.keys():
    #   return

    self.init_flg=True
    
    # エンジンの初期化
    # CPUで実行する場合は use_gpu=False に設定
    # initialize_cores(use_gpu=True, cpu_num_threads=0)
    


    # コアを初期化
    core_manager = CoreManager()
    # .aivmx があるフォルダ
    core_path = Path(r"C:\Users\Soji\Documents\AI_MyProgram\Project_AI\CHARADEAR\character\voice\aivmx")
    # core_path = Path(r"C:\Users\Soji\AppData\Roaming\AivisSpeech-Engine")

    # CoreWrapper を生成（GPU 使用する場合 True に変更）
    core_wrapper = CoreWrapper(use_gpu=False, core_dir=core_path, cpu_num_threads=16, load_all_models=True)

    core_manager = initialize_cores(use_gpu=False, cpu_num_threads=16, voicelib_dirs=[core_path], enable_mock=False, load_all_models=True)


    # CoreAdapter に変換して CoreManager に登録
    core_manager.register_core(CoreAdapter(core_wrapper), "0.0.0") #, core_wrapper.metas()[0]["version"])

    # 利用可能なコア一覧を取得
    for version, core_wrapper in core_manager.items():
       print("Version:", version, "CoreWrapper:", core_wrapper)

    # 例として最初のコアを使う場合
    version, core_wrapper = next(iter(core_manager.items()))

    # core_wrapper = core_manager["モデル名"]  # 取得できる CoreWrapper

    # TtsEngineのインスタンスを作成
    engine = TTSEngine(core_wrapper)

    print("エンジンが初期化されました。音声合成を開始します...")
    
    # 1. Accent Phrase (アクセント句) を生成
    accent_phrases = engine.create_accent_phrases(TEXT_TO_SYNTHESIZE, style_id=STYLE_ID)

    # 2. Audio Query (音声合成用のクエリ) を生成
    query = FrameAudioQuery(
        accent_phrases=accent_phrases,
        speedScale=1.0,
        pitchScale=0.0,
        intonationScale=1.0,
        volumeScale=1.0,
        prePhonemeLength=0.1,
        postPhonemeLength=0.1,
        outputSamplingRate=24000,
        outputStereo=False,
    )

    # 3. 音声波形 (Wave) を合成
    wave_bytes = engine.synthesize_wave(query, speaker=SPEAKER_ID)

    # 4. .wavファイルとして保存
    output_path = Path(OUTPUT_WAV_PATH)
    soundfile.write(file=output_path, data=wave_bytes, samplerate=query.outputSamplingRate)
    
    print(f"音声合成が完了しました: {output_path.resolve()}")

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
  
  voc = VoiceSoundAivmx(style_id=style_id)
  
  # voc = VoiceSoundAivAPI(style_id=1431611904)
  # voc = VoiceSoundAivAPI(style_id=0)
  voc.create_voice("こんにちわ")
  # asyncio.run(main())
  print("finished!")
    