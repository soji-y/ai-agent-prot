import os
import glob
import yaml
import sys
import cv2
import json
import time
import ctypes
import threading
import numpy as np
from PIL import Image
import multiprocessing as mp
from datetime import datetime
from PIL import Image
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QBrush, QDragEnterEvent, QDropEvent, QPen
from PyQt5.QtWidgets import (
  QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
  QLineEdit, QPushButton, QGraphicsView, QActionGroup,
  QApplication, QMainWindow, QGraphicsScene, QAction, QMenu, QFrame, 
)
from PyQt5.QtCore import Qt, QTimer, QThread

# from simple_llm import SimpleLLM
from advanced_ai_agent import AdvancedAgent
from utils.speech_bubble import SpeechBubble
# from voice.voice import VoiceSoundAivAPI
from resizeble_graphics_view import ResizableGraphicsView

from game.otthello_board_view import OthelloBoardView

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# VIEWER_WIDTH = 480
# VIEWER_HIGHT = 527

USER_NAME = "マスター"
MAX_STEPS = 10 # 最大思考回数

VIEWER_SIZE = 300
INPUT_WIN_SIZE = 480 # 入力画像ウィンドウ
FONT_SIZE = 15
BASE_BUBBLE_DURATION = 3000 # ふきだしの基本の表示秒

# VIDEO_REVERSE = True # 逆再生
# FADE_MSEC = 0.5 # フェード秒数
LINE_BREAK = True # 読点後に改行

# =========================
# 入力画像ウィンドウ用のグラフィックビュー
# =========================
class InputGraphicsView(ResizableGraphicsView):
  imageDeleted = pyqtSignal() # 画像の削除
  dropImage = pyqtSignal(str) # ドラッグドロップの画像パス

  def __init__(self, parent=None):
    super().__init__(parent)
    
  # 右クリックメニューの表示
  def show_context_menu(self, pos):
    menu = QMenu(self)

    # if self.input_mode:
    delete_action = QAction("画像を削除", self)
    menu.addAction(delete_action)
    # アクションとスロットの接続
    delete_action.triggered.connect(self.delete_image)

    # メニューを表示
    if menu.actions():
      menu.exec_(self.mapToGlobal(pos))

  # --- ドラッグ&ドロップ対応 ---
  def dragEnterEvent(self, event: QDragEnterEvent):    
    if event.mimeData().hasUrls():
      event.acceptProposedAction()
    else:
      event.ignore()

  def dragMoveEvent(self, event):    
    if event.mimeData().hasUrls():
      event.acceptProposedAction()
    else:
      event.ignore()

  def dropEvent(self, event: QDropEvent):
    urls = event.mimeData().urls()
    if urls:
      path = urls[0].toLocalFile()
      if path:
        self.dropImage.emit(path)
        
    event.ignore()

  # 画像を削除するスロット
  def delete_image(self):
    # ディスプレイクリア
    self.display_clear()
   
    # 画像が削除されたことをMainWindowに通知
    self.imageDeleted.emit()
       
# =========================
# エージェントウィンドウ用のグラフィックビュー
# =========================
class AgentGraphicsView(ResizableGraphicsView):
  speechMode = pyqtSignal(bool) # 音声入力
  inputWindow = pyqtSignal() # 画像入力ウィンドウ
  selectAgent = pyqtSignal(int) # エージェントを選択
  gameWindow = pyqtSignal(int) # オセロウィンドウ

  def __init__(self, parent=None, vision_flg=False):
    super().__init__(parent)

    # 「画像入力画面」用のモード
    # self.input_mode = input_mode
    self.owner = parent

    # 入力画像ウィンドウの表示
    self.vision_flg = vision_flg

    # 音声入力
    self.speech_use = False
    
    # エージェントの切り替え用の名前リスト
    self.agent_names = []
    self.select_agent_idx = 0
    
    # ゲーム名
    self.game_names = []

    self.setAcceptDrops(True)


  # 右クリックメニューの表示
  def show_context_menu(self, pos):
    menu = QMenu(self)

    # 音声入力
    speech_action = QAction("音声入力", self)
    speech_action.setCheckable(True)
    speech_action.setChecked(self.speech_use)  # 現在の状態を反映
    
    # def update_speech_use(checked):
    #   self.speech_use = checked
    # # チェック状態が変わったら変数を更新
    # speech_action.toggled.connect(update_speech_use)
    menu.addAction(speech_action)
    
    # アクションとスロットの接続
    speech_action.toggled.connect(self.speech_enable)
    # speech_action.triggered.connect(self.speechMode.emit)    

    menu.addSeparator()
      
    if self.vision_flg:
      input_win_action = QAction("画像入力ウィンドウ", self)
      menu.addAction(input_win_action)
      
      # アクションとスロットの接続
      input_win_action.triggered.connect(self.inputWindow.emit)    
      if self.agent_names:
        menu.addSeparator()
    
    if self.agent_names:
      # 「エージェント」サブメニューを作成
      agent_menu = menu.addMenu("エージェント")
      if self.owner.agent_running:
        agent_menu.setEnabled(False) # エージェント切り替えOFF

      # グループを作成（排他選択用）
      action_group = QActionGroup(menu)
      action_group.setExclusive(True)

      for i, name in enumerate(self.agent_names):
        agent_action = QAction(name, agent_menu)
        agent_action.setCheckable(True)
        
        # クリック時にインデックスを固定して渡す
        agent_action.triggered.connect(lambda checked=False, idx=i: self.select_agent(idx))
        
        action_group.addAction(agent_action)   # グループに登録
        agent_menu.addAction(agent_action)

      # 現在選択中のエージェントにチェックをつける
      if 0 <= self.select_agent_idx < len(action_group.actions()):
        action_group.actions()[self.select_agent_idx].setChecked(True)
  
      # 「エージェント」サブメニューを作成
      if self.game_names:
        game_menu = menu.addMenu("ゲーム")
        for i, name in enumerate(self.game_names):
          game_action = QAction(name, game_menu)
          # agent_action.setCheckable(True)
          
          # クリック時にインデックスを固定して渡す
          game_action.triggered.connect(lambda checked=False, idx=i: self.select_game(idx))
          
          # action_group.addAction(agent_action)   # グループに登録
          game_menu.addAction(game_action)
        
      # othello_action = QAction("オセロ", game_menu)
      # othello_action.triggered.connect(self.gameWindow.emit)
      
    # メニューを表示
    if menu.actions():
      menu.exec_(self.mapToGlobal(pos))
  
  # 音声入力チェック
  def speech_enable(self, chk):
    self.speech_use = chk
    self.speechMode.emit(chk)
  
  # エージェントの変更
  def select_agent(self, idx):
    self.select_agent_idx = idx
    self.selectAgent.emit(idx)
  
  # ゲームの選択
  def select_game(self, idx):
    # self.select_agent_idx = idx
    self.gameWindow.emit(idx)

# =========================
# Agent処理用スレッド
# =========================
class AgentWorker(QThread):
  finished = pyqtSignal(object, object)  # 応答を返すシグナル

  def __init__(self, agent, user_input, image_path=None, queue_thought=None, speech_flg=False): #image:Image=None):
    super().__init__()
    self.agent = agent
    self.user_input = user_input
    # self.image = image
    self.image_path = image_path
    self.queue_thought = queue_thought
    self.speech_flg= speech_flg # 音声入力

  def run(self):
    # LLMなど重い処理はここで実行
    # response = self.agent.run(self.user_input,  max_steps=MAX_STEPS, user_name=USER_NAME, image=self.image)
    start_time = time.time()
    response, action = self.agent.run(self.user_input,  max_steps=MAX_STEPS, user_name=USER_NAME, image_path=self.image_path, queue_thought=self.queue_thought, speech_flg=self.speech_flg)
    print(f"最終応答時間: {time.time()-start_time:.3f}s")
    self.finished.emit(response, action)  # メインスレッドへ戻す

# =========================
# 入力画像ウィンドウ
# =========================
class InputImageWindow(QMainWindow):
  def __init__(self, parent=None):
    super().__init__(parent)

    self.image_path:str = None
    self.image:Image = None

    # === UI設定 ===
    self.setWindowTitle(f"入力画像ウィンドウ")
        
    # 画面サイズ(作業領域)
    screen = QApplication.primaryScreen()
    rect = screen.availableGeometry()
    width = rect.width()
    height = rect.height()
    # print(f"モニタサイズ(作業領域): {width} x {height}")
          
    viewer_w = viewer_h = INPUT_WIN_SIZE
    
    self.setGeometry(int((width-viewer_w)/2), int((height-viewer_h)/2), viewer_w, viewer_h)

    central = QWidget(self)
    self.setCentralWidget(central)
    layout = QVBoxLayout(central)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    
    # self.viewer = ResizableGraphicsView(input_mode=True)
    self.viewer = InputGraphicsView(self)
    layout.addWidget(self.viewer)
        
    # layout = QVBoxLayout()
    # layout.addWidget(QLabel("ここにResizableGraphicsViewを配置"))
    # self.setLayout(layout)
    
    # === シグナルを接続 ===
    self.viewer.dropImage.connect(self.input_image)
    self.viewer.imageDeleted.connect(self.delete_image)

  # フォームを閉じたときのイベント
  def closeEvent(self, event):
    # ディスプレイクリア
    self.viewer.display_clear()

    # 画像情報削除
    self.delete_image()
    
    super().closeEvent(event)
     
  # 入力画像を設定
  def input_image(self, img_path, bg_color=(255, 255, 255)):
    self.image = None
    self.image_path = None
    try:
      if img_path and os.path.isfile(img_path):
        self.image_path = img_path
        img = Image.open(img_path)
        if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
          # αチャンネルを白背景に合成
          img = img.convert("RGBA")  # まず RGBA に統一
          bg_img = Image.new("RGB", img.size, bg_color)
          bg_img.paste(img, mask=img.split()[3])  # αチャンネルをマスクとして使用
        else:
          bg_img = img.convert("RGB")

        self.image = bg_img
        self.image_path = img_path
        
        # self.image = Image.open(img_path).convert("RGB")
        self.viewer.display_image(self.image)
    except Exception as e:
      print(f"Error: {e}")
    
  # 画像情報削除
  def delete_image(self):
    self.image_path = None
    self.image = None
    
    
# =========================
# エージェントビューワークラス
# =========================
class AgentViewer(QMainWindow):
  def __init__(self, agent):
    super().__init__()

    self.agent = agent
    self.video_paths = agent.video_paths
    self.video_reverse = agent.video_reverse
    self.video_fade_msec = agent.video_fade_msec

    # エージェント実行中
    self.agent_running = False
    
    # 入力画像ウィンドウ
    self.input_window = None
    
    # ゲームウィンドウ
    self.othello_window = None # オセロ
    
    # 音声入力フラグ
    self.speech_use = False
    
    # === UI設定 ===
    title_add = f"：{agent.name}" if agent.name else "0"
    self.setWindowTitle(f"AIエージェント{title_add}")
        
    # 画面サイズ
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()  # 高DPI対応
    self.screen_width = user32.GetSystemMetrics(0)
    self.screen_height = user32.GetSystemMetrics(1)
    # print(f"モニタサイズ: {self.screen_width} x {self.screen_height}")

    screen = QApplication.primaryScreen()
    rect = screen.availableGeometry()
    width = rect.width()
    height = rect.height()
    # print(f"モニタサイズ(作業領域): {width} x {height}")
    
    viewer_w = VIEWER_SIZE
    viewer_h = viewer_w + 40
    
    self.setGeometry(int((width-viewer_w)/2), int((height-viewer_h)/2), viewer_w, viewer_h)

    self.bubble_queue = mp.Queue() # バブルに表示するキュー
    self.bubble_queue_thought = mp.Queue()
    self.bubble = None
    
    vision_flg = self.agent.vision_model if self.agent else False
    
    central = QWidget(self)
    self.setCentralWidget(central)
    layout = QVBoxLayout(central)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    # === エージェント映像 ===
    # self.viewer = ResizableGraphicsView(input_mode=False, vision_flg=vision_flg)
    self.viewer = AgentGraphicsView(self, vision_flg=vision_flg)

    layout.addWidget(self.viewer)

    # エージェントの切り替え
    self.viewer.agent_names = [cfg["name"] for cfg in agent.agents_cfg]
    self.viewer.select_agent_idx = agent.select_idx
  
    # ゲームのツールがあるときだけメニューに表示
    self.viewer.game_names = []
    for tool in self.agent.tools.values():
      if tool.type() == "game":
        self.viewer.game_names.append(tool.title())
      
    # === 入力UI ===
    input_layout = QHBoxLayout()
    self.input_box = QLineEdit()
    self.input_box.setPlaceholderText("メッセージを入力...")
    self.send_button = QPushButton("送信")
    input_layout.addWidget(self.input_box)
    input_layout.addWidget(self.send_button)
    layout.addLayout(input_layout)

    self.input_box.setStyleSheet(f"font-size: {FONT_SIZE}px;")

    # === 初期フォーカス設定 ===
    self.input_box.setFocus()
    
    # === イベント接続 ===
    self.send_button.clicked.connect(self.on_send_message)
    self.input_box.returnPressed.connect(self.on_send_message)

    # === 動画再生準備 ===
    self.create_video_frame()
    
    # === シグナルを接続 ===
    self.viewer.speechMode.connect(self.speech_input) # 音声入力
    self.viewer.inputWindow.connect(self.view_input)
    self.viewer.selectAgent.connect(self.change_agent)
    self.viewer.gameWindow.connect(self.view_game)

    # === タイマーでフレーム更新 ===
    self.timer = QTimer(self)
    self.timer.timeout.connect(self._update_frame)
    self.timer.start(33)  # 約30fps
    self.timer.start(int(1/self.fps*1000)) # 約30fps

    # QTimerでループ処理
    q_timer = QTimer(self)
    q_timer.timeout.connect(self._check_bubble_queue)
    q_timer.start(100) # 100msチェック
 
  # 動画用のフレームを生成
  def create_video_frame(self):
    self.current_frame = 0
    self.frames, self.fps = self.load_video_sequence(self.video_paths, fade_duration=self.video_fade_msec, reverse_flg=self.video_reverse)

  # 動画をロード
  def load_video_sequence(self, video_paths, fade_duration=0.5, reverse_flg=False):
    def read_frames(cap):
      frames = []
      while True:
        ret, frame = cap.read()
        if not ret:
          break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
      return frames

    def create_fade_transition(last_frames, next_frames, fade_frames):
      transition = []
      last_len = len(last_frames)
      next_len = len(next_frames)
      for i in range(fade_frames):
        alpha = (i + 1) / (fade_frames + 1)
        last_idx = max(0, last_len - fade_frames + i)
        next_idx = min(next_len - 1, i)
        blended = cv2.addWeighted(
          last_frames[last_idx], 1 - alpha,
          next_frames[next_idx], alpha,
          0
        )
        transition.append(blended)
      return transition

    # フェード時間に対応するフレーム数を求める
    temp_cap = cv2.VideoCapture(video_paths[0])
    fps = temp_cap.get(cv2.CAP_PROP_FPS) or 30
    temp_cap.release()
    fade_frames = int(fps * fade_duration)

    sequence_frames = []
    all_frames_per_video = []

    # 各動画を読み込み（順再生＋逆再生にも対応）
    for path in video_paths:
      if not os.path.isfile(path):
        all_frames_per_video.append([])
        continue

      cap = cv2.VideoCapture(path)
      frames = read_frames(cap)
      cap.release()

      if reverse_flg:
        frames_combined = frames + frames[::-1]
      else:
        frames_combined = frames

      all_frames_per_video.append(frames_combined)

    n = len(all_frames_per_video)
    if n == 0:
      return [], fps

    # フェード付きで全動画を結合
    for i in range(n):
      frames = all_frames_per_video[i]
      if not frames:
        continue

      sequence_frames.extend(frames)

      # 次の動画（最後→最初も含める）
      next_i = (i + 1) % n
      next_frames = all_frames_per_video[next_i]
      if not next_frames:
        continue

      # フェードを生成
      fade_transition = create_fade_transition(
        frames[-fade_frames:] if len(frames) >= fade_frames else frames,
        next_frames[:fade_frames] if len(next_frames) >= fade_frames else next_frames,
        min(fade_frames, len(frames), len(next_frames))
      )

      # 通常：すべての接続にフェードを入れる（最後→最初も含む）
      sequence_frames.extend(fade_transition)

    return sequence_frames, fps

  # --------------------------
  # フレーム更新
  # --------------------------
  def _update_frame(self):
    if not self.frames:
      return
    frame = self.frames[self.current_frame]
    self.current_frame = (self.current_frame + 1) % len(self.frames)
    pil_img = Image.fromarray(frame)
    self.viewer.display_image(pil_img)

  # --------------------------
  # メッセージ送信イベント
  # --------------------------
  def on_send_message(self, message=None, speech_use=False):
    if not message:
      user_input = self.input_box.text().strip()
      if not user_input:
        return
      if self.agent_running:
        return
    else:
      user_input = message

    # 入力テキストボックスを非活性
    self.send_button.setEnabled(False)
    self.input_box.setEnabled(False)
    
    self.agent_running = True

    # === スレッドでエージェントを動作 ===
    # image = self.input_window.image if self.input_window else None
    image_path = self.input_window.image_path if self.input_window else None
    
    # self.worker_thread = AgentWorker(self.agent, user_input, image)
    self.worker_thread = AgentWorker(self.agent, user_input, image_path, self.bubble_queue_thought, speech_use)
    self.worker_thread.finished.connect(self.on_agent_finished)
    self.worker_thread.start()
    
  # --------------------------
  # エージェント応答完了時
  # --------------------------
  def on_agent_finished(self, response, action):
    print(f"💬 応答: {response}")

    # 入力テキストボックスを空に
    self.input_box.clear()

    # 入力テキストボックスを活性
    self.send_button.setEnabled(True)
    self.input_box.setEnabled(True)
    # self.viewer.agent_menu.setEnabled(True) # エージェント切り替えON
    self.input_box.setFocus()

    if response is not None:

      if LINE_BREAK:
        response = response.replace("\n\n", "\n").replace("。\n", "。").replace("。", "。\n").strip()

      # ふきだし表示
      self.bubble_queue.put(response)

      if self.agent.voice:
        self.agent.voice.create_voice(response)

    
    if action:
      if action == "<open:othello>":
        idx = self.viewer.game_names.index("オセロ")
        self.view_game(idx)
      elif action == "<close:othello>":
        if self.othello_window:
          self.othello_window.close()
      else: # action == "<othello>":
        if self.othello_window:
          # オセロゲーム画面が開いているときは画面更新
          self.othello_window.draw_board()    

    self.agent_running = False
    
    # bubble = False
    # if win_image_process and hasattr(win_image_process, "x"):
    #   x = win_image_process.x
    #   y = win_image_process.y
    #   width = win_image_process.width
    #   bubble = win_image_process.bubble

    #   if bubble:
    #     self.bubble_queue.put({"text":chara_msg, "x":x+width-40, "y":y-40, "width":width})

  # ふきだしにテキストを表示
  def _check_bubble_queue(self):
    if not self.bubble_queue.empty():
      rect = self.geometry()
      # print(f"位置とサイズ: x={rect.x()}, y={rect.y()}, w={rect.width()}, h={rect.height()}")        
      text = self.bubble_queue.get()
      if text == "": text = "･･･"
        
      x = rect.x() + rect.width() #dic["x"]
      y = rect.y() # dic["y"]
      max_width = min(int(rect.width() * 1.0), self.screen_width)
      # 表示秒(文字数に応じて変える)
      duration = min(BASE_BUBBLE_DURATION + len(text) * 130, 60000)

      # print(f"ふきだし表示秒: {duration/1000:.3f}s")
      
      # del self.bubble
      # self.bubble = None
      
      SpeechBubble(text, x, y, max_width=max_width, duration=duration)

    # 思考の吹き出し表示
    if not self.bubble_queue_thought.empty():
      rect = self.geometry()
      text = self.bubble_queue_thought.get()
      if not text: text = "･･･"
      
      margin = 7
      #  + rect.width() #dic["x"]
      y = rect.y() + rect.height() + 2
      max_width = min(int(rect.width() * 1.0), self.screen_width) - margin * 4
      x = rect.x()
      
      # 表示秒(文字数に応じて変える)
      duration = min(BASE_BUBBLE_DURATION + len(text) * 130, 60000)

      # print(f"ふきだし(思考)表示秒: {duration/1000:.3f}s")
      # del self.bubble
      # self.bubble = None
      
      SpeechBubble(text, x, y, max_width=max_width, duration=duration, tail_direction=1, tail_shape=1)
  
  # 音声入力を使用
  def speech_input(self, speech_flg):

    self.speech_use = speech_flg

    if speech_flg:
      from speech_recognizer import LlmSignals, SpeechRecognizer
      
      self.audio_enabled = True

      self.signals = LlmSignals()

      # シグナルオブジェクトをインスタンス化
      self.signals.recognized.connect(self.on_recognized_gui_thread)

      self.recognizer = SpeechRecognizer(self.signals) 
      threading.Thread(target=self.recognizer.start, daemon=True).start()
      print("🎙️ 音声入力を開始しました。")
    else:
      self.audio_enabled = False
      if self.recognizer:
        self.recognizer.stop()
        self.recognizer = None
      print("🛑 音声入力を停止しました。")

  @pyqtSlot(str) 
  def on_recognized_gui_thread(self, text):
    """音声認識結果をGUIスレッドで処理"""
    # self.textbox.append(f"👤 あなた: {text}")
    print(f"音声検知: {text}")
    if self.agent_running:
      print(f" => エージェント思考中のためスキップ")
      return
    self.input_box.setText(text)
    self.on_send_message(speech_use=True)
    
  # 入力画像ウィンドウの表示
  def view_input(self):
    if self.input_window is None:
      self.input_window = InputImageWindow()
      self.input_window.setParent(self, Qt.Window)
    self.input_window.show()
    self.input_window.raise_()
    self.input_window.activateWindow()

  # エージェントの切り替え
  def change_agent(self, idx):
    if idx != self.agent.select_idx:
      self.agent.change(idx)
      self.video_paths = self.agent.video_paths
      self.video_reverse = self.agent.video_reverse
      self.video_fade_msec = self.agent.video_fade_msec
      
      title_add = f"：{agent.name}" if agent.name else "0"
      self.setWindowTitle(f"AIエージェント{title_add}")
      
      # 入力画像ウィンドウの表示
      self.viewer.vision_flg = self.agent.vision_model
    
      self.viewer.game_names = []
      for tool in self.agent.tools.values():
        if tool.type() == "game":
          self.viewer.game_names.append(tool.title())
      
      if not "othello" in self.viewer.game_names:
        # ゲームのツールがない場合は閉じる
        self.agent.mode = "nomal"
        
        if self.othello_window is not None:
          self.othello_window.close()
          self.othello_window = None # オセロ

      # 動画の変更
      self.create_video_frame()

  # ゲームウィンドウの表示
  def view_game(self, idx):
    game_title = self.viewer.game_names[idx]
    for tool in self.agent.tools.values():
      if tool.title() == game_title:
        if self.othello_window is None:
          self.othello_window = OthelloBoardView(tool, self, ops_turn=0)
          self.othello_window.setParent(self, Qt.Window)
        self.agent.mode = tool.name()
        self.othello_window.draw_board()
        self.othello_window.show()
        self.othello_window.raise_()
        self.othello_window.activateWindow()


# yamlの読み込み
def load_all_yaml_files(yaml_dir_path):
  yaml_files = glob.glob(os.path.join(yaml_dir_path, '**', '*.y*ml'), recursive=True)
  all_data = []

  for file_path in yaml_files:
    with open(file_path, 'r', encoding='utf-8') as f:
      try:
        data = yaml.safe_load(f)
        all_data.append(data)
      except yaml.YAMLError as e:
        print(f"⚠️ YAML読み込みエラー: {file_path} - {e}")

  return all_data

# =========================
# 実行部分
# =========================
if __name__ == "__main__":

  ollama_host = 'http://192.168.1.100:11434'

  # 設定されたエージェント(yaml)を読み込み
  agent_cfgs = load_all_yaml_files("./agent_cfg")
  agent = AdvancedAgent(agent_cfgs, def_idx=0, host=ollama_host)

  # フォームを起動
  app = QApplication(sys.argv)  
  viewer = AgentViewer(agent)
  viewer.show()

  sys.exit(app.exec_())
  
# 【入力例】
# こんにちは。
# 最新のニュースを教えてください。
# 12345678901234567890 * 98765432109876543210 を計算してください。
# 円周率を教えてください。
# もっと下の桁までお願いします。
