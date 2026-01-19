from PyQt5 import QtCore, QtGui, QtWidgets

# ==== ふきだし設定 ====
FONT_FAMILY = "Meiryo"           # 使用フォント
FONT_SCALE = 0.02                # フォントサイズ（キャンバス幅に対する割合）
BUBBLE_TEXT_MARGIN = 5           # 吹き出し内の文字との余白
BUBBLE_RADIUS = 10               # 吹き出しの角の丸み
DURATION = 7000                  # フェードアウト時間（ミリ秒）

class SpeechBubble(QtWidgets.QWidget):
  
  _instances = []  # クラス変数で参照を保持
  
  def __init__(self, text, x=0, y=0, width=None, max_width=200, duration=None, tail_direction=0, tail_shape=0):
    super().__init__()

    self.text = text
    self.x = x
    self.y = y
    self.tail_direction = tail_direction
    self.tail_shape = tail_shape
    
    # 色
    if tail_shape == 0:
      font_color = "black"
      self.back_color = QtGui.QColor(255, 255, 255)
    else:
      font_color = "white"
      self.back_color = QtGui.QColor(0, 0, 0)
    
    # self.parent_window = parent_window
    self.duration = duration if duration else DURATION
    
    self.tail_height = 8
    self.opacity = 1.0

    self.setWindowFlags(
      QtCore.Qt.FramelessWindowHint |
      QtCore.Qt.WindowStaysOnTopHint | # 最前面
      QtCore.Qt.Tool
    )
    self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
    
    # 最小＆最大幅を制限した自動幅計算
    margin = 7
    max_width = max(max_width, 100)
    font_size = min(int(max_width * FONT_SCALE) + 7, 24)
    font = QtGui.QFont(FONT_FAMILY, font_size)       
    metrics = QtGui.QFontMetrics(QtGui.QFont(FONT_FAMILY, int(font_size*0.8)))
    

    # 文字サイズから矩形を取得（必要に応じて折り返す）
    text_rect = metrics.boundingRect(0, 0, max_width + margin, 1000, QtCore.Qt.TextWordWrap, self.text)
    if width:
      self.bubble_width = width
    else:
      self.bubble_width = min(text_rect.width(), max_width) + margin *4
    self.bubble_height = text_rect.height() + margin *2
    
    self.resize(self.bubble_width, self.bubble_height + self.tail_height)


    if tail_direction == 0:
      self.x_pos = 0
      self.mal = -1
    else:
      self.x_pos = max(0, self.bubble_width - 50)
      self.mal = 1 # 吹き出しの尻尾をどちらに出すか(mal=-1:左, 1:右)

    self.label = QtWidgets.QLabel(self.text, self)
    self.label.setFont(font)
    self.label.setWordWrap(True)
    self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    self.label.setStyleSheet(f"color: {font_color}; font-size: {font_size}px; background: transparent;")
    self.label.setGeometry(margin, margin, self.bubble_width - margin*3, self.bubble_height - margin*2)

    # 閉じるボタンの追加
    self.close_btn = QtWidgets.QPushButton("×", self)
    self.close_btn.setFixedSize(16, 16)
    self.close_btn.clicked.connect(self.close)
    self.close_btn.setStyleSheet(f"""
        QPushButton {{
            background-color: transparent;
            border: none;
            font-size: 12px;
            color: {font_color};
        }}
        QPushButton:hover {{
            color: red;
        }}
    """)

    # 右上ぎりぎりに配置
    self.close_btn.move(self.width() - self.close_btn.width() - 2, 2)

    self.close_btn.clicked.connect(self.close)

    self.opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
    self.setGraphicsEffect(self.opacity_effect)
    self.opacity_effect.setOpacity(self.opacity)

    self.timer = QtCore.QTimer(self)
    self.timer.timeout.connect(self.fade_out)
    self.timer.setSingleShot(True)
    self.timer.start(self.duration)

    self.fade_timer = QtCore.QTimer(self)
    self.fade_timer.timeout.connect(self.perform_fade)

    self.old_pos = None
    self.adjust_position()
    
    # 表示位置など設定後に自動表示
    self.show()

    # 自身の参照をクラス変数に保持
    SpeechBubble._instances.append(self)

    # 閉じられたときにリストから除外
    self.destroyed.connect(lambda: SpeechBubble._instances.remove(self))
    

  def adjust_position(self):
    screen_geometry = QtWidgets.QApplication.primaryScreen().availableGeometry()
   
    x = max(0, min(self.x, screen_geometry.width() - self.width()))
    y = max(0, min(self.y, screen_geometry.height() - self.height()))
    self.move(x, y)
    
    if self.tail_direction:
      if self.x - x > self.width()/2:
        self.mal = 1
    else:
      if self.x - x < self.width()/2:
        self.mal = -1
        
  def paintEvent(self, event):
    painter = QtGui.QPainter(self)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    
    painter.setBrush(QtGui.QBrush(self.back_color))
    
    painter.setPen(QtCore.Qt.NoPen)

    path = QtGui.QPainterPath()
    radius = BUBBLE_RADIUS
    rect = QtCore.QRectF(0, 0, self.bubble_width, self.bubble_height)
    path.addRoundedRect(rect, radius, radius)

    # 尻尾（左に太くカーブ）
    cx = 20 # 40  # 尻尾の根元のx位置（やや右へ）
    cy = self.bubble_height
    tail_path = QtGui.QPainterPath()
    tail_path.moveTo(cx + self.x_pos, cy)

    h = self.tail_height

    if self.tail_shape == 0:
      # 尻尾の曲線（より滑らかなパラメータに調整）
      # 右側の線
      tail_path.cubicTo(
          cx + 0.8 * h * self.mal + self.x_pos, cy + 0.5 * h,     # やや下へ
          cx + 1.5 * h * self.mal + self.x_pos, cy + 1.2 * h,     # なだらかに左下
          cx + 2.4 * h * self.mal + self.x_pos, cy + h            # 尻尾の先端
      )
      # 左側の線
      tail_path.cubicTo(
          cx + 1.8 * h * self.mal + self.x_pos, cy + 0.7 * h,     # やや浅く戻る
          cx + 1.6 * h * self.mal + self.x_pos, cy + 0.3 * h,     # なめらかに上昇
          cx + 1.5 * h * self.mal + self.x_pos, cy                # 吹き出し本体へ戻る
      )
    # else:
    #   # --- self.tail_shape == 1 の修正 (画像のような小さな尖った尻尾) ---
      
    #   # --- self.tail_shape == 1 の再修正 (画像に忠実な、小さく鋭い尻尾) ---
      
    #   # 根元のX位置をさらに左端に寄せ、丸角の内側ギリギリに設定
    #   cx = 3
    #   # 尻尾の高さhを非常に小さく使う (例: hの20%)
    #   small_h = 0.2 * h 
      
    #   # 根元の開始点: 吹き出し本体の左下角に近い位置からスタート
    #   tail_path.moveTo(cx + self.x_pos, cy) 

    #   # 1. 右側の線（左下へ短く伸びる）
    #   tail_path.cubicTo(
    #       # 制御点1: 浅いカーブ
    #       cx + 0.05 * small_h * self.mal + self.x_pos, cy + 0.1 * small_h, 
    #       # 制御点2: 尖らせるために先端に近づける
    #       cx + 0.2 * small_h * self.mal + self.x_pos, cy + 0.8 * small_h, 
    #       # 先端の座標: 左下へごくわずか
    #       cx + 0.3 * small_h * self.mal + self.x_pos, cy + small_h 
    #   )
      
    #   # 2. 左側の線（吹き出し本体へ戻る）
    #   tail_path.cubicTo(
    #       # 制御点3: 鋭さを維持
    #       cx + 0.2 * small_h * self.mal + self.x_pos, cy + 0.8 * small_h, 
    #       # 制御点4: 根元に戻る際のカーブを小さく
    #       cx + 0.05 * small_h * self.mal + self.x_pos, cy + 0.1 * small_h, 
    #       # 吹き出し本体へ戻る点: 根元に戻る
    #       cx + 0.4 * small_h * self.mal + self.x_pos, cy # 尻尾の幅を調整
    #   )
      # NOTE: self.tail_shape == 1 の場合も path.addPath(tail_path) が必要なので
      # if/else の外側にある path.addPath(tail_path) は残しておく
      
      # # --- 丸い独立した尻尾（吹き出し下に接近） ---
      # circle_radius = 6
      
      # # 吹き出し本体のすぐ下に配置
      # offset_y = 1  # 前より小さくして距離を縮める
      # circle_center_x = cx + self.mal  # 尻尾の横位置を吹き出し根元に近づける
      # circle_center_y = cy + offset_y + circle_radius / 2  # 下端に近づける

      # painter.setBrush(QtGui.QBrush(self.back_color))
      # painter.drawEllipse(QtCore.QPointF(circle_center_x, circle_center_y),
      #                     circle_radius, circle_radius)
      
    path.addPath(tail_path)
    painter.drawPath(path)
      
    # ウィジェットのマスクを吹き出しの形に合わせる
    region = QtGui.QRegion(path.toFillPolygon().toPolygon())
    self.setMask(region)


  def mousePressEvent(self, event):
    if event.button() == QtCore.Qt.LeftButton:
      self.old_pos = event.globalPos()

  def mouseMoveEvent(self, event):
    if self.old_pos:
      delta = event.globalPos() - self.old_pos
      self.move(self.pos() + delta)
      self.old_pos = event.globalPos()

  def mouseReleaseEvent(self, event):
    self.old_pos = None

  def fade_out(self):
    self.opacity_effect.setOpacity(self.opacity)
    self.fade_timer.start(50)

  def perform_fade(self):
    self.opacity -= 0.05
    if self.opacity <= 0:
      self.fade_timer.stop()
      self.close()
    else:
      self.opacity_effect.setOpacity(self.opacity)
      self.repaint()
      
      
