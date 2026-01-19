from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QBrush, QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QMenu, QFrame

from PIL import Image

# === 画像ビューワークラス ===
class ResizableGraphicsView(QGraphicsView):
  wheelScrolled = pyqtSignal(int)  # 上下のホイール回転を通知するシグナル(+1:上, -1:下)
  middleButton = pyqtSignal() # 中央ボタン
  leftButton = pyqtSignal() # 左クリック

  def __init__(self, parent=None): 
    super().__init__(parent)
        
    # 表示画像
    self.image_item = None
    
    # 枠線を非表示
    self.setFrameShape(QFrame.NoFrame)    

    self.setBackgroundBrush(QBrush(Qt.black))
    self.setScene(QGraphicsScene(self))
    self.setAlignment(Qt.AlignCenter)
    
  # --- キー押下イベント ---
  def keyPressEvent(self, event):  
    super().keyPressEvent(event)  # 他のキーは通常の処理へ
      
  # マウスボタンが押されたときのイベントをオーバーライド
  def mousePressEvent(self, event):
    # クリックの判定
    if event.button() == Qt.RightButton:
      self.show_context_menu(event.pos())
    elif event.button() == Qt.LeftButton:
      self.leftButton.emit()             
    elif event.button() == Qt.MiddleButton:
      self.middleButton.emit()      
    else:
      super().mousePressEvent(event)

  # 右クリックメニューの表示
  def show_context_menu(self, pos):
    menu = QMenu(self)

    # メニューを表示
    if menu.actions():
      menu.exec_(self.mapToGlobal(pos))
    
  # マウスホイールイベント
  def wheelEvent(self, event):
    delta = event.angleDelta().y()
    if delta > 0:
      self.wheelScrolled.emit(1)   # ホイール上
    elif delta < 0:
      self.wheelScrolled.emit(-1)  # ホイール下
    # イベントを親に伝播させたい場合は super() 呼ぶ
    super().wheelEvent(event)

  # リサイズイベント
  def resizeEvent(self, event):
    super().resizeEvent(event)
    self.fit_image_to_view()

  # --- ドラッグ&ドロップ対応 ---
  def dragEnterEvent(self, event: QDragEnterEvent):
    event.ignore()

  def dragMoveEvent(self, event):
    event.ignore()

  def dropEvent(self, event: QDropEvent):
    event.ignore()

  # 画面表示
  def display_image(self, pil_image:Image):
    self.scene().clear()
    self.resetTransform()

    # --- PIL → QImage 変換を直接行う ---
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    data = pil_image.tobytes("raw", "RGB")
    qimg = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGB888)

    pixmap = QPixmap.fromImage(qimg)

    self.image_item = self.scene().addPixmap(pixmap)
    self.scene().setSceneRect(self.image_item.boundingRect())

    self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    self.fit_image_to_view()

  # ディスプレイをクリア
  def display_clear(self):
    self.scene().clear()
    self.image_item = None
      
  # 画像をフィットさせて表示
  def fit_image_to_view(self):
    if self.image_item:
      self.fitInView(self.image_item, Qt.KeepAspectRatio)
