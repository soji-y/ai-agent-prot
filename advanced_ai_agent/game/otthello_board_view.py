import sys
from PyQt5.QtWidgets import QApplication, QGraphicsRectItem, QGraphicsEllipseItem, QMenu, QAction, QGraphicsScene, QGraphicsTextItem
from PyQt5.QtGui import QBrush, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QRectF, QPointF

from resizeble_graphics_view import ResizableGraphicsView

CELL_SIZE = 60
BOARD_SIZE = 8

class OthelloBoardView(ResizableGraphicsView):
  def __init__(self, game_logic, parent, ops_turn=0):
    super().__init__(parent)
    self.setWindowTitle(f"オセロ")
        
    self.parent = parent
    self.game_logic = game_logic
    self.ops_turn = ops_turn  # 0:先手のみ, 1:後手のみ, 2:両方可
    self.last_hover_pos = None

    # parent.agent.mode == "othello"
    # self.last_move = None
    self.cells = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        
    # 明るい黄緑色の背景
    self.setBackgroundBrush(QBrush(QColor(120, 179, 113)))  # ←変更（#3CB371）

    # シーン初期化
    self.scene().setSceneRect(0, 0, CELL_SIZE * BOARD_SIZE, CELL_SIZE * BOARD_SIZE)
    self.draw_board()
    
    # 初期化時に一度だけ作る
    # self.hover_rect = QGraphicsRectItem()
    # self.hover_rect.setPen(QPen(Qt.red, 2))
    # self.scene().addItem(self.hover_rect)
    # self.hover_rect.hide()

    # マウス追跡有効化
    self.setMouseTracking(True)    
    # self.draw_board()
    
  # フォームを閉じたときのイベント
  def closeEvent(self, event):
    # モードを戻す
    self.parent.agent.mode == "nomal"
    self.game_logic.boad_init()
    super().closeEvent(event)
    
  # --- 盤面描画 ---
  def draw_board(self):
    self.scene().clear()
    size = CELL_SIZE * BOARD_SIZE

    # 背景
    # board_color = QColor(0, 100, 0)
    # self.scene().setBackgroundBrush(board_color)
    board_rect = QGraphicsRectItem(0, 0, size, size)
    board_rect.setBrush(QBrush(QColor(0, 120, 0)))  # 緑っぽい
    board_rect.setPen(QPen(Qt.NoPen))
    self.scene().addItem(board_rect)

    if hasattr(self, "_last_hover_pos"):
      x, y = self._last_hover_pos
      if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
        self.hover_rect.setRect(QRectF(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        self.hover_rect.show()
      
    # 枠線とセルを描画
    pen = QPen(Qt.black, 2)
    for x in range(BOARD_SIZE):
      for y in range(BOARD_SIZE):
        rect = QRectF(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        self.scene().addRect(rect, pen)

    # 最終手ハイライト
    if self.game_logic.last_move:
      x, y = self.game_logic.last_move
      highlight = QGraphicsRectItem(
        QRectF(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
      )
      highlight.setBrush(QBrush(QColor(255, 0, 0, 80)))
      # highlight.setPen(Qt.NoPen)
      highlight.setPen(QPen(Qt.NoPen))

      self.scene().addItem(highlight)
      
    # 石を描画
    for x in range(BOARD_SIZE):
      for y in range(BOARD_SIZE):
        cell = self.game_logic.board[x][y]
        if cell == "X":
          color = Qt.black
        elif cell == "O":
          color = Qt.white
        else:
          continue

        ellipse = QGraphicsEllipseItem(
          QRectF(y * CELL_SIZE + 5, x * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10)
        )
        ellipse.setBrush(QBrush(color))
        ellipse.setPen(QPen(Qt.black, 1))
        self.scene().addItem(ellipse)
        self.cells[x][y] = ellipse
        
    # --- マス番号を描画（石が置かれていない場所のみ） ---
    font = QFont("Arial", int(CELL_SIZE/3), QFont.Bold)  # ← 大きめのフォント
    text_color = QColor(255, 255, 255, 50)  # 半透明の白

    for x in range(BOARD_SIZE):
      for y in range(BOARD_SIZE):
        if self.game_logic.board[x][y] == ".":  # 空きマスのみ表示
          # マス番号を決定
          col_letter = chr(ord("A") + y)
          row_number = str(x + 1)
          text = f"{col_letter}{row_number}"

          text_item = QGraphicsTextItem(text)
          text_item.setFont(font)
          text_item.setDefaultTextColor(text_color)  # 半透明の白

          # 位置を調整（マスの中央に表示）
          text_rect = text_item.boundingRect()
          cx = y * CELL_SIZE + (CELL_SIZE - text_rect.width()) / 2
          cy = x * CELL_SIZE + (CELL_SIZE - text_rect.height()) / 2
          text_item.setPos(cx, cy)

          self.scene().addItem(text_item)

    # 操作権限を確認
    if self.ops_turn == 0 and self.game_logic.turn == 'X' or \
       self.ops_turn == 1 and self.game_logic.turn == 'O':
    
      # 合法手の表示
      for (x, y) in self.game_logic.get_legal_moves():
        mark = QGraphicsEllipseItem(
          QRectF(y * CELL_SIZE + int(CELL_SIZE / 3), x * CELL_SIZE + int(CELL_SIZE / 3), int(CELL_SIZE / 3), int(CELL_SIZE / 3))
        )
        mark.setBrush(QBrush(QColor(0, 0, 255, 100)))
        # mark.setPen(Qt.NoPen)
        mark.setPen(QPen(Qt.NoPen))

        self.scene().addItem(mark)

    self.hover_rect = QGraphicsRectItem()
    self.hover_rect.setPen(QPen(Qt.red, 2))
    self.scene().addItem(self.hover_rect)
    # self.hover_rect.hide()

    # --- hover_rect を復活させる ---
    if self.last_hover_pos is not None:
      x, y = self.last_hover_pos
      if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
        self.hover_rect.setRect(QRectF(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        self.hover_rect.show()
    
  # --- マウス移動時（赤枠ハイライト） ---
  def mouseMoveEvent(self, event):
    pos = self.mapToScene(event.pos())
    x, y = int(pos.y() // CELL_SIZE), int(pos.x() // CELL_SIZE)
    self.last_hover_pos = (x, y)  # ここで位置を保存！

    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
      # 既存のhover_rectを再利用
      self.hover_rect.setRect(
        QRectF(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
      )
      self.hover_rect.show()
    else:
      # 盤外に出たら非表示
      self.hover_rect.hide()

    super().mouseMoveEvent(event)
    
  # --- クリック時処理 ---
  def mousePressEvent(self, event):
    if event.button() == Qt.LeftButton:
      pos = self.mapToScene(event.pos())
      x, y = int(pos.y() // CELL_SIZE), int(pos.x() // CELL_SIZE)

      # 範囲外は無視
      if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return

      # 操作権限を確認
      if self.ops_turn == 0 and self.game_logic.turn != 'X':
        return
      if self.ops_turn == 1 and self.game_logic.turn != 'O':
        return

      # 合法手チェック
      if (x, y) not in self.game_logic.get_legal_moves():
        return

      # 一手打つ
      # self.game_logic.place_stone(x, y)
      self.game_logic.apply_move(x, y)
      # self.last_move = (x, y)
      self.draw_board()
      
      move_str = self.game_logic.move_to_str(x, y)
      if self.parent:
        self.parent.on_send_message(message=f"<game><{move_str}>")
      # 赤枠を非表示
      # self.hover_rect.hide()

    super().mousePressEvent(event)

  # 右クリックメニューの表示
  def show_context_menu(self, pos):
    menu = QMenu(self)

    # --- 「オセロ初期化」項目を追加 ---
    reset_action = QAction("オセロ初期化", self)
    reset_action.triggered.connect(self.reset_othello_board)
    menu.addAction(reset_action)

    # メニューを表示
    menu.exec_(self.mapToGlobal(pos))


  def reset_othello_board(self):
    """オセロ盤を初期化して再描画"""
    # ロジッククラスを持っている前提（例：self.logic）
    if self.game_logic:
      self.game_logic.boad_init()  # ←あなたのロジッククラスの初期化メソッドを呼ぶ

      # 盤面を再描画
      # self.last_move = None
      self.draw_board()
    
# --- デモ用 ---
class DummyOthello:
  def __init__(self):
    self.boad_init()

  def boad_init(self):
    self.board = [['.' for _ in range(8)] for _ in range(8)]
    self.board[3][3] = 'O'
    self.board[3][4] = 'X'
    self.board[4][3] = 'X'
    self.board[4][4] = 'O'
    self.turn = 'X'

  def inside(self, x, y):
    return 0 <= x < 8 and 0 <= y < 8

  def get_legal_moves(self, player=None):
    if not player:
      player = self.turn
    opponent = 'O' if player == 'X' else 'X'
    moves = set()
    DIRECTIONS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    for x in range(8):
      for y in range(8):
        if self.board[x][y] != '.':
          continue
        for dx, dy in DIRECTIONS:
          nx, ny = x + dx, y + dy
          if not self.inside(nx, ny) or self.board[nx][ny] != opponent:
            continue
          while self.inside(nx, ny) and self.board[nx][ny] == opponent:
            nx += dx
            ny += dy
          if self.inside(nx, ny) and self.board[nx][ny] == player:
            moves.add((x, y))
            break
    return list(moves)

  def place_stone(self, x, y):
    self.board[x][y] = self.turn
    self.turn = 'O' if self.turn == 'X' else 'X'

  # 現在の盤面をテキスト形式で返す
  def to_text(self):
    rows = ["  A B C D E F G H"]
    for i, row in enumerate(self.board):
      line = f"{i+1} "
      for cell in row:
        line += {'X': '●', 'O': '○'}.get(cell, '.') + ' '
      rows.append(line)
    return '\n'.join(rows)

  # 座標 (x, y) を "A1" 形式の文字列に変換する
  def move_to_str(self, x, y):
    return f"{chr(y + 65)}{x + 1}"

  # "A1" 形式の文字列を座標 (x, y) に変換する
  def str_to_move(self, move_str):
    return (int(move_str[1]) - 1, ord(move_str[0].upper()) - 65)


if __name__ == '__main__':
  app = QApplication(sys.argv)
  game = DummyOthello()
  view = OthelloBoardView(game, None, ops_turn=2)
  view.resize(700, 700)
  view.show()
  sys.exit(app.exec_())
