import time
import textwrap

# ==== ふきだし設定 ====
FONT_FAMILY = "Meiryo"           # 使用フォント
FONT_SCALE = 0.020               # フォントサイズ（キャンバス幅に対する割合）
BUBBLE_TEXT_MARGIN = 5           # 吹き出し内の文字との余白
BUBBLE_RADIUS = 15               # 吹き出しの角の丸み
DURATION = 7000                  # フェードアウト時間（ミリ秒）

# ==== キャンバスに対する吹き出しの表示位置マージン ====
BUBBLE_MARGIN_X = 5  # キャンバス左・右のマージン
BUBBLE_MARGIN_Y = 5  # キャンバス上のマージン

# ==== 吹き出し見た目 ====
BUBBLE_BORDER_WIDTH = 1          # 枠線の太さ
BUBBLE_FILL_COLOR = "#F0FAC4"    # 吹き出しの背景色
BUBBLE_BORDER_COLOR = "#414613"  # 吹き出しの枠線色
TEXT_COLOR = "#49480D"           # テキストの色（初期）
BUBBLE_FILL_COLOR_ACT = "#C4BEFC"    # 吹き出しの背景色
BUBBLE_BORDER_COLOR_ACT = "#2D1764"  # 吹き出しの枠線色
TEXT_COLOR_ACT = "#160733"           # テキストの色（初期）

# 吹き出し表示
class RoundedText:
  def __init__(self, canvas, text, bb_pos=0, bb_auto_f=0, color_type=0, on_complete=None, duration=None):
    self.canvas = canvas
    self.text_raw = text
    self.color_type = color_type
    self.on_complete = on_complete
    if duration:
      self.duration = duration
    else:
      self.duration = DURATION
    
    self.items = []

    # フォントと描画準備
    canvas_width = self.canvas.winfo_width()
    canvas_height = self.canvas.winfo_height()
    
    if bb_auto_f:
      self.font_size = max(int(self.canvas.winfo_height() * FONT_SCALE), 8)
    else:
      self.font_size = max(int(canvas_height * FONT_SCALE), 8)
    self.font = (FONT_FAMILY, self.font_size)
    self.max_width = canvas_width - 2 * BUBBLE_MARGIN_X

    
    # wrap + サイズ取得
    self.text_wrapped = self._wrap_text()

        # ふきだしテキストのサイズ計算
    self._get_text_w_h()
        
    # 吹き出し表示位置を変更
    self.x = BUBBLE_MARGIN_X
    if bb_pos == 0:
      self.y = BUBBLE_MARGIN_Y * canvas_height / 256
    else:
      self.y = canvas_height - self.height - BUBBLE_MARGIN_Y * canvas_height / 256
    
    self._create_items()
    self._center()
    self._start_fade()

  def _wrap_text(self):
      avg_char_width = self.font_size * 0.6
      max_chars = max(int((self.max_width - 2 * BUBBLE_TEXT_MARGIN) / avg_char_width), 1)
      return textwrap.fill(self.text_raw, width=max_chars)

  def _get_text_w_h(self):
    tmp = self.canvas.create_text(0, 0, text=self.text_wrapped, font=self.font, anchor='nw', width=self.max_width - 2*BUBBLE_TEXT_MARGIN)
    bbox = self.canvas.bbox(tmp)
    self.canvas.delete(tmp)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    self.width =  text_w + 2 * BUBBLE_TEXT_MARGIN
    self.height = text_h + 2 * BUBBLE_TEXT_MARGIN

    # self.x2 = self.x1 + text_w + 2 * BUBBLE_TEXT_MARGIN
    # self.y2 = self.y1 + text_h + 2 * BUBBLE_TEXT_MARGIN
    # self.width = self.x2 - self.x1
    # self.height = self.y2 - self.y1
    
  # 16進数のカラーコードをRGBのタプルに変換
  def hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

  def _create_items(self):
    x1, y1 = self.x, self.y
    x2 = x1 + self.width
    y2 = y1 + self.height
    if self.color_type == 0:
      outline_color = BUBBLE_BORDER_COLOR
      fill_color = BUBBLE_FILL_COLOR
      text_color = TEXT_COLOR
    else:
      outline_color = BUBBLE_BORDER_COLOR_ACT
      fill_color = BUBBLE_FILL_COLOR_ACT
      text_color = TEXT_COLOR_ACT
      
    self.fill_color_rgb = self.hex_to_rgb(fill_color)
      
    rect = self.canvas.create_rectangle(
        x1, y1, x2, y2,
        outline=outline_color,
        width=BUBBLE_BORDER_WIDTH,
        fill=fill_color
    )
          
    text_item = self.canvas.create_text(
        x1 + BUBBLE_TEXT_MARGIN,
        y1 + BUBBLE_TEXT_MARGIN,
        text=self.text_wrapped,
        font=self.font,
        anchor="nw",
        fill=text_color,
        width=self.max_width - 2 * BUBBLE_TEXT_MARGIN
    )

    self.items = [rect, text_item]

  def _center(self):
    current_x = self.x
    canvas_center_x = self.canvas.winfo_width() // 2
    new_x = canvas_center_x - self.width // 2
    dx = new_x - current_x
    if dx != 0:
      for item in self.items:
        self.canvas.move(item, dx, 0)
      self.x = new_x

  def _set_alpha(self, alpha):
    alpha = max(0, min(1, alpha))
    # bg_r = int(255 * alpha + 255 * (1 - alpha))
    # bg_g = int(255 * alpha + 204 * (1 - alpha))
    # bg_b = int(204 * alpha + 204 * (1 - alpha))
    bg_r = int(self.fill_color_rgb[0] * alpha + 255 * (1 - alpha))
    bg_g = int(self.fill_color_rgb[1] * alpha + 204 * (1 - alpha))
    bg_b = int(self.fill_color_rgb[2] * alpha + 204 * (1 - alpha))
    
    bg_color = f"#{bg_r:02x}{bg_g:02x}{bg_b:02x}"

    gray_level = int(0 + 150 * (1 - alpha))
    fg_color = f"#{gray_level:02x}{gray_level:02x}{gray_level:02x}"

    self.canvas.itemconfig(self.items[0], fill=bg_color)
    self.canvas.itemconfig(self.items[1], fill=fg_color)
    # print(alpha)

  def _start_fade(self):
    start = time.time()
    duration = self.duration / 1000

    def fade_step():
      elapsed = time.time() - start
      progress = elapsed / duration
      if progress >= 1.0:
        self._set_alpha(0)
        self._delete()
        if self.on_complete:
          self.on_complete()
        return
      self._set_alpha(1 - progress)
      self.canvas.after(30, fade_step)

    fade_step()

  def _delete(self):
    for item in self.items:
      self.canvas.delete(item)
