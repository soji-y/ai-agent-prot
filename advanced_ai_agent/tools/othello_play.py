# othello_llm_game.py
# オセロのプレイロジックとLLM対話を統合したスクリプト

import re
import ollama
from copy import deepcopy

#==================== 定数設定 ====================
MODEL_NAME_1 = "gemma3:4b" # LLM先手 (X) に使うモデル
MODEL_NAME_2 = "gemma3:4b" # LLM後手 (O) に使うモデル

PLAY_MODE = 1              # 0: 人vsLLM, 1: LLMvsLLM, 2: 人vs人
HUMAN_FIRST = 0            # 0: 人間が先手(X), 1: LLMが先手(X)
GAME_MAX = 100             # 連続対局の回数
LLM_RETRY_LIMIT = 5        # ★修正点: LLMへの最大再試行回数を追加

#==================== オセロ本体クラス ====================
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

# 指定されたテキストから、開始タグと終了タグに囲まれた範囲（タグ自体も含む）を全て削除
# def remove_tagged_sections(text: str, start_tag: str, end_tag: str) -> str:
#   """
#   指定されたテキストから、開始タグと終了タグに囲まれた範囲（タグ自体も含む）を全て削除します。

#   Args:
#       text (str): 処理対象の文字列。
#       start_tag (str): 開始タグの文字列（例: "<query>"）。
#       end_tag (str): 終了タグの文字列（例: "</query>"）。

#   Returns:
#       str: 指定されたタグ範囲が削除された文字列。
#   """
#   # 正規表現の特殊文字をエスケープ
#   escaped_start_tag = re.escape(start_tag)
#   escaped_end_tag = re.escape(end_tag)

#   # 正規表現のパターンを動的に構築
#   # 例: r"<query>.*?</query>"
#   # re.DOTALL (re.S) は、'.' が改行文字にもマッチするようにします
#   # re.M (re.MULTILINE) は、^ と $ が各行の先頭と末尾にマッチするようにしますが、
#   # このケースでは不要です。
#   # re.DOTALL は、タグ間に改行が含まれてもマッチするようにするために重要です。
#   pattern = re.compile(f"{escaped_start_tag}.*?{escaped_end_tag}", re.DOTALL)

#   # マッチした全ての部分を空文字列に置換して削除
#   cleaned_text = pattern.sub("", text).strip()
#   return cleaned_text

TOOL_NAME = "othello"
TOOL_ALIAS = {"othello": "othello", "reversi": "othello"}
TOOL_USING = """
othello:
- Description: Plays Othello.
- Input: Enter <Othello-Start> to start the game. While playing Othello, select a legal move from <A1> to <H8>. To end the game, enter <Othello-End>.
- Output: The processing result.
"""
# Enter the start of Othello <Othello-Start>, Othello moves <A1> to <H8>, the end of Othello <Othello-End>. If you have not started Othello, enter <Othello-Start>.
TOOL_TITLE = "オセロ"
TOOL_TYPE = "game"

# オセロの盤面とルール処理を管理するクラス
class Othello:
  @staticmethod
  def name():
    return TOOL_NAME
  
  @staticmethod
  def alias():
    return TOOL_ALIAS

  @staticmethod
  def using():
    return TOOL_USING

  @staticmethod
  def title():
    return TOOL_TITLE

  @staticmethod
  def type():
    return TOOL_TYPE
  
  # 盤面を初期化する
  def __init__(self):
    self.boad_init()
    
  # ボードの初期化    
  def boad_init(self):
    self.board = [['.' for _ in range(8)] for _ in range(8)]
    self.board[3][3] = "O"
    self.board[3][4] = "X"
    self.board[4][3] = "X"
    self.board[4][4] = "O"
    self.turn = "X"  # X: 黒(先手), O: 白(後手)
    
    self.last_move = None

  # 指定された座標が盤面内にあるか判定する
  def inside(self, x, y):
    return 0 <= x < 8 and 0 <= y < 8

  # 指定されたプレイヤーの合法手を全てリストで返す
  def get_legal_moves(self, player=None):
    if not player:
      player = self.turn
    opponent = "O" if player == "X" else "X"
    moves = set()
    for x in range(8):
      for y in range(8):
        if self.board[x][y] != '.':
          continue
        for dx, dy in DIRECTIONS:
          nx, ny = x + dx, y + dy
          if not self.inside(nx, ny) or self.board[nx][ny] != opponent:
            continue
          # 少なくとも1個の相手石を確認
          while self.inside(nx, ny) and self.board[nx][ny] == opponent:
            nx += dx
            ny += dy
          if self.inside(nx, ny) and self.board[nx][ny] == player:
            moves.add((x, y))
            break  # 1方向でも合法なら十分
    return list(moves)

  # 指定されたプレイヤーの合法手を "A1" 形式の文字列で全てリストで返す
  def get_legal_str_moves(self, player=None):
    legal_moves = self.get_legal_moves()
    legal_str_moves = [self.move_to_str(x, y) for (x, y) in legal_moves]
    return legal_str_moves

  # 指定された座標に石を置き、相手の石を裏返す。手番を変更。returnは次の合法手
  def apply_move(self, x, y, player=None):
    if not player:
      player = self.turn
    else:
      if player != self.turn:
        return self.get_legal_moves()
      
    opponent = "O" if player == "X" else "X"
    self.board[x][y] = player
    for dx, dy in DIRECTIONS:
      nx, ny = x + dx, y + dy
      path = []
      while self.inside(nx, ny) and self.board[nx][ny] == opponent:
        path.append((nx, ny))
        nx += dx
        ny += dy
      if self.inside(nx, ny) and self.board[nx][ny] == player:
        for px, py in path:
          self.board[px][py] = player

    self.turn = "O" if self.turn == "X" else "X"
    moves = self.get_legal_moves()
    if not moves:
      # パス
      self.turn = "O" if self.turn == "X" else "X"
      moves = self.get_legal_moves()
    
    self.last_move = (x, y)

    return moves
    
  # 現在の盤面をコンソールに出力する
  def print_board(self):
    print("  A B C D E F G H")
    for i, row in enumerate(self.board):
      line = f"{i+1} "
      for cell in row:
        line += {"X": '●', "O": '○'}.get(cell, '.') + ' '
      print(line)

  # 現在の盤面をテキスト形式で返す
  def to_text(self):
    rows = ["  A B C D E F G H"]
    for i, row in enumerate(self.board):
      line = f"{i+1} "
      for cell in row:
        line += {"X": '●', "O": '○'}.get(cell, '.') + ' '
      rows.append(line)
    return '\n'.join(rows)

  # 座標 (x, y) を "A1" 形式の文字列に変換する
  def move_to_str(self, x, y):
    return f"{chr(y + 65)}{x + 1}"

  # "A1" 形式の文字列を座標 (x, y) に変換する
  def str_to_move(self, move_str):
    return (int(move_str[1]) - 1, ord(move_str[0].upper()) - 65)

  # 黒(X)と白(O)の石の数を数える
  def count_stones(self):
    x_count = sum(row.count("X") for row in self.board)
    o_count = sum(row.count("O") for row in self.board)
    return x_count, o_count

  # オセロの手を打つ (num=0:黒,1:白)
  def __call__(self, input, num=0, agent=None):
    if "End" in input:
      return "<mode:nomal> オセロを終了しました。"
      
    response = ""
    if "Start" in input:
      self.boad_init()
      response = "<mode:othello> オセロを開始しました。"

    else:
      move = input #.replace("<","").replace(">","")
      if move.startswith("<game>"):
        skip_flg = True
        move = move[len("<game>"):].strip()
      else:
        skip_flg = False

      if "<" in move:
        move = move.split("<", 1)[1]
      if ">" in move:
        move = move.split(">", 1)[0]

      if skip_flg:
        response = f"<True> オセロの合法手<{move}>を打ちました。"
      else:
        try:
          x, y = self.str_to_move(move)
          legal_moves = self.get_legal_moves()
          
          if (x, y) in legal_moves:
            # 石を置く
            turn = "X" if num == 0 else "O"
            bef_turn = turn
            legal_moves = self.apply_move(x, y, turn)
              
            if legal_moves:
              response = f"<True> オセロの合法手<{move}>を打ちました。"
              if self.turn == bef_turn:
                response += "あなたの手はパスされました。"

            else:
              b_cnt, w_cnt = self.count_stones()
              if b_cnt > w_cnt:
                win_str = "黒(先手)の勝ちです。"
              elif w_cnt > b_cnt:
                win_str = "白(後手)の勝ちです。"
              else:
                win_str = "引き分けです。"
              response = f"<True> 合法手がないためオセロを終了します。\n{win_str} [黒:{b_cnt}, 白:{w_cnt}]"
            
          else:
            response = "<False> 合法手を打てませんでした。"
        except Exception as e:
          return "<False> 合法手を打てませんでした。"
            
    legal_str_moves = self.get_legal_str_moves()
    if legal_str_moves:
      response += F"[次の合法手: {', '.join(legal_str_moves)}]"
    else:
      response += F"[次の合法手: なし]"

    return response


#==================== LLM インターフェース ====================
# 指定モデルにプロンプトを送信し、応答を取得する
def query_llm(prompt, model_name):
  response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
  return response['message']['content']

#==================== プロンプト生成 ====================
# 盤面と合法手からLLMへのプロンプトを生成する
def format_prompt(board_text, legal_moves, player):
  moves_str = ', '.join([chr(y + 65) + str(x + 1) for x, y in legal_moves])
  return f"""
あなたはオセロをプレイ中です。
現在の盤面は以下の通りです：

{board_text}

あなたは {'黒(X)' if player == "X" else '白(O)'} です。
合法手は次の通りです：{moves_str}

その中から1つだけ選び、次の形式で答えてください：
・選んだ手: <例: D3>
・理由: <簡単に理由を述べる>
"""

#==================== メイン対局ループ ====================
# 複数局の対局を通して勝敗を記録するメイン処理
def main():
  total_results = {"X": 0, "O": 0, 'draw': 0}

  for game_index in range(1, GAME_MAX + 1):
    head_message = f"[{game_index}/{GAME_MAX}局目]"

    print(f"\n================== 第{game_index}局 ==================")
    game = Othello()
    skip_count = 0
    # total_stones = 0
    players = ["X", "O"]
    human_player = players[HUMAN_FIRST] if PLAY_MODE in (0, 2) else None
    game_ended_by_error = False # LLMエラーによる終了フラグ
    last_turn_message = "" # 最終盤面表示のためのメッセージを保持

    print("\n現在の盤面: <<開始>>")
    game.print_board()

    while True:
      legal_moves = game.get_legal_moves(game.turn)

      if not legal_moves:
        # 途中のパス表示は行わないが、パスカウントは続ける
        skip_count += 1
        if skip_count >= 2:
          break # 両者パスでゲーム終了
        game.turn = "O" if game.turn == "X" else "X"
        continue
      skip_count = 0 # 合法手があった場合はパスカウントをリセット

      move = None
      if PLAY_MODE == 2:
        player_name = '先手(黒)' if game.turn == "X" else '後手(白)'
        print(f"\n{player_name} の番です（{'●' if game.turn == 'X' else '○'}） 合法手: {[game.move_to_str(x, y) for x, y in legal_moves]}")
        while True:
          user_input = input("手を入力（例: D3）: ").strip().upper()
          try:
            x, y = game.str_to_move(user_input)
            if (x, y) in legal_moves:
              move = (x, y)
              break
            else:
              print("その手は合法ではありません。")
          except:
            print("形式が不正です。")

      elif (PLAY_MODE == 0 and game.turn == human_player):
        print(f"\nあなたの番（{'●' if game.turn == 'X' else '○'}） 合法手: {[game.move_to_str(x, y) for x, y in legal_moves]}")
        while True:
          user_input = input("手を入力（例: D3）: ").strip().upper()
          try:
            x, y = game.str_to_move(user_input)
            if (x, y) in legal_moves:
              move = (x, y)
              break
            else:
              print("その手は合法ではありません。")
          except:
            print("形式が不正です。")
      
      # LLMの手番処理
      else:
        model = MODEL_NAME_1 if game.turn == "X" else MODEL_NAME_2 if PLAY_MODE == 1 else MODEL_NAME_1
        
        for i in range(LLM_RETRY_LIMIT):
          prompt = format_prompt(game.to_text(), legal_moves, game.turn)
          print(f"\nLLM({model})に問い合わせ中... (試行 {i + 1}/{LLM_RETRY_LIMIT})")
          
          try:
            response = query_llm(prompt, model)
            # if "<think>" in response:
            #   response = remove_tagged_sections(response, "<think>", "</think>")
            
            print(f"LLM({model})の応答:")
            print(response.strip())
            
            found_move = False
            for line in response.splitlines():
              if "選んだ手" in line:
                move_str = line.split(":")[1].strip()
                try:
                  x, y = game.str_to_move(move_str)
                  if (x, y) in legal_moves:
                    move = (x, y)
                    found_move = True
                    break
                except:
                  continue
            
            if found_move:
              break
            
          except Exception as e:
            print(f"LLMへの問い合わせ中にエラーが発生しました (試行 {i + 1}):", e)

        else: # リトライ上限に達した場合
          print(f"\nLLM({model})が正しい応答を返せませんでした。(試行回数: {LLM_RETRY_LIMIT}回)")
          
          if game.turn == "X":
            print("勝者: ○(後手) - 先手(黒)の応答失敗による")
            total_results["O"] += 1
          else:
            print("勝者: ●(先手) - 後手(白)の応答失敗による")
            total_results["X"] += 1
          
          game_ended_by_error = True
          break

      if game_ended_by_error:
        break

      turn_player = game.turn
      game.apply_move(move[0], move[1], game.turn)

      # ★修正箇所: 最終盤面表示前のパスメッセージの制御
      x_count, o_count = game.count_stones()
      total_stones = x_count + o_count
      
      # 最終盤面表示のためのメッセージを更新
      if PLAY_MODE == 1:
        model_name = MODEL_NAME_1 if turn_player == "X" else MODEL_NAME_2
        last_turn_message = f"{head_message} 最終盤面: <<{'先手(黒)' if turn_player == 'X' else '後手(白)'}[LLM({model_name})]着手後>>"
      elif PLAY_MODE == 0:
        if turn_player == human_player:
          last_turn_message = f"{head_message} 最終盤面: <<{'先手(黒)' if turn_player == 'X' else '後手(白)'}[プレイヤー1]着手後>>"
        else:
          last_turn_message = f"{head_message} 最終盤面: <<{'先手(黒)' if turn_player == 'X' else '後手(白)'}[LLM({MODEL_NAME_1})]着手後>>"
      else:
        last_turn_message = f"{head_message} 最終盤面: <<{'先手(黒)' if turn_player == 'X' else '後手(白)'}[プレイヤー{'1' if turn_player == 'X' else '2'}]着手後>>"

      # ★修正箇所: 各着手後の盤面表示を元に戻す
      if not game_ended_by_error:
        if total_stones < 64:
          if PLAY_MODE == 1:
            model_name = MODEL_NAME_1 if turn_player == 'X' else MODEL_NAME_2
            print(f"\n{head_message} 現在の盤面: <<{'先手(黒)' if turn_player == 'X' else '後手(白)'}[LLM({model_name})]着手後>>")
          elif PLAY_MODE == 0:
            if turn_player == human_player:
              print(f"\n{head_message} 現在の盤面: <<{'先手(黒)' if turn_player == 'X' else '後手(白)'}[プレイヤー1]着手後>>")
            else:
              print(f"\n{head_message} 現在の盤面: <<{'先手(黒)' if turn_player == 'X' else '後手(白)'}[LLM({MODEL_NAME_1})]着手後>>")
          else:
            print(f"\n{head_message} 現在の盤面: <<{'先手(黒)' if turn_player == 'X' else '後手(白)'}[プレイヤー{'1' if turn_player == 'X' else '2'}]着手後>>")
          game.print_board()
      
      # game.turn = "O" if game.turn == 'X' else 'X'

    if game_ended_by_error:
        continue

    # 64マス埋まっていない かつ 両者パスで終了した場合のみメッセージ表示
    if total_stones < 64 and skip_count >= 2:
      player_name = '先手(黒)' if game.turn == 'X' else '後手(白)' # game.turnは最後の番をパスした側
      print(f"{player_name} に合法手がありません。パスします。")
      print(f"両者に合法手がないため、対局終了。")

    if last_turn_message.strip() != "":
      print(f"\n{last_turn_message}") 
      game.print_board()
      
    x_count, o_count = game.count_stones()
    print(f"{head_message} ●(先手): {x_count} ○(後手): {o_count}")
    if x_count > o_count:
      print(f"{head_message} 勝者: ●(先手)")
      total_results["X"] += 1
    elif o_count > x_count:
      print(f"{head_message} 勝者: ○(後手)")
      total_results["O"] += 1
    else:
      print(f"{head_message} 引き分けです。")
      total_results['draw'] += 1

  print("\n================== 対局結果まとめ ==================")
  print(f"総対局数: {GAME_MAX}")
  print(f"先手(黒) 勝利数: {total_results['X']}")
  print(f"後手(白) 勝利数: {total_results['O']}")
  print(f"引き分け: {total_results['draw']}")
  if GAME_MAX > 0:
      x_win_rate = total_results["X"] / GAME_MAX * 100
      o_win_rate = total_results["O"] / GAME_MAX * 100
      print(f"勝率 先手(黒): {x_win_rate:.1f}%  / 勝率 後手(白): {o_win_rate:.1f}%")

if __name__ == "__main__":
  main()
