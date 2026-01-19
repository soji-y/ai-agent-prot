import os
import io
import time
import glob
from pathlib import Path
from collections import deque
import re
import datetime
from PIL import Image
import base64
import ollama # VLMでの画像入力のために追加
import json

# langchainのバージョンによっては 'langchain_core' からのimportが推奨されます
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import TextLoader, PyPDFLoader #, PandasExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
import pandas as pd
from langchain_core.documents import Document

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MAX_HISTORY_LEN = 30 # 最大履歴数
IMAGE_SIZE = 640
INPUT_DOC_EXTS = [".txt", ".pdf", ".xlsx"]

# Transformerを使用する
USE_TRANSFORMERS = False #True

if USE_TRANSFORMERS:
  # --- HuggingFace関連のインポート ---
  from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
  from langchain_community.llms import HuggingFacePipeline # LLM用
  from langchain_community.embeddings import HuggingFaceEmbeddings # 埋め込み用
  import torch

# 画像をリサイズ
def resize_image_ratio(image_pil: Image.Image, max_size: int) -> Image.Image:
    """
    PIL.Imageオブジェクトを、大きい方の辺がmax_sizeになるようにアスペクト比を維持してリサイズする。

    Args:
        image_pil (PIL.Image.Image): リサイズするPIL.Imageオブジェクト。
        max_size (int): 大きい方の辺の目標サイズ。

    Returns:
        PIL.Image.Image: リサイズされた新しいPIL.Imageオブジェクト。
    """
    original_width, original_height = image_pil.size
    # print(f"元の画像サイズ: 幅={original_width}, 高さ={original_height}")

    # リサイズ後の新しいサイズを計算
    if original_width > original_height:
        # 幅の方が大きい場合、幅をmax_sizeに合わせる
        new_width = max_size
        new_height = int(original_height * (max_size / original_width))
    else:
        # 高さの方が大きい、または同じ場合、高さをmax_sizeに合わせる
        new_height = max_size
        new_width = int(original_width * (max_size / original_height))

    # アスペクト比を維持してリサイズ
    # Image.LANCZOS は高品質なリサンプリングフィルター
    resized_image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
    # print(f"リサイズ後の画像サイズ: 幅={resized_image_pil.width}, 高さ={resized_image_pil.height}")

    return resized_image_pil

# PIL.ImageオブジェクトをBase64エンコードされた文字列に変換
def encode_pil_image_to_base64(image_pil: Image.Image) -> str:
  buffered = io.BytesIO()
  # JPEG形式で保存。必要に応じてPNGなど他の形式も指定可能。
  # モデルが受け入れやすい形式を選ぶのが良いでしょう。
  image_pil.save(buffered, format="PNG")
  return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ブロックをマーカー込みで削除
def remove_blocks(text:str, start_char:str, end_char:str) -> str:
    pattern = re.compile(rf"{start_char}[\s\S]*?{end_char}")
    return re.sub(pattern, '', text).strip()

# 現在の時刻を取得
def get_current_time_str():
  # ▼▼▼ 日時取得処理を共通化 ▼▼▼
  current_datetime = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
  weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]
  weekday_str = weekdays_jp[current_datetime.weekday()]
  
  # current_time_str = current_datetime.strftime(f"%Y年%m月%d日({weekday_str}) %H時%M分%S秒")
  current_time_str = (
    f"{current_datetime.year}年"
    f"{current_datetime.month:d}月"
    f"{current_datetime.day:d}日"
    f"（{weekday_str}） "
    f"{current_datetime.hour:d}時"
    f"{current_datetime.minute:02d}分"
    f"{current_datetime.second:02d}秒"
)

  return current_time_str

# RAGを用いたAIエージェント
class AgentCharacter:
  def __init__(self, model_name:str="llama3", emb_model_name:str=None, system_role:str = "あなたは役立つAIアシスタントです。", chunk_size:int=1000, chunk_overlap:int=200, max_history_length:int=6):
    """
    RAG Agentを初期化します。

    Args:
      model_name (str): Ollamaで利用するモデル名。
      system_role (str): LLMに渡すシステムプロンプトの役割設定。
      chunk_size (int): テキスト分割のチャンクサイズ。
      chunk_overlap (int): テキスト分割のチャンクオーバーラップ。
      max_history_length (int): 保持する会話履歴の最大長（往復数×2）。
    """
    self.model_name = model_name
    if emb_model_name:
      self.emb_model_name = emb_model_name
    else:
      self.emb_model_name = model_name
    self.system_role = system_role
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    self.max_history_length = max_history_length # 会話履歴の最大長
    self.vector_store = None
    self.rag_chain = None
    self.chat_history = deque(maxlen=max_history_length) # [] # 会話履歴を保存するリスト
    if USE_TRANSFORMERS:
      self._initialize_llm_and_embeddings_transformers()
    else:
      self._initialize_llm_and_embeddings()

  # LLMと埋め込みモデルを初期化
  def _initialize_llm_and_embeddings(self):
    """LLMと埋め込みモデルを初期化します。"""
    print(f"LLM({self.model_name})と埋め込みモデル({self.emb_model_name})を初期化しています...")
    try:
      self.llm = Ollama(model=self.model_name)
      self.embeddings = OllamaEmbeddings(model=self.emb_model_name)
    except Exception as e:
      print(f"LLMまたは埋め込みモデルの初期化中にエラーが発生しました: {e}")
      print("Ollamaサーバーが起動しているか、指定したモデルがダウンロードされているか確認してください。")
      raise e
    print("初期化が完了しました。")

  # LLMと埋め込みモデルを初期化
  def _initialize_llm_and_embeddings_transformers(self):
    """LLMと埋め込みモデルを初期化します。"""
    print(f"HuggingFace LLM({self.model_name})と埋め込みモデル({self.emb_model_name})を初期化しています...")
    try:
      # --- LLMの初期化 (HuggingFacePipeline) ---
      # トークナイザーとモデルをロード
      llm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      llm_model = AutoModelForCausalLM.from_pretrained(
          self.model_name,
          torch_dtype=torch.float16, #self.torch_dtype, # 必要に応じて指定 (例: torch.float16, torch.bfloat16)
          low_cpu_mem_usage=True, # メモリ使用量を抑える
      )
      self.device = "cuda:0"
      # モデルをデバイスに移動
      llm_model.to(self.device)
      llm_model.eval() # 評価モードに設定

      # テキスト生成パイプラインを作成
      # モデルによってはtrust_remote_code=Trueが必要な場合があります
      # また、max_new_tokens, do_sample, top_k, temperature など、
      # 推論パラメータもここで設定できます。
      llm_pipeline = pipeline(
          "text-generation",
          model=llm_model,
          tokenizer=llm_tokenizer,
          max_new_tokens=512, # 生成する最大トークン数
          pad_token_id=llm_tokenizer.eos_token_id, # EOSトークンIDを指定
          device=0 if "cuda" in self.device else -1, # pipelineのdevice設定 (0でCUDA:0, -1でCPU)
      )
      self.llm = HuggingFacePipeline(pipeline=llm_pipeline)

      # --- 埋め込みモデルの初期化 (HuggingFaceEmbeddings) ---
      # 埋め込みモデルは、Sentence Transformers形式のモデルIDを使用するのが一般的
      self.embeddings = HuggingFaceEmbeddings(
          model_name=self.emb_model_name,
          model_kwargs={'device': self.device} # デバイスを指定
      )

    except Exception as e:
      print(f"HuggingFaceモデルの初期化中にエラーが発生しました: {e}")
      print("指定したモデルIDが存在するか、必要なライブラリがインストールされているか確認してください。")
      print("また、GPUを使用する場合はCUDAが正しく設定されているか確認してください。")
      raise e
    print("初期化が完了しました。")

  # エクセルをロード
  def _load_excel_with_pandas(self, file_path: str) -> list[Document]:
    """
    pandasを使用してExcelファイルを読み込み、LangChainのDocumentオブジェクトのリストに変換します。
    各シートの内容を個別のDocumentとして扱います。
    """
    try:
      # 全てのシートをDataFrameとして読み込む
      xls = pd.ExcelFile(file_path)
      sheet_names = xls.sheet_names

      documents = []
      for sheet_name in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # データフレームを文字列に変換（例：CSV形式）
        # ここではシンプルにCSV形式の文字列にする例
        content_str = df.to_csv(index=False)

        # メタデータとしてファイル名、シート名などを追加
        metadata = {"source": file_path, "file_type": "excel", "sheet_name": sheet_name}

        # Documentオブジェクトとして追加
        documents.append(Document(page_content=content_str, metadata=metadata))

      return documents
    except Exception as e:
      print(f"Excelファイルの読み込み中にエラーが発生しました: {e}")
      # エラーが発生した場合は空のリストを返すか、例外を再発生させる
      return []

  # 指定されたフォルダからDocumentを読み込み
  def load_doc_directory(self, vector_doc_dir_path: str, save_flg=False):
    doc_path_list = []

    for ext in INPUT_DOC_EXTS:
      doc_path_list += glob.glob(os.path.join(vector_doc_dir_path, f"*{ext}"))

    first_vect = ""
    for doc_path in doc_path_list:
      vect_path = os.path.splitext(doc_path)[0]
      if first_vect == "":
        first_vect = vect_path # 最初のパスを取得
      self.load_and_process_document(doc_path, vect_path, save_flg=save_flg, no_builde_chain=True)

    # ラングチェインを構築
    self._build_rag_chain(chara_flg=True)
    # if self.vector_store:
    #   if save_flg:
    #     vector_path = f"{first_vect}_set"
    #     self.vector_store.save_local(vector_path)
    #     print(f"ベクトルストアを保存しました => [{vector_path}]")

  # 指定されたファイルを読み込み
  def load_and_process_document(self, file_path: str, vector_path:str=None, save_flg=False, no_builde_chain=False):
    """指定されたファイルを読み込み、ベクトルストアを作成し、RAGチェーンを構築します。"""
    vector_store = None
    if vector_path and os.path.isdir(vector_path):
      # ベクトルストアがあるとき
      try:
        vector_store = FAISS.load_local(vector_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"ベクトルストアをロードしました。[{str(Path(vector_path))}]")
        if self.vector_store:
          self.vector_store.merge_from(vector_store)
          print(f"ベクトルストアをマージしました。")
        else:
          self.vector_store = vector_store
      except Exception as e:
        print(f"ベクトルストアのロードに失敗しました: {e}")
        vector_store = None

    if not vector_store:
      if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

      _, file_extension = os.path.splitext(file_path)
      file_extension = file_extension.lower()
      print(f"'{str(Path(file_path))}' ({file_extension}形式) を読み込んでいます...")

      try:
        if file_extension == ".txt": loader = TextLoader(file_path, encoding="utf-8")
        elif file_extension == ".pdf": loader = PyPDFLoader(file_path)
        # elif file_extension == ".xlsx": loader = PandasExcelLoader(file_path, sheet_name=None, pandas_kwargs={"header": None})
        elif file_extension == ".xlsx":
          # ここで自作のload_excel_with_pandas関数を呼び出す
          docs = self._load_excel_with_pandas(file_path)
          if not docs: # Excel読み込みでエラーが発生し、空リストが返された場合
            print(f"Excelファイルの読み込みで問題が発生したため、処理を中断します。")
            return # 処理を中断

        else: raise ValueError(f"サポートされていないファイル形式です: {str(Path(file_extension))}")
        docs = loader.load()
      except Exception as e:
          print(f"ドキュメントの読み込み中にエラーが発生しました: {e}")
          return

      print("ドキュメントを分割し、ベクトルストアを作成中...")
      try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        splits = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        print("ベクトルストアが正常に作成されました。")
        if self.vector_store:
          self.vector_store.merge_from(vector_store)
          print(f"ベクトルストアをマージしました。")
        else:
          self.vector_store = vector_store

        if save_flg:
          vector_path = f"{os.path.splitext(file_path)[0]}"
          vector_store.save_local(vector_path)
          print(f"ベクトルストアを保存しました => [{str(Path(vector_path))}]")
      except Exception as e:
        print(f"ベクトルストアの作成に失敗しました: {e}")
        return

    if not no_builde_chain:
      self._build_rag_chain(chara_flg=True)

  # 会話履歴を考慮したRAGチェインを構築
  def _build_rag_chain(self, user_name="マスター", chara_flg=False):
    """会話履歴を考慮したRAGチェインを構築します。"""
    if self.vector_store is None:
      self.retriever = None
      self.rag_chain = None
      print("ベクトルストアが初期化されていないので、RAGリトリーバーを構築しません。")
      return

    print("会話履歴を考慮したRAGリトリーバーを構築しています...")
    retriever = self.vector_store.as_retriever()

    # 1. 履歴を考慮したリトリーバーの作成
    contextualize_q_system_prompt = (
        # "チャット履歴と最新のユーザーからの質問が与えられます。"
        # "チャット履歴を使って、ユーザーの質問を文脈的に理解し、"
        # "検索に必要な独立した質問に言い換えてください。"
        # "言い換える必要がない場合は、質問をそのまま使用してください。"
      f"会話履歴と{user_name}からのセリフを元に調べたい話題を抽出してください。"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        self.llm, retriever, contextualize_q_prompt
    )
    self.retriever = history_aware_retriever
    print("RAGリトリーバーが構築されました。")
    return
    # # 2. 質問応答チェーンの作成
    # qa_system_prompt = (
    #     f"{self.system_role}\n"
    #     # "あなたは与えられたコンテキスト情報とチャットの文脈にのみ基づいて、ユーザーの質問に回答してください。\n"
    #     # "知らない情報については、正直に「分かりません」と答えてください。\n"
    #     # "回答は日本語で行ってください。\n\n"
    #     # "コンテキスト:\n{context}"
    #     "コンテキスト:\n{context}"
    # )
    # qa_prompt = ChatPromptTemplate.from_messages([
    #     ("system", qa_system_prompt),
    #     MessagesPlaceholder("dynamic_system_message"),
    #     MessagesPlaceholder("chat_history"),
    #     ("human", "{input}"),
    #     ("system", "{context}")
    # ])
    # doc_chain = create_stuff_documents_chain(self.llm, qa_prompt)

    # # 3. 上記2つを結合して最終的なRAGチェーンを作成
    # self.rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)
    # print("RAGチェインが構築されました。")

  # メモリ履歴をメッセージ形式に変換
  def _set_history(self, sys_prompt):
    history_messages = [{"role": "system", "content": sys_prompt}]
    for msg in self.chat_history:
      match = re.match(r"^(.+?):\s(.*)", msg.content, re.DOTALL)
      if isinstance(msg, HumanMessage) and match:
        history_messages.append({"role": "user", "content": match.group(2)})
      elif isinstance(msg, AIMessage):
        history_messages.append({"role": "assistant", "content": msg.content})
        
    return history_messages
  
  # RAGAgentクラスのaskメソッドのみを抜粋
  def ask(self, message:str, user:str="マスター", image_path:str=None, verbose:bool=False):
    """
    RAGシステムに質問を投げかけ、会話履歴を更新し、回答を取得します。

    Args:
      message (str): ユーザーからのメッセージ。
      image_path (str, optional): 入力する画像のファイルパス。Defaults to None.
      verbose (bool): Trueの場合、チェーンの各ステップの詳細なログを出力します。
    Returns:
      str: LLMによって生成された回答。
    """
    # if self.rag_chain is None:
    #   raise ValueError("RAGチェインが構築されていません。")


    dynamic_system_message_content = f"※現在の日時: {get_current_time_str()}\n"
    # ▲▲▲ ここまで ▲▲▲

    # 画像が入力されていない場合は、既存のRAGチェーンを使用
    if self.rag_chain and image_path is None:
      # print("回答を生成中...\n")
      try:
        # 共通化された日時情報からSystemMessageオブジェクトを作成
        dynamic_system_messages = [SystemMessage(content=dynamic_system_message_content)]

        chat_history_list = []
        # ollama_messages = [{"role": "system", "content": full_system_prompt}]
        for msg in self.chat_history:
          match = re.match(r"^(.+?):\s(.*)", msg.content, re.DOTALL)
          if isinstance(msg, HumanMessage) and match:
            chat_history_list.append({"role": "user", "content": match.group(2)})
          elif isinstance(msg, AIMessage):
            chat_history_list.append({"role": "assistant", "content": msg.content})

        # 非同期
        response = ""
        for chunk in self.rag_chain.stream({
            "dynamic_system_message": dynamic_system_messages,
            "chat_history": chat_history_list, # list(self.chat_history)
            "input": message
            }, config={"verbose": verbose}):
          if "answer" in chunk and chunk["answer"] is not None:
              yield chunk["answer"]
              response += chunk["answer"]

        # 会話履歴を更新
        self.chat_history.append(HumanMessage(content=f"{user}: {message}"))
        self.chat_history.append(AIMessage(content=response))

        return response
      except Exception as e:
        print(f"回答生成中にエラーが発生しました: {e}")
        return "回答の生成中にエラーが発生しました。"

    else:
      # 画像が入力された場合は、RAGでコンテキストを取得し、ollamaライブラリでVLMを直接呼び出す
      # print("画像付きで回答を生成中...\n")
      try:
        # if not os.path.exists(image_path):
        #   raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        # --- RAGルーティング ---
        # "あなたはユーザーの質問の意図を判断するAIアシスタントです。\n" + \
        # f"\n【重要】まず、{user}からのメッセージがRAGシステム（提供されたドキュメントからの情報検索）を必要とするかどうかを判断し、「Yes」or「No」のみで回答してください。"
        router_sys_prompt = self.system_role + f"\n【重要】まず、{user}からのメッセージが詳しい回答を質問しているかを判断し、詳細調査の有無を「Yes」or「No」のみで回答してください。"
             
        # 質問の意図を判断するためのプロンプト
        # router_prompt = ChatPromptTemplate.from_messages([
        #     ("system", router_sys_prompt),
  
        #     # 以下のいずれかのカテゴリを出力してください。
        #     #- "rag_needed": ドキュメント検索が必要な場合 (例: 「〇〇とは？」「この資料の××について教えて」など、質問内容が明確な情報検索を必要とする場合)
        #     #- "general_chat": 一般的な会話、挨拶、常識的な質問、RAGドキュメントに直接関連しない質問の場合 (例: 「こんにちは」「今日の天気は？」「あなたについて教えて」など)
        #     # """),
        #     # JSONフォーマット:
        #     # {"category": "category_name"}
        #     # 回答はJSON形式のみで出力してください。それ以外のテキストは含めないでください。
        #     # """),
        #     MessagesPlaceholder("chat_history"), # 履歴も判断材料にする
        #     ("human", "{input}"),
        # ])

        if False:
          # ルーティング用LLMを呼び出し、カテゴリを判断
          # chat_history は list() で変換
          router_response_chunks = self.llm.stream(
              input="", # input="" は必要に応じて
              messages=router_prompt.format_messages(
                  input=message, 
                  chat_history=list(self.chat_history)
              )
          )

          router_response_str = ""
          for chunk in router_response_chunks:
            if chunk and "content" in chunk:
              router_response_str += chunk["content"]

          # JSONパースを試みる
          try:
              decision = json.loads(router_response_str)
              category = decision.get("category", "general_chat") # デフォルトは一般的な会話
          except json.JSONDecodeError:
              print(f"警告: ルーターの応答をJSONとしてパースできませんでした: {router_response_str}")
              category = "general_chat" # パース失敗時は汎用会話にフォールバック
        
        # ollama_messages = [{"role": "system", "content": router_sys_prompt}]
        # for msg in self.chat_history:
        #   ollama_messages.append(msg)

        ollama_messages = self._set_history(router_sys_prompt)
        ollama_messages.append({"role": "user", "content": message})

        router_res = ollama.chat(
          model=self.model_name,
          messages=ollama_messages
          # router_prompt.format_messages(
          #         input=message, 
          #         chat_history=list(self.chat_history)
          #         ),
          # stream=True
        )

        print(router_res['message']['content'])
        print()

        rag_use_response = remove_blocks(router_res['message']['content'], "<think>", "</think>")
        if "yes" in rag_use_response.lower().strip() :
          rag_use = True
        else:
          rag_use = False
        # category = "rag_needed"
        # print(f"質問のカテゴリ判断: {category}")

        context = ""
        if self.retriever and rag_use: #category == "rag_needed":
          # 1. RAGによるコンテキスト取得
          retrieved_docs = self.retriever.invoke({
              "input": message,
              "chat_history": list(self.chat_history)
          })
          context = "\n\n".join([d.page_content for d in retrieved_docs])

          # 2. ollama.chat用のメッセージリスト構築
          # 共通化された日時情報をシステムプロンプトに埋め込む
          qa_system_prompt = (
              f"{self.system_role}\n"
              f"{dynamic_system_message_content}"
              "※追加情報:\n{context}"
          )
        else:
          qa_system_prompt = (
              f"{self.system_role}\n"
              f"{dynamic_system_message_content}"
          )

        full_system_prompt = qa_system_prompt.format(context=context)

        # 履歴をollama形式に変換
        ollama_messages = self._set_history(full_system_prompt)
        
        # ollama_messages = [{"role": "system", "content": full_system_prompt}]
        # for msg in self.chat_history:
        #   match = re.match(r"^(.+?):\s(.*)", msg.content, re.DOTALL)
        #   if isinstance(msg, HumanMessage) and match:
        #     ollama_messages.append({"role": "user", "content": match.group(2)})
        #   elif isinstance(msg, AIMessage):
        #     ollama_messages.append({"role": "assistant", "content": msg.content})

        # 現在のユーザーメッセージ（テキスト＋画像）を追加
        # with open(image_path, "rb") as image_file:
        #   encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        user_message_content = { "role": "user", "content": message }
        if image_path:
          # PILを使って画像を読み込む
          image_pil = Image.open(image_path)

          # 画像をリサイズメソッドで処理
          resized_image = resize_image_ratio(image_pil, IMAGE_SIZE)

          # リサイズされたPIL.ImageオブジェクトをBase64エンコード
          encoded_image = encode_pil_image_to_base64(resized_image)

          # user_message_content = {
          #     "role": "user",
          #     "content": message,
          #     "images": [encoded_image]
          # }
          user_message_content["images"] = [encoded_image]

        ollama_messages.append(user_message_content)

        # 3. ollama.chatをストリーミングで呼び出し
        response = ""
        stream = ollama.chat(
            model=self.model_name,
            messages=ollama_messages,
            stream=True
        )
        for chunk in stream:
          if chunk.get('message', {}).get('content'):
            content_chunk = chunk['message']['content']
            yield content_chunk
            response += content_chunk

        # 4. 会話履歴を更新
        self.chat_history.append(HumanMessage(content=f"{user}: {message}"))
        self.chat_history.append(AIMessage(content=response))

        return response

      except Exception as e:
        print(f"画像付き回答生成中にエラーが発生しました: {e}")
        # return "画像付き回答の生成中にエラーが発生しました。"
      return None


if __name__ == "__main__":
  user = "マスター"
  user = "そーじ"
  ai_chara = "LUCA(ルカ)"
  ai_chara = "イコ"

  sys_role = f"あなたは日本語が得意な女性型の優秀なAIエージェント「{ai_chara}」です。明るく元気に回答します。マスターには絶対服従します。"
  sys_role = f""

  if USE_TRANSFORMERS:
    model_name = "Qwen/Qwen3-4B"
    emb_model_name = "Qwen/Qwen3-4B"
  else:
    emb_model_name = "qwen3"
    model_name = "gemma3"
    model_name = "qwen3:14b"
    model_name = "deepseek-r1:14b"
    model_name = "yuma/DeepSeek-R1-Distill-Qwen-Japanese:32b"
    model_name = "yuma/DeepSeek-R1-Distill-Qwen-Japanese:14b"
    # model_name = "qwen3"
    model_name = "deepseek-r1-ja-14b"
    
    # model_name = "Ayaka_0.1.0"
    model_name = "Iko_v0.1.0"
    # model_name = "qwen2.5vl"
    # emb_model_name = "qwen2.5vl:32b"

  # test_txt_file = "ohmizaka_high_school_overview.txt"
  # doc_dir_path = "./doc_date"
  doc_dir_path = "./doc"

  try:
    start_t = time.time()
    agent = AgentCharacter(
      model_name=model_name,
      emb_model_name=emb_model_name,
      system_role=sys_role, #"あなたは与えられたドキュメントの内容に忠実で、優秀な対話型アシスタントです。",
      max_history_length=MAX_HISTORY_LEN # 会話記憶
    )
    print(f"Agent Initialize Time: [{time.time() - start_t:.2f}]s")
    start_t = time.time()

    # ドキュメントファイルを読み込む
    agent.load_doc_directory(vector_doc_dir_path=doc_dir_path, save_flg=True)

    # image_path = "./img/桜ロール.jpg"
    print(f"Loaded Document Time: [{time.time() - start_t:.2f}]s")

    # if agent.rag_chain: # if agent.retriever:
    print("ドキュメントの準備が完了しました。対話を開始します。")
    print()
    while True:
      user_question = input(f"{user}: ")
      if user_question.lower() == 'exit':
        print("終了します")
        break

      image_path = None
      if ' ' in user_question:
        if os.path.isfile(user_question.split(' ')[1]):
          image_path = user_question.split(' ')[1]

      # response = agent.ask(user_question, verbose=True)
      stream_response = agent.ask(user_question, user, image_path=image_path, verbose=True)
      # print(f"{ai_chara}:", end='')
      response = ""
      no_print = False
      for chunk in stream_response:
        if chunk.startswith("<think>"):
          no_print = True
        elif chunk.startswith("</think>"):
          no_print = False
        else:
          if not no_print:
            if response == "":
              print(f"{ai_chara}:", end='')
            response += chunk
            if response.strip() != "":

              print(chunk, end='', flush=True)
      print()
      print("-"*50)
      # response = response.strip()
      # response = remove_blocks(response, "<think>", "</think>")
      # print(f"{ai_chara}: {response}\n")
      

  except FileNotFoundError as e:
    print(e)
  except Exception as e:
    print(f"予期せぬエラーが発生しました: {e}")

