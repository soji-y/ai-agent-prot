## 説明
- AIエージェントのローカル実装テストアプリ

## 機能
- シンプルなローカルLLM実行スクリプト
- Ollamaを使用したローカルLLM実行スクリプト
- シンプルなAIエージェントをローカルLLMを使用して実装したスクリプト
  - Webサーチによる自律的な調査とそれに基づく回答
  - コードインタープリター機能を自律的に使用した回答
- ローカルLLMを使用した高度なAIエージェント
  - 音声会話
  - 画像を参照した会話
  - オセロのプレイ

## 推奨環境
GPU：Geforce RTX4080 (GPUメモリ16GB以上)

## 動作確認済環境
- Intel(R) Core(TM) i9-14900K 3.20 GHz
- RAM 64 GB
- Geforce RTX4090
- Windows11
- CUDA 12.8
- Python 3.12.8

## 動作画面
!["フォーム画像"](https://github.com/soji-y/ai-agent-prot/blob/main/advanced_ai_agent/advanced_ai_agent.png)

## 事前インストール
Ollama: https://ollama.com/download/windows
AivisSpeech: https://aivis-project.com/
 - 執事: [VC] ろてじん（長老ボイス), [style_id] 391794336 
 - 秘書(男性): [VC] にせ, [style_id] 1937616896 
 - 秘書(女性): [VC] るな, [style_id] 345585728 
 - メイド:  [VC] 中２, [style_id] 604166016
## 使用方法
#### 1. 本リポジトリをクローン
#### 2. 必要なライブラリをインストール
(1) venvやanaconda等で仮想環境を作成\
```
# anaconda
conda create -n ai_agent python=3.12.8
```
(2) pytorch等のインストール (CUDA環境に合わせたライブラリ)
https://pytorch.org/
```
# CUDA12.8版
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
(3) クローンしたフォルダに遷移し「requirements.txt」のライブラリをインストール
```
pip3 install -r requirements.txt
```
(4) Playwright実行用ブラウザをインストール
```
playwright install
```
(5) Ollamaの自動Tool判定機能をOFFにして再起動【重要】
```
OLLAMA_NO_TOOLS=1
```
上記を実行してOllamaを再起動する。
※これを行わないとAIエージェント用に出力したJson形式文字列がOllamaに自動パースされてエージェントは動作しません。

#### 3. それぞれのスクリプトの実行 (シンプルなAI、AIエージェント)
(1) シンプルなローカルLLM
```
cd simple_ai_agent
python simple_llm.py
```

(2) シンプルなOllamaを利用したローカルLLM
```
cd simple_ai_agent
python simple_llm_ollama.py
```

(3) シンプルなローカルLLMAIエージェント
```
cd simple_ai_agent
python ai_agent.py
```
以下の機能に対応。
- Webサーチによる最新の情報の取得
- コードインタープリター機能による高度な計算

#### 4. 会話と終了 (シンプルなAI、AIエージェント)
(1) 「目的を入力」の入力エリアに任意のプロンプトを入力して「Enter」 -> 応答  
(2) 「目的を入力」の入力エリアに「exit」を入力して「Enter」 ｰ> 終了
#### 5. 入力テキストボックスに任意の質問文を入力して「送信」ボタン押下
