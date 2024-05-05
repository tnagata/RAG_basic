## LangChainライブラリを使ってローカルでRAGを動かす方法

セキュリティの確保とコストを最小化するために、完全にローカルで無料のLLMモデルと追加データをembeddingする、モデルを一度だけダウンロードし、それらを使用してRAGを動かすスクリプト<br>
python v.3.11.9で確認　（v.3.12では未確認）

1-1 HuggingFaceのLLM(ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf)をローカル(ワーキングディレクトリ）に保存する:<br>
$python download_model.py<br>
1-2 予めhf_modelディレクトリを作成し、embedding用のモデル(intfloat/multilingual-e5-large)をHuggingFaceのサイトよりダウンロードして、ここに保存する<br>
追加データのindexを保存するディレクトリstorageを作成する
追加データ保存用のdataディレクトリを作成し、追加データを保存する（今回は「相撲.txt」という易しくない文章）
index_create.pyを走らせる（追加データを変更したら再度走らせる）
lc_RetrievalQA.py (１つの質問をハードコーディングしたスクリプト）あるいはlc_ConvRetrievalChain.py (質問と答えをチャットできるスクリプト）を走らせる

### この例のディレクトリ構造：<br/>
rag-basic<br/>
├── ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf<br>
├── data<br/>
   └── 相撲.txt<br/>
├── download_model.py<br/>
├── hf_model<br/>
├── index_create.py<br/>
├── lc_ConvRetrievalChain.py<br/>
├── lc_RetrievalQA.py<br/>
└── storage<br/>
