## LangChainライブラリを使ってローカルでRAGを動かす方法

セキュリティの確保とコストを最小化するために、完全にローカルで無料のLLMモデルと追加データをembeddingする、モデルを一度だけダウンロードし、それらを使用してRAGを動かすスクリプト<br>
python v.3.11.9で確認　（v.3.12では未確認）

1-1 以下のスクリプトを実行して、HuggingFaceのLLM(ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf)をローカル(ワーキングディレクトリ）に保存する:<br>
$ python download_model.py<br>
1-2 予めhf_modelディレクトリを作成し、embedding用のモデル(intfloat/multilingual-e5-large)をHuggingFaceのサイトよりダウンロードして、ここに保存する<br>

2-1 追加データをembeddingしたindexを生成して、保存する:<br>
追加データのindexを保存するディレクトリstorageを作成する<br>
追加データ保存用のdataディレクトリを作成し、追加データを保存する（今回は「相撲.txt」という易しくない文章）<br>
以下のコマンドを実行して、必要なライブラリをダウンロードする（１回だけ）<br>
$ pip install lamgchain<br>
$ pip install langchain-community==0.0.26   *最新版の0.0.27はバグがありNG　*1)<br>
MetaのFAISS(ベクトル検索ライブラリ)をインストールする<br>
NVIDIA製GPUがある場合は、gpu版のインストール<br>
$ pip install faiss-gpu<br>
そうでない場合は、cpu版のインストール<br>
$ pip install faiss-cpu<br>
2-2 以下のスクリプトを実行する<br>
$ python index_create.py    **追加データを変更したら再度走らせる<br>

3-1 RAGを試すために２つのスクリプトのいずれかを走らせる<br>
lc_RetrievalQA.py (質問をハードコーディングしたスクリプト）<br>
あるいは<br>
lc_ConvRetrievalChain.py (質問と答えをチャットできるスクリプト）<br>

### この例のディレクトリ構造：<br>
rag-basic<br>
├── ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf<br>
├── data<br>
   └── 相撲.txt<br>
├── download_model.py<br>
├── hf_model<br>
├── index_create.py<br>
├── lc_ConvRetrievalChain.py<br>
├── lc_RetrievalQA.py<br>
└── storage<br>

### 参考情報<br>
https://zenn.dev/yumefuku/articles/llm-langchain-rag#fn-cf80-2<br>　（全般にこれを基にしているが、ところどころ環境の違いでエラーが出る）<br>
https://qiita.com/matntntn/items/7cdca4f33ce03970ef20 （モデルのローカル利用に関して）<br>
https://wonderhorn.net/programming/hfdownload.html（モデルのローカル利用に関して）<br>
*1) v.0.0.27で下記のエラー<br>
"The de-serialization relies loading a pickle file. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine.You will need to set `allow_dangerous_deserialization` to `True` to enable deserialization.
原因：This comes from a breaking change in langchain-community 0.0.27<br>
*2) lc_ConvRetrievalChain.pyの実行で３回ぐらいの質問で下記のエラーが出る<br>
Requested tokens (514) exceed context window of 512<br>
対策として、LlamaCpp(model_path=model, verbose=True, n_ctx=2048)<br>
Add n_ctx=2048 to increase context length　（ググったらこのヒントが出てきた。効果あり）<br>


