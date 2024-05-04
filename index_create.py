from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# 資料の格納場所（ディレクトリ）
data_dir = "./data"

# ベクトル化したインデックスの保存場所（ディレクトリ）
index_path = "./storage"

# ディレクトリの読み込み
loader = DirectoryLoader(data_dir)

embedding_model_pass = "./hf_model"
# 埋め込みモデルの読み込み
embedding_model = HuggingFaceEmbeddings(
    model_name=f"{embedding_model_pass}"
)

# テキストをチャンクに分割
split_texts = loader.load_and_split(
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200, # 分割したチャンクごとの文字数
        chunk_overlap=50  # チャンク間で被らせる文字数
    )
)
# チャンクの確認
for i, doc in enumerate(split_texts, 1):
    print(f"\nチャンク{i}: ================================")
    print(doc.page_content)
    print(f"文字数: {len(doc.page_content)}")
    print("===========================================")
# インデックスの作成
index = FAISS.from_documents(
    documents=split_texts,
    embedding=embedding_model,
)

# インデックスの保存
index.save_local(
    folder_path=index_path
)