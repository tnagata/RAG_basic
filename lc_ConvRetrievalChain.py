from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain_core.messages.base import messages_to_dict

# インデックスのパス
index_path = "./storage"

# モデルのパス
model_path = "./ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf"

embedding_model_dir = "./hf_model"
# 埋め込みモデルの読み込み
embedding_model = HuggingFaceEmbeddings(
    model_name=f"{embedding_model_dir}"
)


# インデックスの読み込み
index = FAISS.load_local(
    folder_path=index_path, 
    embeddings=embedding_model
)

# プロンプトテンプレートの定義
question_prompt_template = """

{context}

Question: {question}
Answer: """

# プロンプトの設定
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, # プロンプトテンプレートをセット
    input_variables=["context", "question"] # プロンプトに挿入する変数
)

# モデルの設定
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=0, # gpuに処理させるlayerの数  25-->0
    stop=["Question:", "Answer:"], # 停止文字列
    n_ctx=2048, # コンテキストの最大長 を追加 TN 2024/5/4
)

# メモリー（会話履歴）の設定
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", # メモリーのキー名
    output_key="answer", # 出力ののキー名
    k=5, # 保持する会話の履歴数
    return_messages=True, # チャット履歴をlistで取得する場合はTrue
)

# （RAG用）会話chainの設定
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=index.as_retriever(
        search_kwargs={'k': 3} # indexから上位いくつの検索結果を取得するか
    ),
    combine_docs_chain_kwargs={'prompt': QUESTION_PROMPT}, # プロンプトをセット
    chain_type="stuff", # 検索した文章の処理方法
    memory=memory  # メモリーをセット
)

# 会話開始
while True:
    user_input = input("Human: ")
    if user_input == "exit":
        break
    
    # LLMの回答生成
    response = chain.invoke({"question": user_input})

    # 回答を確認
    response_answer = response["answer"]
    print(f"AI: {response_answer}")

# 会話履歴の確認
chat_history_dict = messages_to_dict(memory.chat_memory.messages)
print(f"\nmemory-------------------------------------------------------")
for i, chat_history in enumerate(chat_history_dict, 1):
    chat_history_type = chat_history["type"]
    chat_history_context = chat_history["data"]["content"]
    print(f"\n{chat_history_type}: {chat_history_context}")
print("-------------------------------------------------------------")