from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader
import os

import json

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings  # Updated import



def preprocess_txt():

    # 1. 读取文本文件

    loader = TextLoader(r"data\data1.txt", encoding="utf-8")

    docs = loader.load()


    # 2. 切分文本

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    documents = splitter.split_documents(docs)


    # 3. 构建中文向量模型（bge）

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh")


    # 4. 创建向量库（FAISS）

    db = FAISS.from_documents(documents, embedding)


    # 5. 保存本地

    db.save_local("data/my_faiss_db")


# 在文件开头添加单例模型管理
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh")
    return _embedding_model

def preprocess_pdf(pdf_path="data/pdf", vector_db_path="data/pdf_faiss_db", chunk_size=500, chunk_overlap=100):
    """

    读取PDF文件并创建向量库
    

    Args:

        pdf_path: PDF文件或文件夹路径

        vector_db_path: 向量库保存路径

        chunk_size: 文本块大小

        chunk_overlap: 文本块重叠大小
    

    Returns:

        FAISS向量库实例
    """

    # 初始化文档列表

    all_docs = []
    

    # 判断是文件还是文件夹

    if os.path.isfile(pdf_path):

        # 单个PDF文件处理

        loader = PyPDFLoader(pdf_path)

        all_docs.extend(loader.load())

    elif os.path.isdir(pdf_path):

        # 处理文件夹中的所有PDF文件

        for file in os.listdir(pdf_path):

            if file.lower().endswith('.pdf'):

                file_path = os.path.join(pdf_path, file)

                loader = PyPDFLoader(file_path)

                all_docs.extend(loader.load())
    

    if not all_docs:

        print("未找到PDF文件")

        return None
    

    # 切分文本

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    documents = splitter.split_documents(all_docs)
    

    # 使用全局embedding模型
    embedding = get_embedding_model()
    
    # 创建向量库
    pdf_db = FAISS.from_documents(documents, embedding)
    

    # 保存向量库到本地

    pdf_db.save_local(vector_db_path)

    print(f"PDF向量库已保存到 {vector_db_path}")
    

    return pdf_db


def preprocess_json_papers(json_path, vector_db_path="data/papers_faiss_db", chunk_size=500, chunk_overlap=100, download_dir="data/pdf"):
    """
    从JSON文件中提取论文标题，下载PDF并创建向量库
    
    Args:
        json_path: JSON文件路径
        vector_db_path: 向量库保存路径
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        download_dir: PDF下载目录
    
    Returns:
        FAISS向量库实例
    """
    # 导入下载函数
    from down_PDF import download_pdf_by_title
    
    # 初始化文档列表
    all_docs = []
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    # 提取论文标题列表
    titles = papers["extra"]["recall_papers"]
    print(f"从JSON中提取到 {len(titles)} 篇论文标题")
    
    # 确保下载目录存在
    os.makedirs(download_dir, exist_ok=True)
    
    # 下载每篇论文的PDF
    downloaded_paths = []
    for title in titles:
        print(f"正在下载论文: {title}")
        pdf_path = download_pdf_by_title(title, download_dir)
        if pdf_path:
            downloaded_paths.append(pdf_path)
    
    print(f"成功下载 {len(downloaded_paths)} 篇论文")
    
    # 处理下载的PDF文件
    if downloaded_paths:
        for pdf_path in downloaded_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"处理PDF文件 {pdf_path} 时出错: {str(e)}")
    else:
        print("JSON文件中未找到论文标题列表")
        return None
    
    if not all_docs:
        print("未成功处理任何PDF文件")
        return None
    
    # 切分文本
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = splitter.split_documents(all_docs)
    
    # 构建中文向量模型（bge）
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh")
    
    # 创建向量库
    pdf_db = FAISS.from_documents(documents, embedding)
    
    # 保存向量库到本地
    pdf_db.save_local(vector_db_path)
    print(f"论文向量库已保存到 {vector_db_path}")
    
    return pdf_db


if __name__ == "__main__":
    preprocess_json_papers("data/results/0.json")