# streamlit run d:\code\myRAG\src\demo_web.py
# Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.
import streamlit as st
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from down_PDF import download_pdf_by_title, find_arxiv_id_by_title, download_pdf_by_arxiv_id
from preprocess import preprocess_pdf, get_embedding_model
import requests
from preprocess import get_embedding_model
# 添加导入
import hashlib
import concurrent.futures

# 设置页面标题
st.set_page_config(page_title="论文检索系统", layout="wide")

# 设置页面样式
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .paper-item {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    .paper-title {
        font-weight: bold;
        font-size: 1.1rem;
    }
    .paper-abstract {
        font-size: 0.9rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# 移除这里的重复标题显示
# st.title("论文检索系统")

# 初始化会话状态
if 'papers' not in st.session_state:
    st.session_state.papers = []
if 'selected_papers' not in st.session_state:
    st.session_state.selected_papers = []

# 加载向量库
@st.cache_resource
def load_vector_db(db_path="data/papers_faiss_db"):
    if os.path.exists(db_path):
        # 使用单例模式获取embedding模型
        embedding = get_embedding_model()
        return FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
    return None

# 搜索论文
def search_papers(query, prompt="", top_k=5):
    db = load_vector_db()
    if db is None:
        st.error("向量库未找到，请先创建向量库")
        return []
    
    # 如果有提示词，将其与查询组合
    if prompt:
        query = f"{prompt}: {query}"
    
    # 执行搜索
    results = db.similarity_search_with_score(query, k=top_k)
    
    # 处理结果
    papers = []
    for doc, score in results:
        title = doc.metadata.get("title", "未知标题")
        arxiv_id = doc.metadata.get("arxiv_id", "")
        
        # 如果没有arxiv_id，尝试通过标题查找
        if not arxiv_id:
            arxiv_id = find_arxiv_id_by_title(title)
        
        papers.append({
            "title": title,
            "arxiv_id": arxiv_id,
            "abstract": doc.page_content,
            "score": score,
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
        })
    
    return papers

# 下载选定的论文
def download_selected_papers(selected_papers):
    download_dir = "data/web_pdf"
    os.makedirs(download_dir, exist_ok=True)
    
    success_count = 0
    for paper in selected_papers:
        title = paper["title"]
        arxiv_id = paper["arxiv_id"]
        
        if arxiv_id:
            # 使用arxiv_id下载
            from down_PDF import download_pdf_by_arxiv_id
            save_path = os.path.join(download_dir, f"{arxiv_id}.pdf")
            download_pdf_by_arxiv_id(arxiv_id, save_path)
            success_count += 1
        else:
            # 使用标题下载
            result = download_pdf_by_title(title, download_dir)
            if result:
                success_count += 1
    
    return success_count

# 创建侧边栏
def get_llm_summary(content: str, prompt: str = "") -> str:
    """
    调用本地后端服务进行总结
    
    Args:
        content: 需要总结的内容
        prompt: 提示词
    
    Returns:
        str: 总结结果
    """
    if not prompt:
        prompt = "请总结以下论文的主要内容，包括研究目的、方法和结论："
    
    try:
        # 调用本地后端服务
        response = requests.post(
            "http://localhost:8080/summarize",
            json={
                "content": content,
                "prompt": prompt
            }
        )
        response.raise_for_status()
        return response.json()["summary"]
    except Exception as e:
        return f"生成总结时出错: {str(e)}"

def process_papers_to_vectordb(arxiv_ids, pdf_dir="data/web_pdf", vector_db_path="data/web_pdf_faiss_db"):
    """下载论文并创建向量库"""
    os.makedirs(pdf_dir, exist_ok=True)
    downloaded_paths = []
    
    # 下载所有论文
    for arxiv_id in arxiv_ids:
        pdf_path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
        download_pdf_by_arxiv_id(arxiv_id, pdf_path)
        if os.path.exists(pdf_path):
            downloaded_paths.append(pdf_path)
    
    # 创建向量库
    if downloaded_paths:
        return preprocess_pdf(pdf_dir, vector_db_path)
    return None

def get_paper_info(arxiv_id):
    """获取论文信息"""
    try:
        import arxiv
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        return {
            "title": paper.title,
            "abstract": paper.summary,
            "url": f"https://arxiv.org/abs/{arxiv_id}"
        }
    except Exception as e:
        st.error(f"获取论文信息失败: {str(e)}")
        return None

def query_local_model(query):
    """调用本地大模型API"""
    try:
        response = requests.post(
            "http://localhost:8000/process_user_query",
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()["arxiv_ids"]
    except Exception as e:
        st.error(f"调用本地模型失败: {str(e)}")
        return []

def get_llm_response(query, context, prompt=""):
    """调用本地大模型进行总结或问答"""
    try:
        response = requests.post(
            "http://localhost:8000/generate",
            json={
                "query": query,
                "context": context,
                "prompt": prompt
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        st.error(f"调用本地模型失败: {str(e)}")
        return ""

def get_query_hash(query: str) -> str:
    """生成查询的哈希值作为向量库标识"""
    return hashlib.md5(query.encode()).hexdigest()

def get_vector_db_path(query: str) -> str:
    """获取特定查询对应的向量库路径"""
    query_hash = get_query_hash(query)
    return os.path.join("data", "query_vector_dbs", query_hash)

def search_papers(query):
    # 调用本地模型获取arxiv_ids
    arxiv_ids = query_local_model(query)
    if not arxiv_ids:
        return []
    
    # 获取论文信息
    papers = []
    with st.spinner("正在搜索相关论文..."):  # 使用spinner保持搜索状态
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_arxiv_id = {executor.submit(get_paper_info, arxiv_id): arxiv_id for arxiv_id in arxiv_ids}
            for future in concurrent.futures.as_completed(future_to_arxiv_id):
                arxiv_id = future_to_arxiv_id[future]
                try:
                    paper_info = future.result()
                    if paper_info:
                        paper_info["arxiv_id"] = arxiv_id
                        papers.append(paper_info)
                except Exception as e:
                    st.error(f"获取论文信息失败: {str(e)}")
    
    return papers

# 添加向量库搜索和总结功能
# 删除不再需要的函数
# 删除 get_query_hash, get_vector_db_path, process_papers_to_vectordb 函数

def search_and_summarize(query, prompt=""):
    """调用后端生成总结"""
    try:
        # 获取当前会话中的论文信息
        arxiv_ids = [paper["arxiv_id"] for paper in st.session_state.papers if paper.get("arxiv_id")]
        
        response = requests.post(
            "http://localhost:8000/generate",
            json={
                "query": query,
                "prompt": prompt,
                "arxiv_ids": arxiv_ids  # 添加arxiv_ids到请求中
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        st.error(f"调用后端服务失败: {str(e)}")
        return ""

# 修改主界面部分
def main():
    st.title("论文检索与分析系统")
    
    # 初始化会话状态
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'summary_text' not in st.session_state:
        st.session_state.summary_text = ""
    
    # 侧边栏设置
    with st.sidebar:
        st.header("设置")
        summary_prompt = st.text_area(
            "总结提示词",
            value="请总结这些论文的主要研究内容，包括研究方向、创新点和实验结果。",
            height=100
        )
        
        st.markdown("**论文总结**")  # 加粗“论文总结”
        # 使用markdown渲染总结内容
        st.markdown("""
            <div style="border:1px solid #ccc; border-radius:5px; padding:10px; height:300px; overflow-y:auto;">
                {}
            </div>
            """.format(st.session_state.summary_text), unsafe_allow_html=True)
        
        if st.button("生成总结"):
            if st.session_state.papers:
                query = st.session_state.get('query', '')  # Ensure query is defined
                with st.spinner("正在生成总结..."):
                    summary = search_and_summarize(query, summary_prompt)
                    st.session_state.summary_text = summary
                    st.rerun()
            else:
                st.warning("请先搜索论文")
    
    # 主界面搜索区域
    col_query, _ = st.columns([2, 1])  # 主界面占2/3，侧栏占1/3
    
    with col_query:
        st.markdown("**输入查询**")  # 加粗“输入查询”
        query = st.text_area("", placeholder="请输入您的研究问题", height=200)  # 移除标签文本，保持对齐
        st.session_state.query = query  # Store query in session state
        if st.button("搜索论文"):
            if query:
                papers = search_papers(query)
                st.session_state.papers = papers
            else:
                st.warning("请输入查询内容")
    
    # 显示论文列表
    if st.session_state.papers:
        st.header(f"搜索结果 ({len(st.session_state.papers)}篇)")
        for paper in st.session_state.papers:
            with st.container():
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.markdown(f"### [{paper['title']}]({paper['url']})")
                    st.text(f"arXiv ID: {paper['arxiv_id']}")
                    with st.expander("显示摘要"):
                        st.markdown(paper['abstract'])
                with col2:
                    # 获取PDF内容
                    try:
                        response = requests.get(f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf")
                        if response.status_code == 200:
                            st.download_button(
                                label="下载",
                                data=response.content,
                                file_name=f"{paper['arxiv_id']}.pdf",
                                mime="application/pdf",
                                key=f"download_{paper['arxiv_id']}"
                            )
                    except Exception as e:
                        st.error("下载失败")

if __name__ == "__main__":
    main()