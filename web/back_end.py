# python -m uvicorn src.mock_local_model:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI
from typing import List, Dict
from openai import OpenAI
import os
import concurrent.futures
from dotenv import load_dotenv
import hashlib
from langchain_community.vectorstores import FAISS
import logging
from .preprocess import get_embedding_model, preprocess_pdf  # Changed to relative import
from .down_PDF import download_pdf_by_arxiv_id  # Also update this import
import os
import json
import argparse
from models      import Agent
from ..paper_agent import PaperAgent
from datetime    import datetime, timedelta
from fastapi import FastAPI
from typing import List, Dict
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument('--input_file',     type=str, default="data/RealScholarQuery/test.jsonl")
parser.add_argument('--crawler_path',   type=str, default="checkpoints/pasa-7b-crawler")
parser.add_argument('--selector_path',  type=str, default="checkpoints/pasa-7b-selector")
parser.add_argument('--output_folder',  type=str, default="results")
parser.add_argument('--expand_layers',  type=int, default=2)
parser.add_argument('--search_queries', type=int, default=5)
parser.add_argument('--search_papers',  type=int, default=10)
parser.add_argument('--expand_papers',  type=int, default=20)
parser.add_argument('--threads_num',    type=int, default=20)
args = parser.parse_args()

crawler = Agent(args.crawler_path)
selector = Agent(args.selector_path)

# 加载环境变量
load_dotenv()

# 记录生成的回答到日志
logging.basicConfig(
    filename='generation_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    encoding='utf-8'
)

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DS_API_KEY"),
    base_url=os.getenv("DS_API_BASE_URL")
)

app = FastAPI()

@app.post("/process_user_query")
async def process_user_query(request: Dict) -> Dict:
    """
    处理用户查询，返回相关的arxiv ID列表
    请求格式: {
        "query": str
    }
    返回格式: {
        "arxiv_ids": List[str]
    }
    """
    query = request.get("query", "").lower()

    paper_agent = PaperAgent(
            user_query     = query, 
            crawler        = crawler,
            selector       = selector,
            end_date       = datetime.now().strftime("%Y-%m-%d"),
            expand_layers  = args.expand_layers,
            search_queries = args.search_queries,
            search_papers  = args.search_papers,
            expand_papers  = args.expand_papers,
            threads_num    = args.threads_num
        )

    paper_agent.run()
    
    return {"arxiv_ids": paper_agent.root.extra["recall_papers_id"]}

def get_query_hash(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()

def get_vector_db_path(query: str) -> str:
    query_hash = get_query_hash(query)
    return os.path.join("data", "query_vector_dbs", query_hash)

def process_papers_to_vectordb(arxiv_ids, query):
    """下载论文并创建向量库"""
    query_hash = get_query_hash(query)
    pdf_dir = os.path.join("data", "web_pdf", query_hash)  # 使用查询的哈希值作为文件夹名
    vector_db_path = get_vector_db_path(query)
    
    # 检查向量库是否存在
    if os.path.exists(vector_db_path):
        return FAISS.load_local(vector_db_path, get_embedding_model(), allow_dangerous_deserialization=True)
    
    # 如果PDF目录存在，说明已经下载过，直接返回
    if os.path.exists(pdf_dir):
        logging.info(f"PDF目录 {pdf_dir} 已存在，跳过下载。")
        return preprocess_pdf(pdf_dir, vector_db_path)
    
    # 创建新的向量库和PDF目录
    os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    # 下载论文并处理
    downloaded_paths = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_arxiv_id = {executor.submit(download_pdf_by_arxiv_id, arxiv_id, os.path.join(pdf_dir, f"{arxiv_id}.pdf")): arxiv_id for arxiv_id in arxiv_ids}
        for future in concurrent.futures.as_completed(future_to_arxiv_id):
            arxiv_id = future_to_arxiv_id[future]
            try:
                future.result()  # 等待线程完成
                pdf_path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
                if os.path.exists(pdf_path):
                    downloaded_paths.append(pdf_path)
            except Exception as e:
                logging.error(f"下载论文 {arxiv_id} 时出错: {str(e)}")
    
    if downloaded_paths:
        return preprocess_pdf(pdf_dir, vector_db_path)
    return None

@app.post("/generate")
async def generate_response(request: dict) -> dict:
    """接收查询请求，返回生成的回复"""
    query = request.get("query", "")
    prompt = request.get("prompt", "")
    arxiv_ids = request.get("arxiv_ids", [])  # 从请求中获取arxiv_ids
    
    if not arxiv_ids:
        return {"response": "未找到相关论文"}
    
    # 处理向量库
    vector_db = process_papers_to_vectordb(arxiv_ids, query)
    if not vector_db:
        return {"response": "创建向量库失败"}
    
    # 在向量库中搜索相关内容
    results = vector_db.similarity_search(query, k=3*len(arxiv_ids))
    context = "\n".join([doc.page_content for doc in results])
    
    # 构建完整的 prompt
    system_prompt = """你是一个学术助手，请基于给定的论文片段，回答用户的学术问题。
    回答时请：
    1. 保持客观准确
    2. 适当引用论文内容
    3. 条理清晰地组织回答
    4. 如果论文片段中没有相关信息，请明确指出"""
    
    full_prompt = f"""用户问题：{query}
    用户提示：{prompt}
    相关论文片段：{context}
    请基于以上信息生成回答。"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("DP_V3_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=4096
        )
        
        logging.info(f"Query: {query}")
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Context: {context}")
        logging.info(f"Generated Response: {response.choices[0].message.content}")
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": f"生成回答时发生错误: {str(e)}"}