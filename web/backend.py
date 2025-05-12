import os
import json
import argparse
from models      import Agent
from ..paper_agent import PaperAgent
from datetime    import datetime, timedelta
from fastapi import FastAPI
from typing import List, Dict
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    return paper_agent.root.extra["recall_papers_id"]
    
@app.post("/generate")
async def generate_response(request: dict) -> dict:
    """
    接收查询请求，返回生成的回复
    请求格式: {
        "query": str,
        "context": str,
        "prompt": str
    }
    返回格式: {
        "response": str
    }
    """
    query = request.get("query", "")
    context = request.get("context", "")
    prompt = request.get("prompt", "")
    
    # 如果用户没有提供 prompt，设置默认的论文综述提示
    if not prompt:
        prompt = """请基于以上论文片段，编写一篇与查询主题相关的论文综述。综述应该：
1. 概述研究现状和主要进展
2. 分析不同方法的优缺点
3. 总结关键发现和结论
4. 指出未来可能的研究方向"""
    
    # 构建完整的 prompt
    system_prompt = """你是一个学术助手，请基于给定的论文片段，回答用户的学术问题。
    回答时请：
    1. 保持客观准确
    2. 适当引用论文内容
    3. 条理清晰地组织回答
    4. 如果论文片段中没有相关信息，请明确指出"""
    
    full_prompt = f"""用户问题：{query}

用户提示：{prompt}

相关论文片段：
{context}

请基于以上信息生成回答。"""

    try:
        # 调用 OpenAI API
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=4096
        )
        
        # 获取生成的回答
        generated_response = response.choices[0].message.content
        
        return {"response": generated_response}
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return {"response": "抱歉，生成回答时发生错误。", "error": str(e)}

        