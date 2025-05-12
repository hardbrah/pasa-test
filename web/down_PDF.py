import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import SSLError
import urllib.parse
import arxiv


def find_arxiv_id_by_title(title: str) -> str:
    """
    根据论文标题查找 arXiv ID
    
    Args:
        title (str): 论文标题
    
    Returns:
        str: 找到的 arXiv ID，如果未找到则返回空字符串
    """
    try:
        # 使用arxiv库直接搜索标题
        search = arxiv.Search(
            query=f'ti:"{title}"',
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # 获取搜索结果
        results = list(search.results())
        
        if results:
            # 返回第一个结果的ID
            return results[0].get_short_id()
        else:
            print(f"未找到标题为 '{title}' 的论文")
            return ""
            
    except Exception as e:
        print(f"查找 arXiv ID 时出错: {str(e)}")
        return ""


def download_pdf_by_arxiv_id(arxiv_id: str, save_path: str) -> None:
    """
    Download a PDF from arXiv using the arXiv ID and save it to the specified path.

    Args:
        arxiv_id (str): The arXiv ID of the paper to download.
        save_path (str): The path where the PDF will be saved.
    """


    # Create a session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    # Construct the URL for the PDF
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        # Send a GET request to download the PDF with SSL verification disabled
        response = session.get(url, verify=False)
        response.raise_for_status()

        # Save the PDF to the specified path
        with open(save_path, "wb") as pdf_file:
            pdf_file.write(response.content)
        print(f"PDF downloaded successfully and saved to {save_path}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {str(e)}")
    finally:
        session.close()


def download_pdf_by_title(title: str, save_dir: str = "data/pdf", filename: str = None) -> str:
    """
    根据论文标题查找并下载PDF文件
    
    Args:
        title (str): 论文标题
        save_dir (str): 保存PDF的目录
        filename (str): 自定义文件名，如果为None则使用arXiv ID作为文件名
    
    Returns:
        str: 保存的PDF文件路径，如果下载失败则返回空字符串
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 查找arXiv ID
    arxiv_id = find_arxiv_id_by_title(title)
    
    if not arxiv_id:
        print(f"无法下载论文: '{title}'，未找到对应的arXiv ID")
        return ""
    
    # 确定保存路径
    if filename:
        # 确保文件名以.pdf结尾
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        save_path = os.path.join(save_dir, filename)
    else:
        save_path = os.path.join(save_dir, f"{arxiv_id}.pdf")
    
    # 下载PDF
    download_pdf_by_arxiv_id(arxiv_id, save_path)
    
    # 检查文件是否成功下载
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return save_path
    else:
        return ""


if __name__ == "__main__":
    out_dir = "data/pdf"
    os.makedirs(out_dir, exist_ok=True)
    
    # 使用新函数下载PDF
    pdf_path = download_pdf_by_title("A Survey on Large Language Models", out_dir)
    if pdf_path:
        print(f"论文已成功下载到: {pdf_path}")