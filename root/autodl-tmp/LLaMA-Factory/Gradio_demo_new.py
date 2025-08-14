import os
import re
import json
import gradio as gr
import requests
from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalRAGSystem:
    def __init__(self, data_path: str, api_port: int = 8000):
        self.data_path = data_path
        self.api_port = api_port
        self.api_url = f"http://localhost:{api_port}"
        self.embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        self.documents = []
        self.embeddings = None
        self.index = None
        self.metadata = []  # 存储文档元数据，包括文件名和章节信息
        
        # Token限制设置
        self.max_context_length = 2000
        self.max_doc_length = 500
        self.max_tokens = 1000
        
        # 初始化系统
        self.load_documents()
        self.create_vector_store()

    def truncate_text(self, text: str, max_length: int) -> str:
        """截断文本到指定长度"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def load_documents(self):
        """加载医学教材文档"""
        logger.info("开始加载医学文档...")
        txt_files = list(Path(self.data_path).glob("*.txt"))
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                chapters = self.split_by_chapters(content, file_path.name)
                self.documents.extend(chapters)
            except Exception as e:
                logger.error(f"加载文件 {file_path} 时出错: {e}")
        
        logger.info(f"成功加载 {len(self.documents)} 个文档片段")

    def split_by_chapters(self, content: str, filename: str) -> List[Dict]:
        """按章节分割文档内容"""
        chapters = []
        
        # 使用正则表达式匹配章节标题
        chapter_pattern = r'第[一二三四五六七八九十\d]+章|第[一二三四五六七八九十\d]+节|Chapter\s*\d+|Section\s*\d+'
        
        # 分割章节
        parts = re.split(chapter_pattern, content)
        chapter_titles = re.findall(chapter_pattern, content)
        
        # 处理第一部分（可能没有章节标题）
        if parts[0].strip():
            chapters.append({
                'content': self.truncate_text(parts[0].strip(), self.max_doc_length),
                'filename': filename,
                'chapter': '前言或介绍',
                'chapter_index': 0
            })
        
        # 处理其他章节
        for i, (title, content_part) in enumerate(zip(chapter_titles, parts[1:]), 1):
            if content_part.strip():
                # 进一步分割成段落
                paragraphs = [p.strip() for p in content_part.split('\n\n') if p.strip()]
                for j, paragraph in enumerate(paragraphs):
                    if len(paragraph) > 50:  # 过滤太短的段落
                        chapters.append({
                            'content': self.truncate_text(paragraph, self.max_doc_length),
                            'filename': filename,
                            'chapter': title,
                            'chapter_index': i,
                            'paragraph_index': j
                        })
        
        return chapters

    def create_vector_store(self):
        """创建向量存储"""
        if not self.documents:
            logger.error("没有文档可以创建向量存储")
            return
        
        logger.info("开始创建向量存储...")
        
        # 提取文本内容
        texts = [doc['content'] for doc in self.documents]
        self.metadata = [
            {
                'filename': doc['filename'],
                'chapter': doc['chapter'],
                'chapter_index': doc.get('chapter_index', 0),
                'paragraph_index': doc.get('paragraph_index', 0)
            }
            for doc in self.documents
        ]
        
        # 生成embeddings
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # 创建FAISS索引
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # 标准化embeddings以便使用内积计算余弦相似度
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info("向量存储创建完成")

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """检索相关文档"""
        if self.index is None:
            return []
        
        # 编码查询
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx]['content'],
                    self.metadata[idx],
                    float(score)
                ))
        
        return results

    def build_context_with_citations(self, retrieved_docs: List[Tuple[str, Dict, float]]) -> Tuple[str, str]:
        """构建上下文和引用信息，确保不超过长度限制"""
        context_parts = []
        citation_parts = []
        current_length = 0
        
        for i, (content, metadata, score) in enumerate(retrieved_docs, 1):
            doc_text = f"资料{i}（来源：{metadata['filename']} - {metadata['chapter']}）：{content}"
            
            # 检查添加这个文档是否会超过限制
            if current_length + len(doc_text) > self.max_context_length:
                # 如果超过限制，截断最后一个文档
                remaining_length = self.max_context_length - current_length
                if remaining_length > 100:  # 至少保留100个字符
                    truncated_text = f"资料{i}（来源：{metadata['filename']} - {metadata['chapter']}）：{content[:remaining_length-50]}..."
                    context_parts.append(truncated_text)
                    # 添加截断的引用信息
                    citation_parts.append(
                        f"**检索依据 {i}：**\n"
                        f"- 文件：{metadata['filename']}\n"
                        f"- 章节：{metadata['chapter']}\n"
                        f"- 相似度：{score:.3f}\n"
                        f"- 内容片段：{content[:100]}...\n"
                    )
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
            
            # 添加完整的引用信息
            citation_parts.append(
                f"**检索依据 {i}：**\n"
                f"- 文件：{metadata['filename']}\n"
                f"- 章节：{metadata['chapter']}\n"
                f"- 相似度：{score:.3f}\n"
                f"- 内容片段：{content[:100]}...\n"
            )
        
        context = "\n\n".join(context_parts)
        citations = "\n".join(citation_parts)
        
        return context, citations

    def test_api_connection(self):
        """测试API连接"""
        try:
            test_response = requests.get(f"{self.api_url}/v1/models", timeout=5)
            logger.info(f"API健康检查状态码: {test_response.status_code}")
            return test_response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"API连接测试失败: {e}")
            return False

    def generate_mcq_response(self, question: str, options: Dict[str, str], context: str) -> str:
        """调用LLM生成选择题回答"""
        # 构建选择题的prompt
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        
        prompt = f"""你是一名医学博士，请根据用户的提问给出专业答复，并给出答复依据。

题目：{question}

选项：
{options_text}

参考资料：
{context}

请分析题目和选项，基于提供的参考资料给出正确答案，并详细解释你的推理过程。请按以下格式回答：

答案：[选择的选项字母]

解析：[详细的解释和推理过程]

依据：[基于参考资料的具体依据]
"""
        
        # 先测试API连接
        if not self.test_api_connection():
            return "❌ 无法连接到LLM API服务，请确保API服务正在运行"
        
        try:
            payload = {
                "model": "test",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,  # 降低温度以获得更确定的答案
                "max_tokens": self.max_tokens
            }
            
            logger.info(f"发送API请求，prompt长度: {len(prompt)} 字符")
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            
            logger.info(f"API响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    choice = result['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        return choice['message']['content']
                    elif 'text' in choice:
                        return choice['text']
                return f"✅ API调用成功但响应格式未知: {str(result)[:200]}..."
            else:
                error_text = response.text[:500] if response.text else "无错误信息"
                return f"❌ API调用失败，状态码: {response.status_code}，错误: {error_text}"
        
        except requests.exceptions.Timeout:
            return "❌ API调用超时，请检查网络连接"
        except requests.exceptions.RequestException as e:
            return f"❌ API请求失败: {str(e)}"
        except json.JSONDecodeError as e:
            return f"❌ 响应JSON解析失败: {str(e)}"
        except Exception as e:
            return f"❌ 生成回答时出错: {str(e)}"

    def generate_text_response(self, query: str, context: str) -> str:
        """调用LLM生成文本回答"""
        # 构建更简洁的prompt
        prompt = f"""基于以下医学资料回答问题：

{context}

问题：{query}

请提供简洁的医学回答："""
        
        # 先测试API连接
        if not self.test_api_connection():
            return "❌ 无法连接到LLM API服务，请确保API服务正在运行"
        
        try:
            payload = {
                "model": "test",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,
                "max_tokens": self.max_tokens
            }
            
            logger.info(f"发送API请求，prompt长度: {len(prompt)} 字符")
            
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            
            logger.info(f"API响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    choice = result['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        return choice['message']['content']
                    elif 'text' in choice:
                        return choice['text']
                
                return f"✅ API调用成功但响应格式未知: {str(result)[:200]}..."
            else:
                error_text = response.text[:500] if response.text else "无错误信息"
                return f"❌ API调用失败，状态码: {response.status_code}，错误: {error_text}"
                
        except requests.exceptions.Timeout:
            return "❌ API调用超时，请检查网络连接"
        except requests.exceptions.RequestException as e:
            return f"❌ API请求失败: {str(e)}"
        except json.JSONDecodeError as e:
            return f"❌ 响应JSON解析失败: {str(e)}"
        except Exception as e:
            return f"❌ 生成回答时出错: {str(e)}"

    def answer_mcq_question(self, question_json: str) -> Tuple[str, str]:
        """处理选择题查询"""
        try:
            # 解析JSON输入
            if isinstance(question_json, str):
                question_data = json.loads(question_json)
            else:
                question_data = question_json
            
            question = question_data.get('question', '')
            options = question_data.get('options', {})
            
            if not question.strip():
                return "请输入完整的题目信息。", ""
            
            if not options:
                return "请提供选择题的选项。", ""
            
            # 构建查询文本（包含问题和选项）
            query_text = question + " " + " ".join(options.values())
            
            # 检索相关文档
            retrieved_docs = self.retrieve_documents(query_text, top_k=5)
            
            if not retrieved_docs:
                return "抱歉，没有找到相关的医学资料。", ""
            
            # 构建上下文和引用信息
            context, citations = self.build_context_with_citations(retrieved_docs)
            
            # 生成回答
            answer = self.generate_mcq_response(question, options, context)
            
            return answer, citations
        
        except json.JSONDecodeError:
            return "输入格式错误，请输入正确的JSON格式。", ""
        except Exception as e:
            logger.error(f"处理问题时出错: {e}")
            return f"处理问题时出错: {str(e)}", ""

    def answer_text_query(self, query: str) -> Tuple[str, str]:
        """处理文本查询"""
        if not query.strip():
            return "请输入您的医学问题。", ""
        
        # 检索相关文档
        retrieved_docs = self.retrieve_documents(query, top_k=3)
        
        if not retrieved_docs:
            return "抱歉，没有找到相关的医学资料。", ""
        
        # 构建上下文和引用信息
        context, citations = self.build_context_with_citations(retrieved_docs)
        
        # 生成回答
        answer = self.generate_text_response(query, context)
        
        return answer, citations

# 全局RAG系统实例
rag_system = None

def initialize_rag_system():
    """初始化RAG系统"""
    global rag_system
    try:
        rag_system = MedicalRAGSystem(
            data_path="/root/autodl-tmp/LLaMA-Factory/2025_05_22_med_data_zh_paragraph",
            api_port=8000
        )
        return "✅ RAG系统初始化成功！"
    except Exception as e:
        logger.error(f"初始化RAG系统失败: {e}")
        return f"❌ RAG系统初始化失败: {str(e)}"

def process_text_query(query: str, history: List[List[str]]) -> Tuple[List[List[str]], str]:
    """处理文本医学问题"""
    if rag_system is None:
        return history + [[query, "❌ 请先初始化RAG系统"]], ""
    
    try:
        answer, citations = rag_system.answer_text_query(query)
        new_history = history + [[query, answer]]
        return new_history, citations
    except Exception as e:
        error_msg = f"处理查询时出错: {str(e)}"
        logger.error(error_msg)
        return history + [[query, error_msg]], ""

def process_mcq_question(question_input: str) -> Tuple[str, str]:
    """处理医学选择题"""
    if rag_system is None:
        return "❌ 请先初始化RAG系统", ""
    
    try:
        answer, citations = rag_system.answer_mcq_question(question_input)
        return answer, citations
    except Exception as e:
        error_msg = f"处理查询时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""

def clear_text_conversation():
    """清除文本对话历史"""
    return [], ""

def clear_mcq_fields():
    """清除选择题字段"""
    return "", "", ""

def create_gradio_interface():
    """创建Gradio用户界面"""
    with gr.Blocks(title="医学RAG问答系统", theme=gr.themes.Soft()) as interface:
        gr.HTML("""
            <h1 style='text-align: center;'>🏥 医学RAG问答系统</h1>
            <p style='text-align: center;'>基于33个医学教材的专业问答系统 - 支持文本问答和选择题</p>
        """)
        
        # 系统初始化区域
        with gr.Row():
            with gr.Column(scale=4):
                init_btn = gr.Button("🚀 初始化系统", variant="primary", size="lg")
            with gr.Column(scale=2):
                status_output = gr.Textbox(
                    label="📊 系统状态",
                    value="请点击'初始化系统'开始使用",
                    interactive=False,
                    lines=1
                )
        
        # 创建两个标签页
        with gr.Tabs():
            # 文本问答标签页
            with gr.TabItem("💬 文本医学问答"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_chatbot = gr.Chatbot(
                            label="医学问答对话",
                            show_label=True,
                            height=500
                        )
                        
                        with gr.Row():
                            text_query_input = gr.Textbox(
                                placeholder="请输入您的医学问题...",
                                label="问题输入",
                                scale=4
                            )
                            text_submit_btn = gr.Button("🔍 提问", variant="primary", scale=1)
                        
                        text_clear_btn = gr.Button("🗑️ 清除对话", variant="secondary")
                    
                    with gr.Column(scale=2):
                        text_citations_output = gr.Markdown(
                            label="📚 检索依据",
                            value="",
                            height=500
                        )
            
            # 选择题问答标签页
            with gr.TabItem("📝 医学选择题"):
                with gr.Row():
                    with gr.Column(scale=3):
                        # 输入区域
                        with gr.Group():
                            gr.Markdown("### 题目输入")
                            mcq_question_input = gr.Textbox(
                                placeholder='请输入JSON格式的选择题，例如：\n{"question": "经调查证实出现医院感染流行时，医院应报告当地卫生行政部门的时间是（　　）。", "options": {"A": "2小时", "B": "4小时内", "C": "8小时内", "D": "12小时内", "E": "24小时内"}}',
                                label="选择题输入（JSON格式）",
                                lines=6,
                                max_lines=10
                            )
                            
                            with gr.Row():
                                mcq_submit_btn = gr.Button("🔍 分析题目", variant="primary", scale=4)
                                mcq_clear_btn = gr.Button("🗑️ 清除", variant="secondary", scale=1)
                        
                        # 输出区域
                        with gr.Group():
                            gr.Markdown("### 🤖 AI分析结果")
                            mcq_answer_output = gr.Textbox(
                                label="题目分析和答案",
                                lines=10,
                                max_lines=15,
                                interactive=False
                            )
                    
                    with gr.Column(scale=2):
                        # 检索依据
                        with gr.Group():
                            gr.Markdown("### 📚 检索依据")
                            mcq_citations_output = gr.Markdown(
                                value="",
                                label="检索来源",
                                height=500
                            )
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 功能介绍
            本系统基于33个医学教材，提供两种问答模式：
            
            #### 1. 文本医学问答
            - 自然语言对话形式的医学问答
            - 保持连续的对话历史
            - 显示答案的检索依据和内容片段
            
            #### 2. 医学选择题
            - 支持JSON格式的选择题输入
            - 提供详细的题目分析和推理过程
            - 显示答案的检索依据和内容片段
            
            ### 使用步骤
            1. **初始化系统**：点击"初始化系统"按钮，等待系统加载完成
            2. **选择功能**：
               - 文本问答：切换到"文本医学问答"标签页，直接输入问题
               - 选择题：切换到"医学选择题"标签页，输入JSON格式的题目
            3. **查看结果**：系统会显示答案和相应的检索依据
            
            ### 选择题输入格式示例
            ```json
            {
                "question": "经调查证实出现医院感染流行时，医院应报告当地卫生行政部门的时间是（　　）。",
                "options": {
                    "A": "2小时",
                    "B": "4小时内", 
                    "C": "8小时内",
                    "D": "12小时内",
                    "E": "24小时内"
                }
            }
            ```
            
            ### 注意事项
            - 首次初始化可能需要几分钟时间
            - 确保LLM API服务正在运行（端口8000）
            - 选择题输入必须是有效的JSON格式
            - 系统会自动检索相关医学资料并提供专业分析
            """)
        
        # 事件绑定
        # 初始化系统
        init_btn.click(
            fn=initialize_rag_system,
            outputs=status_output
        )
        
        # 文本问答事件
        text_submit_btn.click(
            fn=process_text_query,
            inputs=[text_query_input, text_chatbot],
            outputs=[text_chatbot, text_citations_output]
        ).then(
            fn=lambda: "",
            outputs=text_query_input
        )
        
        text_query_input.submit(
            fn=process_text_query,
            inputs=[text_query_input, text_chatbot],
            outputs=[text_chatbot, text_citations_output]
        ).then(
            fn=lambda: "",
            outputs=text_query_input
        )
        
        text_clear_btn.click(
            fn=clear_text_conversation,
            outputs=[text_chatbot, text_citations_output]
        )
        
        # 选择题事件
        mcq_submit_btn.click(
            fn=process_mcq_question,
            inputs=[mcq_question_input],
            outputs=[mcq_answer_output, mcq_citations_output]
        )
        
        mcq_clear_btn.click(
            fn=clear_mcq_fields,
            outputs=[mcq_question_input, mcq_answer_output, mcq_citations_output]
        )
    
    return interface

if __name__ == "__main__":
    # 启动Gradio应用
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )