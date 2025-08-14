import torch
import json
import re
import os
import time
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import logging
from transformers.utils.versions import require_version

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

# 配置更详细的日志记录
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_evaluation.log'),
        logging.StreamHandler()
    ]
)

# 检查GPU可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logging.warning("GPU不可用，使用CPU。")

# 下载NLTK的句子分隔器模型
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class RAGEvaluator:
    def __init__(self, documents_dir, api_port=8000, debug_mode=True):
        """
        初始化RAG评估器
        
        Args:
            documents_dir: 文档目录路径
            api_port: API端口号
            debug_mode: 是否开启调试模式
        """
        self.documents_dir = documents_dir
        self.api_port = api_port
        self.debug_mode = debug_mode
        self.embedder = None
        self.documents = []
        
        # 初始化嵌入模型
        self._init_embedder()
        
        # 加载文档
        self._load_documents()
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key="{}".format(os.environ.get("API_KEY", "0")),
            base_url=f"http://localhost:{api_port}/v1",
        )
        
        # 测试API连接
        self._test_api_connection()
    
    def _test_api_connection(self):
        """测试API连接"""
        try:
            logging.info("测试API连接...")
            test_response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": "你好，请回答：测试"}],
                model="test",
                max_tokens=50,
                temperature=0.1,
            )
            
            if test_response and test_response.choices:
                response_content = test_response.choices[0].message.content
                logging.info(f"API连接测试成功，响应: {response_content}")
                return True
            else:
                logging.error("API连接测试失败：无响应内容")
                return False
                
        except Exception as e:
            logging.error(f"API连接测试失败: {e}")
            return False
    
    def _init_embedder(self):
        """初始化嵌入模型"""
        try:
            self.embedder = SentenceTransformer('shibing624/text2vec-base-chinese').to(device)
            logging.info("嵌入模型加载成功")
        except Exception as e:
            logging.error(f"加载Sentence-BERT模型时出错: {e}")
            self.embedder = None
    
    def _load_documents(self):
        """加载并分块文档"""
        if not os.path.exists(self.documents_dir):
            logging.error(f"文档目录不存在: {self.documents_dir}")
            return
        
        documents = []
        txt_files = [f for f in os.listdir(self.documents_dir) if f.endswith(".txt")]
        
        if not txt_files:
            logging.error(f"在目录 {self.documents_dir} 中未找到.txt文件")
            return
        
        logging.info(f"找到 {len(txt_files)} 个txt文件")
        
        for filename in txt_files:
            file_path = os.path.join(self.documents_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    # 按句子分块，使用更小的chunk以避免上下文过长
                    sentences = sent_tokenize(text)
                    chunk = []
                    max_chunk_sentences = 10  # 减少每个chunk的句子数
                    
                    for sentence in sentences:
                        chunk.append(sentence)
                        if len(chunk) >= max_chunk_sentences:
                            chunk_text = ' '.join(chunk)
                            # 限制chunk长度
                            if len(chunk_text) <= 1000:  # 限制chunk长度
                                documents.append(chunk_text)
                            chunk = []
                    
                    if chunk:  # 添加最后的chunk
                        chunk_text = ' '.join(chunk)
                        if len(chunk_text) <= 1000:
                            documents.append(chunk_text)
                            
                logging.info(f"成功处理文件: {filename}")
            except Exception as e:
                logging.error(f"处理文件 {filename} 时出错: {e}")
        
        self.documents = documents
        logging.info(f"总共加载了 {len(self.documents)} 个文档块")
    
    def retrieve_relevant_chunks(self, query, top_k=3):
        """检索相关文档块"""
        if self.embedder is None or not self.documents:
            logging.error("嵌入模型未加载或文档为空，无法进行检索。")
            return ""
        
        try:
            # 向量化查询和文档
            query_embedding = self.embedder.encode([query], convert_to_tensor=True, device=device)
            doc_embeddings = self.embedder.encode(self.documents, convert_to_tensor=True, device=device)
            
            # 计算余弦相似度
            cos_scores = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings, dim=1)
            
            # 将张量移动到CPU并转换为列表
            cos_scores = cos_scores.squeeze().cpu().tolist()
            
            # 获取最相关的文档块
            top_indices = np.argsort(cos_scores)[::-1][:top_k]
            relevant_chunks = [self.documents[i] for i in top_indices]
            
            # 去重并组合上下文，限制总长度
            unique_chunks = []
            seen = set()
            total_length = 0
            max_context_length = 2000  # 限制上下文总长度
            
            for chunk in relevant_chunks:
                if chunk not in seen and total_length + len(chunk) <= max_context_length:
                    unique_chunks.append(chunk)
                    seen.add(chunk)
                    total_length += len(chunk)
            
            context = "\n\n".join(unique_chunks)
            return context
            
        except Exception as e:
            logging.error(f"检索相关文档块时出错: {e}")
            return ""
    
    def request_api(self, prompt, context):
        """调用API获取预测结果"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # 构建包含上下文的完整prompt，限制长度
                if context.strip():
                    full_prompt = f"基于以下医学知识背景回答问题：\n\n【医学知识背景】\n{context[:1500]}\n\n【问题】\n{prompt}"
                else:
                    full_prompt = prompt
                
                # 限制prompt总长度
                if len(full_prompt) > 3000:
                    full_prompt = full_prompt[:3000] + "..."
                
                if self.debug_mode and attempt == 0:
                    logging.info(f"发送请求，prompt长度: {len(full_prompt)}")
                
                messages = [{"role": "user", "content": full_prompt}]
                
                # 调用API
                result = self.client.chat.completions.create(
                    messages=messages,
                    model="test",
                    max_tokens=256,  # 减少max_tokens
                    temperature=0.1,
                    top_p=0.95,
                )
                
                # 获取回复
                if result and result.choices:
                    response = result.choices[0].message.content
                    if self.debug_mode and attempt == 0:
                        logging.info(f"API响应成功，长度: {len(response) if response else 0}")
                    return response if response else ""
                else:
                    logging.warning(f"API返回空结果，尝试 {attempt + 1}/{max_retries}")
                    
            except Exception as e:
                logging.error(f"调用API时出错 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    
        return ""
    
    def extract_answer(self, response):
        """提取预测答案"""
        if not response or response.strip() == "":
            return None
        
        response = response.strip()
        
        # 多种答案提取模式
        patterns = [
            r'答案[:：是为]\s*([A-E])',
            r'选择\s*([A-E])',
            r'([A-E])\s*[是为]正确',
            r'正确答案[是为]\s*([A-E])',
            r'\b([A-E])\b[.。,，)]',
            r'^([A-E])[.。,，\s]',
            r'([A-E])(?:\s*[.。,，]|\s*$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                if answer in ['A', 'B', 'C', 'D', 'E']:
                    return answer
        
        # 如果没有匹配到，查找任何A-E字母
        letters = re.findall(r'\b([A-E])\b', response, re.IGNORECASE)
        if letters:
            return letters[0].upper()
        
        return None
    
    def evaluate_test_set(self, test_path, save_results=True):
        """评估测试集"""
        # 加载测试数据
        test_data = []
        try:
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line))
            logging.info(f"加载了 {len(test_data)} 条测试数据")
        except Exception as e:
            logging.error(f"加载测试数据时出错: {e}")
            return None
        
        # 评估过程
        logging.info("开始RAG模型评估...")
        correct_count = 0
        total_count = 0
        results = []
        api_error_count = 0
        
        for item in tqdm(test_data, desc="评估进度"):
            total_count += 1
            
            question = item["question"]
            options = item["options"]
            correct_answer = item["answer"]
            correct_idx = item["answer_idx"]
            
            # 格式化选项
            options_text = ""
            for key, value in options.items():
                options_text += f"{key}. {value}\n"
            
            # 创建prompt
            prompt = f"请回答以下医学选择题，只需给出正确选项字母（A/B/C/D/E）：\n\n问题：{question}\n选项：\n{options_text}\n答案："
            
            # 使用RAG检索相关上下文
            context = self.retrieve_relevant_chunks(question, top_k=2)
            
            # 获取模型预测
            response = self.request_api(prompt, context)
            
            if not response:
                api_error_count += 1
                
            # 提取预测答案
            predicted_answer = self.extract_answer(response)
            
            # 检查预测是否正确
            is_correct = predicted_answer == correct_idx if predicted_answer else False
            
            if is_correct:
                correct_count += 1
            
            # 保存结果
            result = {
                "question_id": total_count,
                "question": question,
                "options": options,
                "correct_answer": correct_idx,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": response,
                "retrieved_context_length": len(context) if context else 0
            }
            results.append(result)
            
            # 调试模式下输出详细信息
            if self.debug_mode and total_count <= 3:
                logging.info(f"问题 {total_count}: 预测={predicted_answer}, 正确={correct_idx}, 响应长度={len(response)}")
            
            # 每10个样本输出一次进度
            if total_count % 10 == 0:
                current_accuracy = correct_count / total_count
                logging.info(f"进度: {total_count}/{len(test_data)}, 当前准确率: {current_accuracy:.4f} ({correct_count}/{total_count}), API错误: {api_error_count}")
        
        # 计算最终准确率
        final_accuracy = correct_count / total_count if total_count > 0 else 0
        logging.info(f"评估完成！")
        logging.info(f"最终准确率: {final_accuracy:.4f} ({correct_count}/{total_count})")
        logging.info(f"API错误次数: {api_error_count}")
        
        # 保存详细结果
        if save_results:
            result_file = "rag_model_evaluation_results_improved.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump({
                    "total_questions": total_count,
                    "correct_predictions": correct_count,
                    "accuracy": final_accuracy,
                    "api_error_count": api_error_count,
                    "results": results
                }, f, ensure_ascii=False, indent=2)
            logging.info(f"详细结果已保存到: {result_file}")
        
        return {
            "accuracy": final_accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "api_error_count": api_error_count,
            "results": results
        }

def main():
    """主函数"""
    # 配置路径
    documents_dir = "/root/autodl-tmp/LLaMA-Factory/2025_05_22_med_data_zh_paragraph"
    test_path = "/root/autodl-tmp/LLaMA-Factory/data/test.jsonl"
    api_port = 8000
    
    # 检查路径是否存在
    if not os.path.exists(documents_dir):
        logging.error(f"文档目录不存在: {documents_dir}")
        return
    
    if not os.path.exists(test_path):
        logging.error(f"测试数据文件不存在: {test_path}")
        return
    
    try:
        # 初始化RAG评估器
        evaluator = RAGEvaluator(documents_dir, api_port, debug_mode=True)
        
        # 评估测试集
        results = evaluator.evaluate_test_set(test_path)
        
        if results:
            print(f"\n=== RAG base模型评估结果 ===")
            print(f"总问题数: {results['total_count']}")
            print(f"正确预测数: {results['correct_count']}")
            print(f"准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            print(f"API错误次数: {results['api_error_count']}")
            print("======================")
        else:
            print("评估失败，请检查日志文件")
        
    except Exception as e:
        logging.error(f"主函数执行出错: {e}")

if __name__ == "__main__":
    main()

