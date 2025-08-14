import json
import os

def convert_to_alpaca_format(input_file, output_file):
    """将原始医学问答数据集转换为Alpaca格式"""
    alpaca_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            # 构建问题文本
            question_text = item['question']
            options_text = "\n".join([f"{k}: {v}" for k, v in item['options'].items()])
            full_question = f"问题：{question_text}\n选项：\n{options_text}"
            
            # 构建答案
            answer = f"答案是：{item['answer_idx']}. {item['answer']}"
            
            # 创建Alpaca格式的数据项
            alpaca_item = {
                "instruction": "请回答以下医学选择题，给出正确选项，正确选项只有一个。",
                "input": full_question,
                "output": answer
            }
            
            alpaca_data.append(alpaca_item)
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
   
    print(f"已将{len(alpaca_data)}条数据转换为Alpaca格式并保存至{output_file}")

# 转换训练集和测试集
convert_to_alpaca_format('train.jsonl', 'medical_qa_train.json')
#convert_to_alpaca_format('test.jsonl', 'medical_qa_test.json')
