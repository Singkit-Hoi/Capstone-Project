import torch
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define paths
model_path = "/root/autodl-tmp/Qwen2.5-0.5B-Instruct"
test_path = "/root/autodl-tmp/LLaMA-Factory/data/test.jsonl"

# Load the model and tokenizer
print("Loading original model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Load the test data
print("Loading test data...")
test_data = []
with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))

# Function to generate response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# Function to extract the predicted answer from the response
def extract_answer(response):
    # Check for answer pattern like "答案: A" or "答案是A" or "答案为A"
    answer_pattern = re.search(r'答案[:：是为]\s*([A-E])', response, re.IGNORECASE)
    if answer_pattern:
        return answer_pattern.group(1).upper()
    
    # Check for standalone letter
    standalone_letter = re.search(r'\b([A-E])[.。,，]?\b', response)
    if standalone_letter:
        return standalone_letter.group(1).upper()
    
    # Check for Chinese translations of answer choices
    if "选项" in response.lower() and ("正确" in response or "答案" in response):
        if re.search(r'[AＡ][\s\S]*正确', response, re.IGNORECASE):
            return "A"
        if re.search(r'[BＢ][\s\S]*正确', response, re.IGNORECASE):
            return "B"
        if re.search(r'[CＣ][\s\S]*正确', response, re.IGNORECASE):
            return "C"
        if re.search(r'[DＤ][\s\S]*正确', response, re.IGNORECASE):
            return "D"
        if re.search(r'[EＥ][\s\S]*正确', response, re.IGNORECASE):
            return "E"
    
    # Check for any occurrence of A, B, C, D, E and return the first one
    any_option = re.search(r'([A-E])', response, re.IGNORECASE)
    if any_option:
        return any_option.group(1).upper()
    
    return None

# Process the test data
print("Evaluating original model...")
correct_count = 0
total_count = 0
results = []

for item in tqdm(test_data):
    total_count += 1
    question = item["question"]
    options = item["options"]
    correct_answer = item["answer"]
    correct_idx = item["answer_idx"]
    
    # Format the options
    options_text = ""
    for key, value in options.items():
        options_text += f"{key}. {value}\n"
    
    # Create the prompt
    prompt = f"请回答以下医学选择题，给出正确选项，正确选项只有一个。\n\n问题：{question}\n选项：\n{options_text}\n答案："
    
    # Get the model's response
    response = generate_response(prompt)
    
    # Extract the predicted answer
    predicted_answer = extract_answer(response)
    
    # Check if the prediction is correct
    is_correct = predicted_answer == correct_idx if predicted_answer else False
    if is_correct:
        correct_count += 1
    
    result = {
        "question_id": total_count,
        "question": question,
        "options": options,
        "correct_answer": correct_idx,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "response": response
    }
    results.append(result)
    
    if total_count % 10 == 0:
        print(f"Progress: {total_count}/{len(test_data)}")
        print(f"Current accuracy: {correct_count/total_count:.4f} ({correct_count}/{total_count})")

# Calculate accuracy
accuracy = correct_count / total_count
print(f"Final accuracy: {accuracy:.4f} ({correct_count}/{total_count})")

# Save results for further analysis
with open("original_model_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
