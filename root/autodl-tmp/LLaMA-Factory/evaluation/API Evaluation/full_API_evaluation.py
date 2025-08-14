import json
import re
import requests
from tqdm import tqdm
import time

# API configuration
API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "0"  # Default API key for LLaMA-Factory

# Test data path
test_path = "/root/autodl-tmp/LLaMA-Factory/data/test.jsonl"

# Load the test data
print("Loading test data...")
test_data = []
with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))

# Function to generate response via API
def generate_response(prompt, max_retries=3):
    """Generate response using the LLaMA-Factory API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "default",  # LLaMA-Factory uses "default" as model name
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": 512,
        "stream": False
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"Unexpected API response format: {result}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None

# Function to extract the predicted answer from the response
def extract_answer(response):
    if not response:
        return None
        
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

# Test API connection
print("Testing API connection...")
test_prompt = "你好，请回复'我在正常工作'。"
test_response = generate_response(test_prompt)
if test_response:
    print(f"API is working. Test response: {test_response}")
else:
    print("Warning: API test failed. Please check if the API server is running.")
    print(f"Make sure to run: API_PORT=8000 llamafactory-cli api examples/inference/llama3_full_sft.yaml infer_backend=vllm vllm_enforce_eager=true")
    exit(1)

# Process the test data
print("Evaluating fine-tuned model via API...")
correct_count = 0
total_count = 0
results = []
failed_requests = 0

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
    
    if response is None:
        failed_requests += 1
        predicted_answer = None
        is_correct = False
    else:
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
        "response": response if response else "API request failed"
    }
    results.append(result)
    
    # Progress update
    if total_count % 10 == 0:
        print(f"\nProgress: {total_count}/{len(test_data)}")
        print(f"Current accuracy: {correct_count/total_count:.4f} ({correct_count}/{total_count})")
        if failed_requests > 0:
            print(f"Failed requests: {failed_requests}")
    
    # Add a small delay to avoid overwhelming the API
    time.sleep(0.1)

# Calculate accuracy
if total_count > 0:
    accuracy = correct_count / total_count
    print(f"\nFinal accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    if failed_requests > 0:
        print(f"Total failed requests: {failed_requests}/{total_count}")
else:
    print("No test data processed.")

# Save results for further analysis
output_file = "half_full_finetuned_model_api_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to: {output_file}")

# Print summary statistics
if results:
    print("\nSummary Statistics:")
    print(f"Total questions: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Incorrect answers: {total_count - correct_count - failed_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Accuracy (excluding failed): {correct_count/(total_count-failed_requests):.4f}" if total_count > failed_requests else "N/A")

