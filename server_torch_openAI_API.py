import torch

# load model using Torch
model = torch.load("path/to/model")

# host model according to OpenAI API protocol
# ...

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from flask import Flask, request, jsonify

# 初始化 Flask 應用
app = Flask(__name__)

# 加載模型和分詞器
model_name = "gpt2"  # 替換為你想要使用的模型名稱
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 將模型移動到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 16777215,
                "owned_by": "openai",
                "permissions": None
            }
        ],
        "has_more": False
    })

@app.route('/v1/completions', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_tokens', 50)
    
    # 將提示文本轉換為模型可用的格式
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # 生成文本
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({
        "id": "cmpl-123456789",
        "object": "text_completion",
        "created": 16777215,
        "model": model_name,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
            "total_tokens": len(tokenizer.encode(prompt)) + len(outputs[0]) - len(inputs.input_ids[0])
        }
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completion():
    data = request.json
    messages = data.get('messages', [])
    
    # 提取最後一個消息的內容作為提示文本
    prompt = messages[-1].get('content', '') if messages else ''
    max_length = data.get('max_tokens', 50)
    
    # 將提示文本轉換為模型可用的格式
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # 生成文本
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({
        "id": "chatcmpl-123456789",
        "object": "chat.completion",
        "created": 16777215,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                }
            }
        ],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
            "total_tokens": len(tokenizer.encode(prompt)) + len(outputs[0]) - len(inputs.input_ids[0])
        }
    })

@app.route('/v1/embeddings', methods=['POST'])
def generate_embedding():
    data = request.json
    input_text = data.get('input', '')
    
    # 加載嵌入模型（如果需要）
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 替換為你想要使用的嵌入模型名稱
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)
    
    # 將文本轉換為嵌入向量
    inputs = embedding_tokenizer(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs)[0]
    mean_embedding = torch.mean(embeddings, dim=1).squeeze().tolist()
    
    return jsonify({
        "object": "list",
        "data": [
            {
                "index": 0,
                "embedding": mean_embedding
            }
        ],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(input_text))
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)