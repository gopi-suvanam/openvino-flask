from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

app = Flask(__name__)

# Global variables to hold the model and tokenizer
tokenizer = None
model = None

# Endpoint to load the model
@app.route('/load_model', methods=['GET'])
def load_model():
    global tokenizer, model
    try:
        # Specify your model
        model_id = "OpenVINO/open_llama_7b_v2-fp16-ov"
        
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = OVModelForCausalLM.from_pretrained(model_id)
        
        return jsonify({"message": "Model loaded successfully"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to generate a response based on the prompt
@app.route('/generate_response', methods=['POST'])
def generate_response():
    if not tokenizer or not model:
        return jsonify({"error": "Model is not loaded"}), 400
    
    # Extract prompt from the POST request
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # Tokenize input and generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=2000)
        text = tokenizer.batch_decode(outputs)[0]
        
        return jsonify({"response": text}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
