from flask import Flask, request, jsonify
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype="float32")

@app.route('/generate', methods=['POST'])
def generate():
    input_text = request.json.get('input_text')
    max_length = request.json.get('max_length')

    input_features = tokenizer(input_text, return_tensors="pd")
    outputs = model.generate(**input_features, max_length=max_length)
    result = tokenizer.batch_decode(outputs[0])

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
