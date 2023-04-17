from flask import Flask, request, jsonify
import json


app = Flask(__name__)


@app.route('/getModel', methods=['POST'])
def get_model():
    print("Hello World")
    
    
    # question = request.form['question']
    return jsonify({"answer": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."})
        
@app.route('/')
def hello_world():
    print("hello world")
    return "Hello World"

if __name__ == '__main__':
    app.run(port=2002)
