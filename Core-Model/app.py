from driver import MainProgram
from flask import Flask, jsonify, request

app = Flask(__name__)

driver = MainProgram()


@app.route('/getModel', methods=['POST'])
def get_model():
    question = request.form['question']
    response = driver.forward(question)
    return response


if __name__ == '__main__':
    app.run()
