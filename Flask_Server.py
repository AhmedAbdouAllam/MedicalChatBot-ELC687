from flask import Flask, request, jsonify
from flask_cors import CORS
from main import *

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the QA chain once at the start of the server
qa_chain = Intializer()

def qa(question):
    return "This is the answer to your question: " + question

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')
    answer = qa_chain({"query": question, 'k': 1})
    return jsonify({'answer': answer["result"]})

if __name__ == '__main__':
    app.run(debug=True)
