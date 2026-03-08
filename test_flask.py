from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello from Flask!"

@app.route('/api/test')
def test_api():
    return jsonify({'success': True, 'message': 'API is working'})

if __name__ == '__main__':
    print("Starting test Flask app...")
    app.run(debug=True, port=5001)
