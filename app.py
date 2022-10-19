from flask import Flask
import git

from src.business_logic.logic import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return f'Welcome to Stock Picker!!\nEX: get_stock_val/<ticker>\n'

@app.route('/test/', methods=['GET'])
def test():
    return f'Test is being done'

@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    # return f'the ticker selected is {ticker}'
    return get_prediction(ticker)



if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
