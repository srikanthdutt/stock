from flask import Flask
import git

from src.business_logic.logic import *

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return f'Welcome to Stock Picker!!\nEX: get_stock_val/<ticker>\n'


@app.route('/balance_accuracy/', methods=['GET'])
def test():
    ba = get_overall_accuracy('lstm', 3*365)
    return f'The Balance Accuracy of all of the S&P500 together is {ba}'


@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    # return f'the ticker selected is {ticker}'
    return get_prediction(ticker.upper(), mname='lstm', days_cnt=3 * 365)


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
