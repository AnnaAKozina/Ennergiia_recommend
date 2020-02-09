from flask import Flask, render_template, request, redirect, jsonify
from recommend import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

stock = ERP_Stock(ERP_SHOP_ID)
df = stock.get_erp_items_attributes()

@app.route('/', methods=['POST', 'GET'])
def index():
    return 'Hello'

@app.route('/api/<string:item>', methods = ['GET','POST'])
def result(item):
    if request.method == 'POST':
        if item:
            recom = Recommendation(item, df)
            rec = recom.get_recommendations()
            if rec:
                return jsonify(rec)
            else:
                return jsonify({'answer': 'There is no such product'})
        return "No product information is given"
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)