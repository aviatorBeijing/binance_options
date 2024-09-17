from flask import Flask, jsonify, request, send_file
from flask_swagger_ui import get_swaggerui_blueprint
import numpy as np
import pandas as pd
import os 

app = Flask(__name__)

# Swagger UI setup
SWAGGER_URL = '/swagger'  # URL for Swagger UI
API_URL = '/static/swagger.json'  # Path to the Swagger JSON file

# Setup the blueprint for swagger ui
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={  # Swagger UI config
        'app_name': "Flask Swagger Example"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# In-memory data store
items = {}

# GET example: Fetch an item by ID
@app.route('/item/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = items.get(item_id)
    if item:
        return jsonify({'item': item}), 200
    else:
        return jsonify({'error': 'Item not found'}), 404

# POST example: Create a new item
@app.route('/item', methods=['POST'])
def create_item():
    data = request.json
    item_id = data.get('id')
    name = data.get('name')
    if not item_id or not name:
        return jsonify({'error': 'Invalid data'}), 400
    if item_id in items:
        return jsonify({'error': 'Item already exists'}), 400
    items[item_id] = name
    return jsonify({'message': 'Item created'}), 201

# PUT example: Update an existing item
@app.route('/item/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    if item_id not in items:
        return jsonify({'error': 'Item not found'}), 404
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({'error': 'Invalid data'}), 400
    items[item_id] = name
    return jsonify({'message': 'Item updated'}), 200

@app.route('/historical_vol', methods=['GET'])
def historical_vol():
    ric = request.args.get('ric')
    years = request.args.get('years', type=int)
    scanning = request.args.get('scanning')
    if scanning:
        scanning = scanning.lower()=="true"

    from strategy.gamma_scalping.merton_jump_model import _sim,read_prices_from_csv,calc_log_returns
    from strategy.gamma_scalping._configs import nDays
    x = int(years)

    if os.getenv("YAHOO_LOCAL",None):
        prices, dates = read_prices_from_csv(ric)
    else:
        from spot_trading.market_data import binance_kline
        df = binance_kline(ric.upper().replace('-','/'), span='1d', grps=10)
        df.columns = [v.lower() for v in df.columns]
        prices = df.close.values 
        dates = pd.date_range( end=df['timestamp'].iloc[-1],periods=df.shape[0] )

    tScale = nDays    # Indicates the time scale in the data "prices"
    n_paths = 100
    if not scanning:
        prices = prices[-nDays*x:]
        dates  = dates[-nDays*x:]
        
        lrtns = calc_log_returns(prices)
        arithmatic_sigma = np.std(lrtns)*np.sqrt(365)
        
        mle_sigma,\
            volatilities, mean_volatility,std_volatility,p68, \
                paths,gen_dates = _sim(
                    prices,dates,tScale,n_paths = n_paths)

        return jsonify({
            "meta": {
                "n_paths": n_paths,
                "start" : str(dates[0]),
                "end"   : str(dates[-1]),
            },
            "data":{
                "mle_sigma": mle_sigma,
                "sim_mean": mean_volatility,
                "sim_p68" : p68,
                "art_sigma": arithmatic_sigma,
            }
        }), 200
    else:
        recs = []
        for x in range(1,11):
            xprices = prices[-nDays*x:]
            xdates  = dates[-nDays*x:]
            
            lrtns = calc_log_returns(xprices)
            arithmatic_sigma = np.std(lrtns)*np.sqrt(365)  
            
            mle_sigma,\
                volatilities, mean_volatility,std_volatility,p68, \
                    paths,gen_dates = _sim(
                        xprices,xdates,tScale,n_paths = n_paths)
            recs += [{
                    'years': float(x),
                    "mle_sigma": mle_sigma,
                    "sim_mean": mean_volatility,
                    "sim_p68" : p68,
                    "art_sigma": arithmatic_sigma,
            }]
        df = pd.DataFrame.from_records( recs )
        rows = [list(e) for e in list( df.to_records(index=False))]
        return jsonify(
            {
                "columns": list(df.columns),
                "rows": rows,
            }
        ), 200

@app.route('/price_ranges', methods=['GET'])
def price_ranges():
    underlying = request.args.get('underlying') # BTC,ETH
    atm_contracts = request.args.get('atm_contracts') # Return atm contracts as well
    update = request.args.get('update')
    if atm_contracts:
        atm_contracts = atm_contracts.lower()=='true'
    if update:
        update = update.lower()=='true'

    from sentiments.atms import _wrapper_price_range
    rst = _wrapper_price_range( underlying.upper(), 
                            show_atm_contracts=atm_contracts, 
                            update=update)
    
    return jsonify( rst  ),200

from swagger_template import swagger_json
@app.route('/static/swagger.json')
def swagger_spec():
    return jsonify(swagger_json)


if __name__ == '__main__':
    app.run(port=5057)

