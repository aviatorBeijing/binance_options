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
                "last_close": float(prices[-1]),
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
    """
    @brief
        Show possible spot price ranges, based on the options open interests.
    """
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

@app.route('/pricing_options_from_spot', methods=['GET'])
def pricing_options_from_spot():
    contracts = request.args.get('contracts')
    prange = request.args.get('prange')
    if contracts:
        contracts = contracts.split(',')

    if os.getenv("YAHOO_LOCAL",None):
        return jsonify({
            'ok': True,
            "calls":{
                "columns": ['col1','col2','col3','col4','col5'],
                "data": [[1,2,3,4,5,],[3,4,5,6,7,]]
            },
            "puts":{
                "columns": ['col1','col2','col3','col4','col5'],
                "data": [[1,2,3,4,5,],[3,4,5,6,7,]]
            },
            "spot":{
                "price": 60000,
            }
        }), 200
    else:
        from strategy.options_pricing.options_price_projection_from_spot_price import _main
        if prange:
            px = prange.split(',')
            rst = _main(contracts, list( np.range(px[0],px[1],px[2])) )
        else:
            rst = _main(contracts, list( range(55000,70001,1000)) )
        return jsonify( rst  ),200

@app.route('/calc_straddle', methods=['GET'])
def calc_straddle():
    contracts = request.args.get('contracts')
    if contracts:
        contracts = contracts.split(',')

    if os.getenv("YAHOO_LOCAL",None):
        return jsonify({
            'ok': True,
            "be_prices":[1,2,3,4,5,],
            "be_returns":[1,2,3,4,5,],
            "straddle_returns":[1,2,3,4,5,],
            "left": "put-contract",
            "right": "call-contract",
            "time_values":{
                "put-contract": 100.,
                "call-contract": 100.,
            }

        }), 200
    else:
        from strategy.straddle import _main as cstraddle
        assert len(contracts)==2, f"{contracts} contains elements NOT equal to 2!"
        lc = puts = list(filter(lambda s: s.endswith('-P'), contracts))[0]
        rc = calls = list(filter(lambda s: s.endswith('-C'), contracts))[0]
        resp  = cstraddle(lc,rc,1.)
        resp['be_prices'] = [float(v) for v in resp['be_prices']]
        return jsonify( resp  ),200

@app.route('/pricing_pairs', methods=['GET'])
def pricing_pairs():
    contracts = request.args.get('contracts')
    if contracts:
        contracts = contracts.split(',')

    if os.getenv("YAHOO_LOCAL",None):
        return jsonify({
            'ok': True,
            "columns": 'a,b,c'.split(','),
            "data": [ [1,2,3],[4,5,6]],
            "market": {
                "columns": 'assets,bid,asks,deviation_from_BS'.split(','),
                "data": [
                    ['a',2,3,4],
                    ['b',4,5,6],
                ]
            }
        }), 200
    else:
        from brisk.pricing import _multicontracts_main
        resp = _multicontracts_main( contracts )
        resp['ok'] = True
        return jsonify( resp  ),200


@app.route('/send_buy_order', methods=['GET'])
def send_buy_order():
    contract = request.args.get('contract')
    pce = request.args.get('price')
    qty = request.args.get('qty')
    
    try:
        pce = float(pce)
        qty = float(qty)
        assert pce>0, f'{pce} is not a valid price'
        assert qty>0, f'{qty} is not a valid qty'
        assert len(contract)>1 and ('-P' in contract or '-C' in contract), f'{contract} is not acceptable contract name.'
        
        if os.getenv("YAHOO_LOCAL",None):
            return jsonify({ # Local fake return
                'ok': True,
                'info': {
                    'oid': '4711026509648199680',
                    'datetime':'2024-09-20T08:30:03.384Z',
                    'status': 'ACCEPTED(local fake)',
                }
            }),200
        
        from bbroker.order_mgr import buy_
        info = buy_(contract, qty, pce )
        return jsonify( {
            'ok': True,
            'info': {
                'oid': info['id'],
                'datetime': info['datetime'],
                'status': info['info']['status'],
            }
            }),200
    except Exception as e:
        print('***', str(e))
        return jsonify({
            'ok': False,
            'msg': f'Server error: {str(e)}'
        }, 200)



from swagger_template import swagger_json
@app.route('/static/swagger.json')
def swagger_spec():
    return jsonify(swagger_json)


if __name__ == '__main__':
    app.run(port=5057)

