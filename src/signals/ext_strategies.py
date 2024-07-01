# -*- coding: utf-8 -*-

import datetime,os,requests
import pandas as pd 

from signals.meta import ExtMixedEmitter, ActionT, ExtRsiEmitter, ExtSentimentEmitter, TradeAction,construct_lastest_signal, trade_recs2df

def main():
    uri =  'http://localhost:5001/api/hyper_tunning?is_fetch_trade_suggestions=true'
    resp = requests.get( uri ).json()
    if resp['ok']:
        tds = resp['trades']
        recs = []
        for td in tds:
            if 'mixed' in td['BOT']:
                emitter = ExtMixedEmitter(td["Capital"], td['BOT'].replace('mixed-',''), td['span'])
            elif 'rsi' in td['BOT']:
                emitter = ExtRsiEmitter(td["Capital"], '-'.join(td['BOT'].split('-')[-2:]), td['span'])
            elif 'sentiment' in td['BOT']:
                emitter = ExtSentimentEmitter(td["Capital"], td['BOT'].replace('sentiment-',''), td['span'])
            #import pprint;pprint.pprint( td ) 
            act = None
            if 'buy' in td['action'].lower():
                act = ActionT.BUY
            elif 'sell' in td['action'].lower():
                act = ActionT.SELL
            else:
                raise Exception(f"*** {td['action']} is not valid.")

            rec = construct_lastest_signal(
                    td['ric'].upper(),
                    td['last_ts'],
                    round(float(td['days'])/365,1),   
                    float(td['max_port_rtn']) * 100,
                    float(td['min_port_rtn']) * 100,
                    float(td['cagr'])*100,
                    float(td['bh_cagr'])*100,
                    td['sortino'].split('|')[0],
                    td['bh_sortino'],
                    td['回撤'].split('|')[0],
                    td['BH回撤'],
                    TradeAction(emitter,td['ric'],act,td['成交价'],td['sizing'], 1., td['ts']),
                    td['last_close'],
            )
            recs += [rec]
    df = trade_recs2df(recs)
    print( df )

if __name__ == '__main__':
    main()
