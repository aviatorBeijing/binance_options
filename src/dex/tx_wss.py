
# https://michaelhly.com/solana-py/

import aiohttp
import asyncio
import json
from asyncstdlib import enumerate
from solana.rpc.websocket_api import connect
from solders.pubkey import Pubkey

from solana.rpc.api import Client

ENDPOINT='wss://api.mainnet-beta.solana.com/' #wss://api.devnet.solana.com
RPC_ENDPOINT = "https://api.mainnet-beta.solana.com"
#https://github.com/michaelhly/solana-py/blob/master/src/solana/rpc/websocket_api.py#L215

client = Client(RPC_ENDPOINT)

def parse_transaction_details(transaction_response):
    """
    Parses the response from a Solana transaction query.

    Args:
        transaction_response (dict): The JSON response from the Solana RPC API.

    Returns:
        dict: Parsed transaction details.
    """
    if not transaction_response or not transaction_response.result:
        return {"error": "Transaction not found or still unconfirmed."}

    transaction = transaction_response.result
    print( transaction )

    # Extract relevant fields
    parsed_details = {
        "slot": transaction.get("slot", "N/A"),
        "block_time": transaction.get("blockTime", "N/A"),
        "status": "Success" if transaction.get("meta", {}).get("err") is None else "Error",
        "pre_balances": transaction.get("meta", {}).get("preBalances", []),
        "post_balances": transaction.get("meta", {}).get("postBalances", []),
        "instructions": []
    }

    # Extract instructions
    instructions = transaction.get("transaction", {}).get("message", {}).get("instructions", [])
    for i, instruction in enumerate(instructions):
        parsed_details["instructions"].append({
            "instruction_index": i + 1,
            "program_id_index": instruction.get("programIdIndex"),
            "accounts": instruction.get("accounts"),
            "data": instruction.get("data")
        })

    return parsed_details
async def fetch_transaction_logs(signature, rpc_url=RPC_ENDPOINT):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [signature, {"encoding": "jsonParsed"}]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(rpc_url, json=payload) as response:
            result = await response.json()
            logs = result.get("result", {}).get("meta", {}).get("logMessages", [])
            return logs

async def parse_logs(logs):
    parsed_data = []
    for log in logs:
        if "Program log:" in log:
            parsed_data.append(log.split("Program log:")[1].strip())
    return parsed_data


async def main():
    
    """async with connect( ENDPOINT ) as websocket:
        await websocket.program_subscribe( Pubkey('EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm') )
        first_resp = await websocket.recv()
        subscription_id = first_resp[0].result
        next_resp = await websocket.recv()
        print(next_resp)
        await websocket.program_subscribe(subscription_id)
        """

    """async with connect( ENDPOINT ) as websocket:
        await websocket.logs_subscribe()
        first_resp = await websocket.recv()
        subscription_id = first_resp[0].result
        next_resp = await websocket.recv()
        print(next_resp)
        print('***',  next_resp[0].result.value.logs[0])
        print('***',  next_resp[0].result.value.signature)
        await websocket.logs_unsubscribe(subscription_id)
        """

    # Alternatively, use the client as an infinite asynchronous iterator:
    async with connect( ENDPOINT ) as websocket:
        await websocket.logs_subscribe()
        first_resp = await websocket.recv()
        subscription_id = first_resp[0].result
        async for idx, msg in enumerate(websocket):
            if idx == 10:
                break
            for ele in msg:
                #print(ele)
                for lg in ele.result.value.logs:
                    print('***\t',  lg)
                print('***',  ele.result.value.signature)

                response = client.get_transaction( ele.result.value.signature, max_supported_transaction_version=0 )
                print( response )
                # print( parse_transaction_details( response ) )

                
        await websocket.logs_unsubscribe(subscription_id)

asyncio.run(main())
