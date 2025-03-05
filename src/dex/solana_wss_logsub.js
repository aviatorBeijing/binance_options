const WebSocket = require('ws');
const { Connection, clusterApiUrl } = require('@solana/web3.js');

// Solana WebSocket endpoint
const WS_URL = 'wss://api.mainnet-beta.solana.com';
const connection = new Connection(clusterApiUrl('mainnet-beta'), 'confirmed'); // RPC HTTP endpoint

// Connect to WebSocket
const ws = new WebSocket(WS_URL);

ws.on('open', () => {
    console.log('Connected to Solana WebSocket');

    // Subscription payload for logs
    const subscriptionPayload = {
        jsonrpc: "2.0",
        id: 1,
        method: "logsSubscribe",
        params: [
            {
                mentions: ["EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"] //$WIF
            },
            {
                commitment: "confirmed"
            }
        ]
    };

    // Send subscription request
    ws.send(JSON.stringify(subscriptionPayload));
    console.log('Subscribed to logs for program.');
});

ws.on('message', async (data) => {
    try {
        const response = JSON.parse(data);

        if (response.method === "logsNotification") {
            const logs = response.params?.result?.value;
            if(logs.err==null){
                console.log( response ) // DEBUG 
                console.log('Transaction Logs:', logs);
            }
            else{
                console.log('*** chain error:', logs.signature)
            }
            
            const signature = logs.signature;
            if (null){//signature) { //easily trigger the RPC rate-limit
                console.log('signature: ', signature)
                const transaction = await connection.getTransaction(signature, {
                    commitment: "confirmed",
                    maxSupportedTransactionVersion: 0 // Adjust based on your program version
                });

                if (transaction) {
                    console.log('Transaction Details:', transaction);
                    const instructions = transaction.transaction.message.instructions;
                    console.log('Instructions:', instructions);
                } else {
                    console.warn('Transaction details not found.');
                }
            }

            // Process logs or extract relevant data
        }
    } catch (error) {
        console.error('Error processing WebSocket message:', error.message);
    }
});

ws.on('close', () => {
    console.log('WebSocket connection closed');
});

ws.on('error', (error) => {
    console.error('WebSocket error:', error);
});
