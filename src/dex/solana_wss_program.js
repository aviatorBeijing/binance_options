const WebSocket = require('ws');
const { PublicKey } = require('@solana/web3.js');

// Replace with your RPC WebSocket endpoint
const SOLANA_WS_URL = "wss://api.mainnet-beta.solana.com";

// Replace with your target program ID
const PROGRAM_ID = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"; // WIF

(async () => {
    const ws = new WebSocket(SOLANA_WS_URL);

    ws.on('open', () => {
        console.log("Connected to Solana WebSocket");

        // Subscribe to program
        const subscriptionMessage = {
            jsonrpc: "2.0",
            id: 1, // Unique identifier for the subscription
            method: "programSubscribe",
            params: [
                PROGRAM_ID, // The program ID to subscribe to
                {
                    commitment: 'confirmed', // Use 'finalized' for confirmed transactions
                    encoding: 'jsonParsed', // Change to 'base64' if you prefer raw account data
                
                    filters: []
                }
            ]
        };

        ws.send(JSON.stringify(subscriptionMessage));
        console.log("Subscribed to program:", PROGRAM_ID);
    });

    ws.on('message', (data) => {
        const message = JSON.parse(data);

        if (message.method === "programNotification") {
            const { result } = message.params;
            console.log("Program Notification Received:");
            console.log(JSON.stringify(result, null, 2));

            // Parse the result for your application logic
            // Example: Extract account data or logs
        } else {
            console.log("Other Message Received:", message);
        }
    });

    ws.on('error', (err) => {
        console.error("WebSocket Error:", err);
    });

    ws.on('close', () => {
        console.log("WebSocket Connection Closed");
    });
})();
