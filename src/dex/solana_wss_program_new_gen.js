const WebSocket = require('ws');
const { Connection, clusterApiUrl } = require('@solana/web3.js');

// Replace with your Solana cluster's WebSocket endpoint
const SOLANA_WS_URL = "wss://api.mainnet-beta.solana.com";

// BPFLoader Program IDs
const BPF_LOADER = "BPFLoader1111111111111111111111111111111111";
const BPF_LOADER_UPGRADEABLE = "BPFLoaderUpgradeab1e11111111111111111111111";

(async () => {
    const ws = new WebSocket(SOLANA_WS_URL);

    ws.on('open', () => {
        console.log("Connected to Solana WebSocket");

        // Subscribe to transaction notifications
        const subscriptionMessage = {
            jsonrpc: "2.0",
            id: 1,
            method: "logsSubscribe",
            params: [
                {
                    commitment: "finalized"
                }
            ]
        };

        ws.send(JSON.stringify(subscriptionMessage));
        console.log("Subscribed to logs");
    });

    ws.on('message', (data) => {
        const message = JSON.parse(data);

        if (message.method === "logsNotification") {
            const logs = message.params.result;
            const { signature, logs: logMessages } = logs.value;

            console.log(`Transaction: ${signature}`);

            // Check if the transaction involves BPFLoader or BPFLoaderUpgradeable
            const isProgramDeployment = logMessages.some(log =>
                log.includes(BPF_LOADER) || log.includes(BPF_LOADER_UPGRADEABLE)
            );

            if (isProgramDeployment) {
                console.log("New Program Deployment Detected!");
                console.log("Logs:", logMessages);
                // Additional parsing can be added here
            }
        }
    });

    ws.on('error', (err) => {
        console.error("WebSocket Error:", err);
    });

    ws.on('close', () => {
        console.log("WebSocket Connection Closed");
    });
})();
