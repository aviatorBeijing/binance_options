const WebSocket = require('ws'); // WebSocket library
const { Connection, clusterApiUrl } = require('@solana/web3.js'); // Solana utilities

// Connect to Solana WebSocket RPC endpoint
const WS_URL = "wss://api.mainnet-beta.solana.com"; // Replace with your provider's WebSocket URL
const connection = new WebSocket(WS_URL);

connection.on('open', () => {
    console.log('WebSocket connection established.');

    // Subscribe to `rootSubscribe`
    const message = {
        jsonrpc: "2.0",
        id: 1,
        method: "rootSubscribe",
        params: []
    };

    connection.send(JSON.stringify(message));
    console.log('Subscribed to root updates.');
});

connection.on('message', (data) => {
    const response = JSON.parse(data);

    if (response.method === 'rootNotification') {
        const rootSlot = response.params.result;
        console.log(`New root slot: ${rootSlot}`);

        // Example: Critical information parsing (e.g., slot number)
        handleRootNotification(rootSlot);
    } else if (response.error) {
        console.error('Error received:', response.error);
    }
});

connection.on('close', () => {
    console.log('WebSocket connection closed.');
});

connection.on('error', (err) => {
    console.error('WebSocket error:', err);
});

// Custom handler for critical information
function handleRootNotification(slot) {
    console.log(`Handling root slot: ${slot}`);
    // Add custom logic, e.g., updating a database or triggering further actions
}
