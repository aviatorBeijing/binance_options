const WebSocket = require('ws');
const { Connection, clusterApiUrl, PublicKey } = require('@solana/web3.js');

// Solana WebSocket endpoint (e.g., mainnet-beta)
const WS_URL = 'wss://api.mainnet-beta.solana.com';

// HTTP endpoint to fetch leader schedule
const HTTP_URL = clusterApiUrl('mainnet-beta');

// Function to get the leader for a specific slot
async function fetchLeaderForSlot(slot) {
    try {
        const connection = new Connection(HTTP_URL);
        const leaderSchedule = await connection.getLeaderSchedule(slot);
        const currentLeader = Object.entries(leaderSchedule).find(([_, slots]) =>
            slots.includes(slot)
        );
        return currentLeader ? currentLeader[0] : 'Unknown';
    } catch (error) {
        console.error('Error fetching leader schedule:', error);
        return 'Unknown (error)';
    }
}

(async () => {
    // Establish a WebSocket connection
    const ws = new WebSocket(WS_URL);

    // Handle connection open
    ws.on('open', () => {
        console.log('Connected to Solana WebSocket server.');

        // Send a slotSubscribe request
        const request = {
            jsonrpc: '2.0',
            id: 1,
            method: 'slotSubscribe',
        };

        console.log('Sending slotSubscribe request:', request);
        ws.send(JSON.stringify(request));
    });

    // Handle incoming messages
    ws.on('message', async (data) => {
        const response = JSON.parse(data);

        // Check if it's the subscription confirmation
        if (response.result) {
            console.log('Subscription confirmed with ID:', response.result);
        }

        // Handle slot updates
        if (response.method === 'slotNotification') {
            const { slot, parent, root } = response.params.result;

            console.log('Root Slot:', root);
            console.log('    Slot:', slot);
            console.log('  Parent Slot:', parent);

            // Fetch and log the leader for this slot
            const leader = await fetchLeaderForSlot(slot);
            console.log('Leader for Slot:', leader);
            console.log('')
        }
    });

    // Handle errors
    ws.on('error', (err) => {
        console.error('WebSocket error:', err);
    });

    // Handle connection close
    ws.on('close', () => {
        console.log('WebSocket connection closed.');
    });
})();
