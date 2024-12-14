const Redis = require("ioredis");

// Create Redis clients for wif_logs and wif_signatures
const wifLogsCache = new Redis({
    host: '127.0.0.1', // Replace with your Redis server host
    port: 6379,        // Default Redis port
    db: 0,             // Use database 0 for wif_logs
    keyPrefix: 'wif_logs:' // Optional prefix for keys
});

const wifSignaturesCache = new Redis({
    host: '127.0.0.1', // Replace with your Redis server host
    port: 6379,        // Default Redis port
    db: 1,             // Use database 1 for wif_signatures
    keyPrefix: 'wif_signatures:' // Optional prefix for keys
});

// Example usage: Set and get values
(async () => {
    try {
        // Set a key-value pair in wif_logs
        await wifLogsCache.set('exampleKey', 'exampleValue');
        console.log("wif_logs: exampleKey set!");

        // Get the value back
        const valueFromLogs = await wifLogsCache.get('exampleKey');
        console.log("wif_logs: exampleKey value:", valueFromLogs);

        // Set a key-value pair in wif_signatures
        await wifSignaturesCache.set('exampleKey', 'signatureValue');
        console.log("wif_signatures: exampleKey set!");

        // Get the value back
        const valueFromSignatures = await wifSignaturesCache.get('exampleKey');
        console.log("wif_signatures: exampleKey value:", valueFromSignatures);
    } catch (err) {
        console.error("Error interacting with Redis:", err);
    } finally {
        // Clean up
        wifLogsCache.disconnect();
        wifSignaturesCache.disconnect();
    }
})();