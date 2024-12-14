const Redis = require('ioredis');

// Create a Redis client
const redis = new Redis();

// Get the key name from the command-line arguments
const keyName = process.argv[2];

if (!keyName) {
    console.error("Error: Please provide a key name as a command-line argument.");
    process.exit(1);
}

// Function to clear the content of a specific key
async function clearKeyContent(key) {
    try {
        // Check if the key exists
        const exists = await redis.exists(key);
        if (exists) {
            // Set the key's value to an empty string
            await redis.set(key, "");
            console.log(`Cleared content of key: ${key}`);
        } else {
            console.log(`Key: ${key} does not exist in Redis.`);
        }
    } catch (err) {
        console.error(`Error clearing key content: ${err.message}`);
    } finally {
        // Disconnect the Redis client
        redis.disconnect();
    }
}

// Call the function with the provided key name
clearKeyContent(keyName);
