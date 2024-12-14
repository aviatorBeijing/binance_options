const Redis = require("ioredis");

// Create a Redis client
const redisClient = new Redis({
    host: '127.0.0.1',
    port: 6379,
    db: 0 // Specify the database number (default is 0)
});

(async () => {
    try {
        console.log("Fetching all available keys in Redis with their data sizes...");

        // Fetch all keys
        const keys = await redisClient.keys('*');

        if (keys.length === 0) {
            console.log("No keys found in the Redis database.");
        } else {
            console.log(`Found ${keys.length} keys:`);
            
            // Iterate over each key to fetch its size
            for (const [index, key] of keys.entries()) {
                try {
                    // Use MEMORY USAGE to get the approximate memory usage of the key
                    const size = await redisClient.memory('usage', key);
                    console.log(`${index + 1}: Key = "${key}", Size = ${size || 0} bytes`);
                } catch (err) {
                    console.error(`Failed to get size for key "${key}":`, err);
                }
            }
        }
    } catch (err) {
        console.error("Error fetching keys from Redis:", err);
    } finally {
        // Disconnect the Redis client
        redisClient.disconnect();
    }
})();
