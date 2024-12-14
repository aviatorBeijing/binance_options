const Redis = require('ioredis');
const redis = new Redis();

/**
 * Reads the content of a specific Redis key and parses it as JSON.
 * 
 * @param {string} key - The Redis key to read.
 */
async function readFromRedisLog(key) {
    try {
        // Retrieve all elements from the list
        const data = await redis.lrange(key, 0, -1);

        // Parse each element as JSON
        const parsedData = data.map(item => JSON.parse(item));

        console.log(`Contents of ${key}:`, parsedData);
        return parsedData;
    } catch (error) {
        console.error(`Error reading from Redis key ${key}:`, error);
    }
}

readFromRedisLog(process.argv[2]);
//"wif_logs:raw_object");
