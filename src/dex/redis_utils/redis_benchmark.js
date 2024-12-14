const Redis = require("ioredis");

// Create Redis clients for wif_logs and wif_signatures
const wifLogsCache = new Redis({
    host: '127.0.0.1',
    port: 6379,
    db: 0,
    keyPrefix: 'wif_logs:'
});

const wifSignaturesCache = new Redis({
    host: '127.0.0.1',
    port: 6379,
    db: 1,
    keyPrefix: 'wif_signatures:'
});

// Helper function to generate large fake data
function generateFakeData(sizeInKB) {
    const data = 'A'.repeat(sizeInKB * 1024); // Generate a string of sizeInKB KB
    return data;
}

// Benchmark function
async function benchmarkRedis(redisClient, keyName, data) {
    console.log(`\nTesting Redis performance on key: ${keyName}`);

    // Measure write speed
    const writeStart = Date.now();
    await redisClient.set(keyName, data);
    const writeEnd = Date.now();
    const writeTime = writeEnd - writeStart;
    console.log(`Write Time: ${writeTime} ms`);

    // Measure read speed
    const readStart = Date.now();
    const fetchedData = await redisClient.get(keyName);
    const readEnd = Date.now();
    const readTime = readEnd - readStart;
    console.log(`Read Time: ${readTime} ms`);

    // Verify correctness
    const isDataCorrect = fetchedData === data;
    console.log(`Data Integrity Check: ${isDataCorrect ? 'Passed' : 'Failed'}`);

    return { writeTime, readTime };
}

// Main function
(async () => {
    try {
        const fakeData = generateFakeData(1024); // Generate 1 MB of fake data

        // Benchmark wif_logs cache
        const logsResults = await benchmarkRedis(wifLogsCache, 'large_test_key', fakeData);
        console.log(`wif_logs - Write: ${logsResults.writeTime} ms, Read: ${logsResults.readTime} ms`);

        // Benchmark wif_signatures cache
        const signaturesResults = await benchmarkRedis(wifSignaturesCache, 'large_test_key', fakeData);
        console.log(`wif_signatures - Write: ${signaturesResults.writeTime} ms, Read: ${signaturesResults.readTime} ms`);
    } catch (err) {
        console.error("Error during benchmark:", err);
    } finally {
        // Clean up and disconnect
        wifLogsCache.disconnect();
        wifSignaturesCache.disconnect();
    }
})();
