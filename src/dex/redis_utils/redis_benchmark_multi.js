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

// Calculate mean and standard deviation
function calculateStats(times) {
    const mean = times.reduce((sum, t) => sum + t, 0) / times.length;
    const variance = times.reduce((sum, t) => sum + Math.pow(t - mean, 2), 0) / times.length;
    const std = Math.sqrt(variance);
    return { mean, std };
}

// Benchmark function
async function benchmarkRedis(redisClient, keyName, data, iterations = 50) {
    console.log(`\nTesting Redis performance on key: ${keyName} (${iterations} iterations)`);

    const writeTimes = [];
    const readTimes = [];

    for (let i = 0; i < iterations; i++) {
        // Measure write speed
        const writeStart = Date.now();
        await redisClient.set(keyName, data);
        const writeEnd = Date.now();
        const writeTime = writeEnd - writeStart;
        writeTimes.push(writeTime);

        // Measure read speed
        const readStart = Date.now();
        const fetchedData = await redisClient.get(keyName);
        const readEnd = Date.now();
        const readTime = readEnd - readStart;
        readTimes.push(readTime);

        // Verify data integrity
        const isDataCorrect = fetchedData === data;
        if (!isDataCorrect) {
            console.error(`Data Integrity Failed at iteration ${i + 1}`);
            break;
        }

        console.log(`Iteration ${i + 1}: Write Time = ${writeTime} ms, Read Time = ${readTime} ms`);
    }

    const writeStats = calculateStats(writeTimes);
    const readStats = calculateStats(readTimes);

    console.log(`\nSummary for ${keyName}:`);
    console.log(`- Write: Mean = ${writeStats.mean.toFixed(2)} ms, Std Dev = ${writeStats.std.toFixed(2)} ms`);
    console.log(`- Read: Mean = ${readStats.mean.toFixed(2)} ms, Std Dev = ${readStats.std.toFixed(2)} ms`);

    return { writeStats, readStats };
}

// Main function
(async () => {
    try {
        const fakeData = generateFakeData(1024); // Generate 1 MB of fake data

        // Benchmark wif_logs cache
        const logsResults = await benchmarkRedis(wifLogsCache, 'large_test_key', fakeData, 50);

        // Benchmark wif_signatures cache
        const signaturesResults = await benchmarkRedis(wifSignaturesCache, 'large_test_key', fakeData, 50);
    } catch (err) {
        console.error("Error during benchmark:", err);
    } finally {
        // Clean up and disconnect
        wifLogsCache.disconnect();
        wifSignaturesCache.disconnect();
    }
})();