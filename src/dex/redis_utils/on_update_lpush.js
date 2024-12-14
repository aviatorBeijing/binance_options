const Redis = require('ioredis');
const redis = new Redis();
const subscriber = new Redis();

/**
 * Prerequisite:
 * >> redis-cli config set notify-keyspace-events KEA
 */

/**
 * Sets up a monitor to watch for updates on a specific Redis key and triggers an event to process the update.
 * 
 * @param {string} key - The Redis key to monitor.
 * @param {Function} processUpdate - The callback function to process the update.
 */
async function monitorKeyUpdate(key, processUpdate) {
    try {
        // Enable keyspace notifications
        await redis.config("SET", "notify-keyspace-events", "KEA"); // Enable full notifications

        const channel = `__keyevent@0__:lpush`; // Listen for specific events
        subscriber.subscribe(channel, (err, count) => {
            if (err) {
                console.error("Failed to subscribe to channel:", err);
                return;
            }
            console.log(`Subscribed to ${channel}. Listening for LPUSH events...`);
        });

        subscriber.on("message", async (chan, message) => {
            console.log(`Received event on channel ${chan}: ${message}`);
            if (chan === channel) {
                console.log("LPUSH detected. Processing update...");
                try {
                    const data = await redis.lrange(key, 0, -1);
                    const parsedData = data.map(item => JSON.parse(item));
                    processUpdate(parsedData);
                } catch (err) {
                    console.error("Error processing update:", err);
                }
            }
        });

        // Keep the process alive
        process.on("SIGINT", () => {
            console.log("Terminating monitoring...");
            subscriber.unsubscribe(channel, () => {
                redis.quit();
                subscriber.quit();
                process.exit(0);
            });
        });

        console.log("Monitoring is active. Press Ctrl-C to exit.");
    } catch (error) {
        console.error("Error setting up key monitor:", error);
    }
}

// Example usage
monitorKeyUpdate("wif_logs:raw_object", (updatedData) => {
    console.log("Updated data:", updatedData);
});
