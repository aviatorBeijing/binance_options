const fetch = require("node-fetch");

const JUPITER_API_URL = "https://quote-api.jup.ag/v6/quote";
const SOL_MINT = "So11111111111111111111111111111111111111112"; // SOL mint address
const USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"; // USDC mint address

async function getLatestQuote(solAmount) {
    try {
        const amountInLamports = Math.floor(solAmount * 10 ** 9); // Convert SOL to lamports
        const slippageBps = 50; // 0.5% slippage

        const url = `${JUPITER_API_URL}?inputMint=${SOL_MINT}&outputMint=${USDC_MINT}&amount=${amountInLamports}&slippageBps=${slippageBps}`;

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`Failed to fetch quote: ${response.status} ${response.statusText}`);
        }

        const quoteResponse = await response.json();

        console.log(quoteResponse);
        for (const routePlan of quoteResponse.routePlan) {
            console.log(routePlan);
        }

        if (quoteResponse.data && quoteResponse.data.length > 0) {
            const bestQuote = quoteResponse.data[0];
            console.log("Best Quote:", bestQuote);
            return bestQuote;
        } else {
            console.log("No swap quotes available.");
            return null;
        }
    } catch (error) {
        console.error("Error fetching quote:", error.message);
    }
}

// Example usage: Fetch quote for swapping 1 SOL
getLatestQuote(1);
