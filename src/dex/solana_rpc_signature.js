const axios = require('axios');

// Solana RPC endpoint
const RPC_URL = "https://api.mainnet-beta.solana.com";

function deepLog(obj, depth = 0) {
    const indent = '  '.repeat(depth); // Two spaces per depth level for indentation
    if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
        for (const key in obj) {
            if (obj.hasOwnProperty(key)) {
                console.log(`${indent}${key}:`);
                deepLog(obj[key], depth + 1);
            }
        }
    } else if (Array.isArray(obj)) {
        obj.forEach((item, index) => {
            console.log(`${indent}[${index}]:`);
            deepLog(item, depth + 1);
        });
    } else {
        console.log(`${indent}${obj}`);
    }
}
// Function to get transaction details by signature
async function getTransaction(signature) {
    const payload = {
        jsonrpc: "2.0",
        id: 1,
        method: "getTransaction",
        params: [
            signature,
            {
                commitment: "confirmed", // Options: "processed", "confirmed", "finalized"
                maxSupportedTransactionVersion: 0 // Optional: Use if you're on versioned transactions
            }
        ]
    };

    try {
        // Make the RPC call
        const response = await axios.post(RPC_URL, payload);

        if (response.data && response.data.result) {
            const rst = response.data.result;
            //console.log("Transaction Details:", response.data.result);
            deepLog( response.data.result );
            return response.data.result;
        } else {
            console.error("Error fetching transaction:", response.data);
        }
    } catch (error) {
        console.error("RPC Error:", error.message);
    }
}

// Example transaction signature
const transactionSignature = process.argv[2];

// Call the function
getTransaction(transactionSignature);
