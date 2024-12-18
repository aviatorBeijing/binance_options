const { Connection, PublicKey, clusterApiUrl } = require('@solana/web3.js');

// Constants for program loaders
const BPF_LOADER = new PublicKey("BPFLoader1111111111111111111111111111111111");
const BPF_LOADER_UPGRADEABLE = new PublicKey("BPFLoaderUpgradeab1e11111111111111111111111");

// Solana cluster
const SOLANA_RPC_URL = clusterApiUrl('mainnet-beta'); // Use "devnet" for testing
const connection = new Connection(SOLANA_RPC_URL, 'confirmed');

// Fetch recent transactions for BPF loader
const findLastDeployedProgram = async () => {
    try {
        // Fetch recent signatures for BPF Loader Upgradeable
        console.log("Fetching recent transactions...");
        const signatures = await connection.getSignaturesForAddress(
            BPF_LOADER_UPGRADEABLE, { limit: 3 });

        for (const sigInfo of signatures) {
            const { signature } = sigInfo;

            // Fetch detailed transaction
            console.log(`Checking transaction: ${signature}`);
            const transaction = await connection.getTransaction(
                signature, { 
                    commitment: 'confirmed' 
                });

            if (!transaction) continue;

            const { meta, transaction: tx } = transaction;
            const logs = meta?.logMessages || [];
            const instructions = tx?.message?.instructions || [];

            // Look for program deployment patterns in logs or instructions
            const isDeployment = logs.some(log => log.includes("BPFLoaderUpgradeab1e"));

            if (isDeployment) {
                console.log("New Program Deployment Detected!");
                console.log("Transaction Signature:", signature);

                // Extract deployed program's public key
                for (const instr of instructions) {
                    if (instr.programId.equals(BPF_LOADER_UPGRADEABLE)) {
                        const programKey = instr.accounts?.[0]; // Usually the first account is the program key
                        console.log("Deployed Program PublicKey:", programKey.toBase58());
                        return programKey.toBase58();
                    }
                }
            }
        }
    } catch (err) {
        console.error("Error tracing deployed programs:", err);
    }
};

findLastDeployedProgram();
