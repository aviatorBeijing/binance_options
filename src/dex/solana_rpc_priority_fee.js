const { Connection, clusterApiUrl, Keypair, Transaction, SystemProgram } = require('@solana/web3.js');

async function fetchPriorityFee() {
    // Connect to the Solana cluster
    const connection = new Connection(clusterApiUrl('mainnet-beta'));

    try {
        // Generate a Keypair to act as the fee payer (use your wallet in production)
        const feePayer = Keypair.generate();

        // Fetch the latest blockhash
        const { blockhash } = await connection.getLatestBlockhash();

        // Create a simple transaction
        const transaction = new Transaction().add(
            SystemProgram.transfer({
                fromPubkey: feePayer.publicKey, // Sender's public key (fee payer in this case)
                toPubkey: Keypair.generate().publicKey, // Dummy receiver
                lamports: 1, // Minimal transfer amount
            })
        );

        // Set the fee payer and recentBlockhash
        transaction.feePayer = feePayer.publicKey;
        transaction.recentBlockhash = blockhash;

        // Estimate the fee for the transaction
        const feeCalculator = await connection.getFeeForMessage(transaction.compileMessage());

        console.log('Fee Payer Public Key:', feePayer.publicKey.toBase58());
        console.log('Blockhash:', blockhash);
        console.log('Estimated Fee:', feeCalculator?.value || 'Unavailable');

    } catch (error) {
        console.error('Error fetching priority fee:', error.message);
    }
}

fetchPriorityFee();
