const { Connection, clusterApiUrl, PublicKey } = require('@solana/web3.js');

// Connect to Solana's WebSocket (using Mainnet)
const connection = new Connection(clusterApiUrl('mainnet-beta'), 'confirmed');

// Function to extract mint addresses from transaction instructions
function extractMintAddresses(instructions) {
    const mintAddresses = [];
    instructions.forEach((instruction) => {
        if (instruction.programId.toString() === 'TokenkegQfeZyiNwAJbMNzqfyFg7ftvkdFQw7cFkJ3LVA') {
            instruction.accounts.forEach((account) => {
                try {
                    const accountPublicKey = new PublicKey(account);
                    if (accountPublicKey.toString() !== '11111111111111111111111111111111') {
                        mintAddresses.push(accountPublicKey.toString());
                    }
                } catch (e) {
                    // Ignore invalid accounts
                }
            });
        }
    });
    return mintAddresses;
}

// Function to process new block
async function processNewBlock(blockHeight) {
    try {
        const block = await connection.getParsedBlock(blockHeight, { commitment: 'confirmed' });
        
        const transactions = block.transactions;
        console.log(`Found ${transactions.length} transactions in block ${blockHeight}`);

        transactions.forEach((tx, index) => {
            console.log(`Transaction ${index + 1}:`);
            console.log(`Status: ${tx.meta.err ? 'Failed' : 'Success'}`);

            // Extract mint addresses from the instructions
            //const mintAddresses = extractMintAddresses(tx.transaction.message.instructions);

            /*if (mintAddresses.length > 0) {
                console.log('Mint addresses:', mintAddresses);
            } else {
                console.log('No mint address found in this transaction.');
            }*/
        });
    } catch (err) {
        console.error(`Error processing block ${blockHeight}:`, err);
    }
}

// Subscribe to slot updates via WebSocket
connection.onSlotUpdate(async (slotUpdate) => {
    const blockHeight = slotUpdate.slot;
    console.log(`New block generated at slot: ${blockHeight}`);

    // Process the newly generated block
    await processNewBlock(blockHeight);
});

console.log('Listening for new blocks...');
