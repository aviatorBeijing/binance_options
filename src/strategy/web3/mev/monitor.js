const Web3 = require('web3');

// Parse command line arguments
// node index.js abcdef1234567890abcdef1234567890abcdef12 0x5a5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b 0.5
const args = process.argv.slice(2);
if (args.length < 2) {
    console.error('Usage: node index.js <INFURA_PROJECT_ID> <TARGET_ADDRESS> [BALANCE_THRESHOLD_IN_ETHER]');
    process.exit(1);
}

const infuraProjectId = args[0];
const targetAddress = args[1];
const balanceThreshold = args.length >= 3 ? Web3.utils.toWei(args[2], 'ether') : Web3.utils.toWei('1', 'ether'); // Default to 1 ETH if not provided

// Connect to an Ethereum node
const web3 = new Web3(new Web3.providers.WebsocketProvider(`wss://mainnet.infura.io/ws/v3/${infuraProjectId}`));

console.log(`Listening for pending transactions involving ${targetAddress} with a balance threshold of ${Web3.utils.fromWei(balanceThreshold, 'ether')} ETH...`);

// Subscribe to pending transactions
web3.eth.subscribe('pendingTransactions', async (error, txHash) => {
    if (error) {
        console.error('Error subscribing to pending transactions:', error);
        return;
    }

    try {
        const tx = await web3.eth.getTransaction(txHash);
        if (tx && tx.to && tx.to.toLowerCase() === targetAddress.toLowerCase() && web3.utils.toBN(tx.value).gt(web3.utils.toBN(balanceThreshold))) {
            console.log('Found a transaction involving target address with value > threshold:', tx);
            // Add logic to exploit the opportunity
            // Example: Arbitrage or liquidation
        }
    } catch (err) {
        console.error('Error fetching transaction:', err);
    }

    // Sleep to prevent overwhelming the node
    await new Promise(resolve => setTimeout(resolve, 10));
});

