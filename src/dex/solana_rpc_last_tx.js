// Import required packages
const { Connection, PublicKey, clusterApiUrl, ParsedTransactionWithMeta  } = require('@solana/web3.js');

// Define the Solana connection (mainnet-beta)
const connection = new Connection('https://api.mainnet-beta.solana.com', 'confirmed');

const PROGRAM_ID = new PublicKey(process.argv[2]);

function calculateBalanceChanges(preTokenBalances, postTokenBalances) {
    const balanceChanges = [];

    // Create a map for easy lookup of pre balances by account
    const preBalancesMap = new Map();
    preTokenBalances.forEach(preBalance => {
        preBalancesMap.set(preBalance.accountIndex, {
            amount: Number(preBalance.uiTokenAmount.amount) / Math.pow(10, preBalance.uiTokenAmount.decimals),
            decimals: preBalance.uiTokenAmount.decimals,
            mint: preBalance.mint
        });
    });

    // Iterate over postTokenBalances to calculate changes
    postTokenBalances.forEach(postBalance => {
        const accountIndex = postBalance.accountIndex;
        const decimals = postBalance.uiTokenAmount.decimals;
        const postAmount = Number(postBalance.uiTokenAmount.amount) / Math.pow(10, decimals);

        const preBalanceData = preBalancesMap.get(accountIndex) || {
            amount: 0, decimals, mint: postBalance.mint
        };
        const preAmount = preBalanceData.amount;

        const change = postAmount - preAmount;

        // Only include accounts with a non-zero change
        if (change !== 0) {
            balanceChanges.push({
                accountIndex,
                mint: postBalance.mint,
                decimals,
                preAmount,
                postAmount,
                change
            });
        }
    });

    // Check if the changes sum up to zero
    const totalChange = balanceChanges.reduce((sum, entry) => sum + entry.change, 0);
    if (totalChange !== 0) {
        console.warn(`Warning: Total balance changes do not sum to zero! Total change: ${totalChange}`);
    }

    return balanceChanges;
}

async function getLatestTransaction() {
    //const { latestBlockhash } = await connection.getLatestBlockhash();

    const signatureInfo = await connection.getSignaturesForAddress(PROGRAM_ID,
        {
            limit: 10,
            maxSupportedTransactionVersion: 0
        });
    //console.log( signatureInfo);

    if (signatureInfo.length === 0) {
        console.log('No transactions found for this program ID');
        return;
    }

    for (const e of signatureInfo) {
        const signature = e.signature;
        
        // getParsedTransaction
        const transactionDetails = await connection.getTransaction(signature, 
            { 
                maxSupportedTransactionVersion: 0 
            });

        if (!transactionDetails) {
            console.log('Transaction details not found');
            return;
        }

        console.log('Transaction Details:', transactionDetails);
        console.log()
        console.log( 'fee: ', transactionDetails.meta.fee );
        //for (const fb of transactionDetails.meta.postTokenBalances ){
        //    console.log('\t', fb )
        //}
        const changes = calculateBalanceChanges(transactionDetails.meta.preTokenBalances, transactionDetails.meta.postTokenBalances);
        console.log('token changes: ', changes);
        
        //console.log(transactionDetails.transaction.message.compiledInstructions)
        transactionDetails.transaction.message.compiledInstructions.forEach((index,keys,data) => {
            //console.log(`Instruction ${index}:`);
            //console.log('- Accounts:', keys.map(k => k.pubkey.toBase58()));
            //console.log('- Data (Base64):', data);
        });

        /*const priceDetails = extractPriceInfo(transactionDetails);
        console.log('Extracted Price Information:', priceDetails);

        // Example: If the price information is encoded in the instruction data
        // You need to know how the program encodes the data (for example, price might be a uint64)
        if (transactionDetails.transaction.message.instructions != null) {
            transactionDetails.transaction.message.instructions.forEach((instruction) => {
                console.log(instruction.programId)
                if (instruction.programId && instruction.programId.equals(PROGRAM_ID)) {
                    // Parse the data from the instruction - this depends on the program's structure
                    const price = parsePriceData(instruction.data);
                    console.log('Price:', price);
                }
                else {
                    console.log('No program Id found:')
                    console.log(instruction);
                }
            });
        }
            */
    }

}


// Run the function to get the latest transaction price
getLatestTransaction().catch(console.error);