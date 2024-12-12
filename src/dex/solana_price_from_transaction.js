const { Connection, PublicKey } = require('@solana/web3.js');

// Define the Solana connection (mainnet-beta)
const connection = new Connection('https://api.mainnet-beta.solana.com', 'confirmed');

// Replace this with the Program ID you are interested in
console.log( process.argv );
const PROGRAM_ID = new PublicKey(process.argv[2]);

// Function to get the latest transaction for a specific program ID
async function getLatestTransaction() {
  // Get the most recent blockhash
  //const latestBlockhash = await connection.getRecentBlockhash();
  const { latestBlockhash } = await connection.getLatestBlockhash();

  // Get the list of transactions within the latest block (You can adjust based on your needs)
  const signatureInfo = await connection.getSignaturesForAddress(PROGRAM_ID, { limit: 10, maxSupportedTransactionVersion: 0 });
  console.log( signatureInfo);

  if (signatureInfo.length === 0) {
    console.log('No transactions found for this program ID');
    return;
  }

  // Get the signature of the latest transaction
  //const signature = signatureInfo[0].signature;

  for (const e of signatureInfo) {
       const signature =  e.signature;
  // Fetch the full transaction details
  const transactionDetails = await connection.getTransaction(signature, {maxSupportedTransactionVersion: 0} );

  if (!transactionDetails) {
    console.log('Transaction details not found');
    return;
  }

  // Log the transaction details (or inspect it to find the price info)
  console.log('Transaction Details:', transactionDetails);
  const priceDetails = extractPriceInfo(transactionDetails);
  console.log('Extracted Price Information:', priceDetails);

  // Example: If the price information is encoded in the instruction data
  // You need to know how the program encodes the data (for example, price might be a uint64)
  if( transactionDetails.transaction.message.instructions!=null){
	  transactionDetails.transaction.message.instructions.forEach((instruction) => {
    if (instruction.programId  && instruction.programId.equals(PROGRAM_ID)) {
      // Parse the data from the instruction - this depends on the program's structure
      const price = parsePriceData(instruction.data);
      console.log('Price:', price);
    }
    else{
      console.log('No program Id found:')
      console.log( instruction );
    }
  });
  }
  }

}

// Sample Transaction Details
const transaction = {
    // Assuming the provided transaction details are here
    meta: {
        logMessages: [
            // Include logMessages from your provided data
            'Program log: fee_growth: 368075460682',
            'Program log: 净利润: 5466',
            // Add more logs here if needed
        ]
    }
};

// Function to extract price-related data
function extractPriceInfo(transaction) {
  const logMessages = transaction.meta.logMessages;
  const priceInfo = [];

  logMessages.forEach((log) => {
    const feeGrowthMatch = log.match(/fee_growth:\s(\d+)/);
    if (feeGrowthMatch) {
      priceInfo.push({
        type: 'fee_growth',
        value: parseInt(feeGrowthMatch[1], 10)
      });
    }

    const swapInstructionMatch = log.match(/Instruction:\s(SwapV2|Swap)/);
    if (swapInstructionMatch) {
      priceInfo.push({
        type: 'instruction',
        value: swapInstructionMatch[1]
      });
    }
  });

  return priceInfo;
}

// Sample function to decode price (depends on program encoding)
function parsePriceData(data) {
  // This function should be customized based on the program's instruction data encoding
  // Assuming the price is a uint64 value encoded in the first 8 bytes
  const priceBuffer = Buffer.from(data, 'base64');
  const price = priceBuffer.readBigUInt64LE(0); // Read as a 64-bit unsigned integer
  return price.toString(); // Returning price as string for simplicity
}

// Run the function to get the latest transaction price
getLatestTransaction().catch(console.error);