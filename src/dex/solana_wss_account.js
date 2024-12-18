const { Connection, PublicKey, clusterApiUrl } = require('@solana/web3.js');

// Set up the connection to the Solana devnet (or mainnet or testnet)
const connection = new Connection(clusterApiUrl('devnet'), 'confirmed');

// Replace this with the actual account you want to subscribe to
const accountAddress = 'FikHnt8i9Qeifcfy2d5LSGBhrSKvmDXnLFGrDfBCMVCo'; //on AWS

// Convert to PublicKey object
const publicKey = new PublicKey(accountAddress);

// Subscribe to the account
connection.onAccountChange(publicKey, (accountInfo, context) => {
  console.log('Account Change detected:');
  
  // Parse and log the critical information from the accountInfo
  const lamports = accountInfo.lamports; // The balance in lamports (1 SOL = 1 billion lamports)
  const owner = accountInfo.owner.toString(); // The owner of the account
  const data = accountInfo.data.toString('utf-8'); // Account data (if any)
  const executable = accountInfo.executable; // Boolean: Is the account executable (like a smart contract)?
  const rentEpoch = accountInfo.rentEpoch; // The epoch the account's rent was paid for

  // Output the relevant information
  console.log(`Account balance: ${lamports / 1e9} SOL`); // Convert lamports to SOL
  console.log(`Owner: ${owner}`);
  console.log(`Data: ${data}`);
  console.log(`Executable: ${executable}`);
  console.log(`Rent Epoch: ${rentEpoch}`);
}, 'confirmed');
