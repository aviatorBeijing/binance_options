const { Connection, clusterApiUrl, PublicKey } = require('@solana/web3.js');

(async () => {
  try {
    // Replace with your Chainstack RPC endpoint
    const RPC_ENDPOINT = 'https://solana-mainnet.core.chainstack.com/c01499a742050fe60ae86d122d91c3fa';
    const connection = new Connection(RPC_ENDPOINT, 'confirmed');

    // Fetch the latest confirmed block
    const slot = await connection.getSlot('confirmed');

    // Get the block details for the latest slot
    const block = await connection.getBlock(slot, {maxSupportedTransactionVersion: 0});

    if (!block || !block.transactions.length) {
      console.log('No transactions found in the latest block');
      return;
    }

    // Get the first transaction in the block
    const transaction = block.transactions[0];

    // Extract transaction details
    const { transaction: { message }, meta } = transaction;

    // Extract account keys and instructions
    const accountKeys = message.accountKeys.map(key => key.toBase58());
    const instructions = message.instructions;

    instructions.forEach((instruction, index) => {
      console.log(`Instruction ${index + 1}:`);
      console.log('Program ID:', accountKeys[instruction.programIdIndex]);
      console.log('Accounts:', instruction.accounts.map(i => accountKeys[i]));
      console.log('Data (base64):', instruction.data);

      // Parse data if needed (custom parsing depending on program)
    });

    // Extract asset IDs and amounts from token transactions
    if (meta && meta.postTokenBalances.length > 0) {
      meta.postTokenBalances.forEach((balance, index) => {
        console.log(`Token Balance ${index + 1}:`);
        console.log('Mint (Asset ID):', balance.mint);
        console.log('Owner:', balance.owner);
        console.log('UI Amount:', balance.uiTokenAmount.uiAmountString);
      });
    }
  } catch (error) {
    console.error('Error fetching transactions:', error);
  }
})();
