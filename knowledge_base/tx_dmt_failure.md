# SOP: Domestic Money Transfer (DMT) Failures
    
When a Domestic Money Transfer (DMT) fails:
1. Check the transaction status in the Eko app history.
2. **If status is FAILED**: The money is refunded back to the merchant's Eko Trade Wallet instantly.
3. **If status is PENDING**: The money is held at the bank gateway. Eko will query the partner bank. Do not retry the transaction immediately. Wait 2 hours. If unresolved, it will automatically change to SUCCESS or FAILED.
4. **If status is SUCCESS but beneficiary has not received funds**: Obtain the Unique Transaction Reference (UTR) number from the receipt and ask the customer to check with their bank.