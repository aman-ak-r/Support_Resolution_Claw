# FAQ: Aadhaar Enabled Payment System (AePS) Failures
    
Q: The customer's account was debited, but the AePS transaction failed in the app. Where is the money?
A: This is a bank-end reversal case:
1. When UIDAI or bank servers fail, the transaction is marked Failed in the Eko app, but the bank may have debited the customer's account.
2. The NPCI guidelines dictate that such debited amounts are automatically refunded back to the customer's bank account within 3 to 5 business days.
3. Eko has no control over reversing customer accounts since the money never reached Eko. Advise the customer to wait for the automatic bank reversal.