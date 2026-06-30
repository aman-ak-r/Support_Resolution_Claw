# SOP: Handling Transaction Timeouts and Gateway Timeouts

This document details actions to take when an AePS or DMT transaction times out.

## Instructions
1. **Pending Status**: If transaction status is pending, wait 2 hours for auto-reconciliation.
2. Do not re-initiate the same transaction immediately, as it may cause a duplicate debit.
3. If the gateway reconciles to FAILED, funds are automatically returned to the Trade Wallet.