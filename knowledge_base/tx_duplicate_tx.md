# SOP: Duplicate Transaction Prevention Guard

Rules preventing accidental duplicate debits on the Eko network.

## Rules
1. Eko blocks identical transactions (same beneficiary, same amount) initiated within 5 minutes.
2. If a duplicate debit occurs due to bank errors, the merchant must file a reconciliation dispute immediately with both transaction UTRs.