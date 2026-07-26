================= SNR-GAIN SUMMARY (BER=1e-4 target) =================
Quantity                                      Value (dB)
Paper: Deep LAS gain vs Conv. LAS                   2.55
Ours: Original Deep LAS gain vs Conv. LAS (reproduction)      -0.23
Ours: Pooling net gain vs Conv. LAS                 2.27
Ours: Pooling net gain vs Original Deep LAS (the novelty claim)       2.50
========================================================================

Interpretation:
 - Row 1 vs Row 2: sanity check -- is our reproduction in the same ballpark as the paper's claim?
 - Row 4: the actual novelty result -- positive means the pooling net beats our own reproduced baseline.
 - Unlike Direction 1's GRU, the pooling net's advantage over the GRU (see run_pooling_vs_gru.m)
   WIDENED at high SNR rather than shrinking -- if Row 4 is also positive and grows with SNR,
   that is a much stronger and more paper-worthy result than the GRU ever produced.
