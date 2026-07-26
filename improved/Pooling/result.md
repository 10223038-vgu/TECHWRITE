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

   Distance pooling:
   Warning: Target BER 1.0e-04 is outside the measured range [3.6e-03, 1.1e-01] -- extrapolating. 
> In snrAtTargetBER (line 44)
In run_distance_pooling_vs_original (line 100) 
Warning: Target BER 1.0e-04 is outside the measured range [1.0e-02, 1.2e-01] -- extrapolating. 
> In snrAtTargetBER (line 44)
In run_distance_pooling_vs_original (line 101) 
Warning: Target BER 1.0e-04 is outside the measured range [8.3e-03, 1.1e-01] -- extrapolating. 
> In snrAtTargetBER (line 44)
In run_distance_pooling_vs_original (line 102) 
Warning: Target BER 1.0e-04 is outside the measured range [8.1e-03, 1.8e-01] -- extrapolating. 
> In snrAtTargetBER (line 44)
In run_distance_pooling_vs_original (line 103) 

================= SNR-GAIN SUMMARY (BER=1e-4 target) =================
Quantity                                           Value (dB)
Paper: Deep LAS gain vs Conv. LAS                        2.55
Ours: Original Deep LAS gain vs Conv. LAS              -28.97
Ours: Uniform pooling gain vs Original (previous result)      27.38
Ours: Distance-aware pooling gain vs Original (THE novelty claim)      26.71
Ours: Distance-aware gain vs uniform pooling (isolates THIS change)      -0.68
========================================================================

Interpretation:
 - Row 4 vs Row 3: is distance-awareness actually better than uniform pooling?
 - Row 4 on its own: does it clearly beat the Original reproduction (the real target)?
 - If Row 5 is positive but Row 4 is still <= 0, distance-awareness helped but has not
   yet closed the remaining gap to Original -- consider nearFrac/decayScale tuning next.

Saved comparison table to result/distance_pooling_snr_gain_comparison.csv
Saved the trained distance-aware pooling net and curves to result/.
