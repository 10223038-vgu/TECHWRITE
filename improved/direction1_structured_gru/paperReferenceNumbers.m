function ref = paperReferenceNumbers()
% PAPERREFERENCENUMBERS Headline numbers reported in the original paper's
% text (Ullah et al., IEEE TVT 2024), for use as reference points in
% comparison tables/plots. These are NOT digitized curve data (we don't
% have the paper's raw BER values point-by-point) -- they are the
% specific summary claims stated in the paper's Abstract and Section V,
% each with its source noted below. Treat any comparison against these as
% "does our reproduction/improvement land near the paper's claimed
% operating point," not a full curve overlay.
%
% All fields are [value_4QAM, value_16QAM] unless noted otherwise.

ref = struct();

% Abstract / Section V-B: "the proposed Deep LAS ... achieves the SNR
% gain of 2.55 dB and 3 dB with the conventional LAS while maintaining
% the BER of 10^-4 for 4-QAM and 16-QAM modulation, respectively."
ref.deepLAS_vs_convLAS_gainDB = [2.55, 3.0];
ref.deepLAS_vs_convLAS_atBER = 1e-4;

% Section V-B: "the rough LLR values approximated by the MLP block ...
% achieve the SNR gain of 1.9 dB and 1.8 dB with the conventional LAS"
% (same BER target as above, 10^-4).
ref.mlpOnly_vs_convLAS_gainDB = [1.9, 1.8];
ref.mlpOnly_vs_convLAS_atBER = 1e-4;

% Abstract: "provides a comparable signal-to-noise ratio (SNR) gap of
% 0.4 dB and 1.2 dB with the optimal soft output sphere decoding (SD) to
% achieve a BER of 10^-5 for 4-QAM and 16-QAM, respectively."
ref.deepLAS_vs_SD_gapDB = [0.4, 1.2];
ref.deepLAS_vs_SD_atBER = 1e-5;

% Fig. 10 annotations ("Approx.1dB SNR gap", "Approx.1.6dB SNR gap"):
% present in the figure but not spelled out unambiguously in the body
% text as to exactly which curve pair each annotation refers to. Included
% for completeness -- treat with more caution than the fields above.
ref.fig10_annotatedGapsDB_UNCERTAIN_PAIRING = [1.0, 1.6];

ref.modulationOrders = [4, 16];
ref.systemConfig = 'Nt = Nr = 4, Rayleigh fading, rate-1/2 turbo coding';
end
