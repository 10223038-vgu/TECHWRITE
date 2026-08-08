%% run_distance_pooling_vs_original.m
% Tests THEORY_POOLING.md Section 6's proposal directly: does making
% pooling DISTANCE-AWARE (near/far split + exponentially-decaying-weighted
% mean, instead of one uniform pool over the whole context window) turn
% the "rough parity with Original Deep LAS" result from
% run_pooling_vs_original.m into a real, clear win?
%
% Three-way comparison, all on identical data/SNR range/training budget:
%   - Original Deep LAS (our reproduction)
%   - Uniform pooling net (mean/max/std over the whole context -- the
%     previous "ours" arm)
%   - Distance-aware pooling net (near/far split + distance-weighted mean
%     -- the new "ours" arm)
% plus the classical baselines (MMSE, Conv. LAS) for reference.
%
% Requires original/deep-las-matlab on the MATLAB path (same as the
% other run_*.m scripts in this folder).

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% ---- Path to the original codebase ----
% EDIT THIS LINE to match your setup (same as the other run_*.m scripts).
origDir = 'D:\Study\3rd year\6th Semester\TW\Original\deep_las_matlab\deep_las_matlab';

if isfolder(origDir)
    addpath(origDir);
end
if exist('genChannel', 'file') ~= 2
    error(['Cannot find the original codebase on the MATLAB path (genChannel.m ' ...
           'was not found).\nEdit the ''origDir'' variable near the top of this ' ...
           'script (currently: %s) to point at the folder containing ' ...
           'genChannel.m, qamHelpers.m, trainMLP.m, etc.'], origDir);
end

M = 4;
% Extended from 0:1:14 -- at 14 dB every curve was still sitting at
% BER ~ 1e-2 to 1e-1, nowhere near the 1e-4/1e-5 target lines, which is
% exactly why snrAtTargetBER had to extrapolate wildly (and produced
% meaningless dB values -- always sanity-check that the printed table's
% warnings say "outside the measured range" before trusting its numbers).
SNRdB_range = 0:2:20;
% Reaching low BER needs many more bits simulated per point: at BER~1e-4
% you need ~1.5e6 bits (150 errors / 1e-4) to get a reliable estimate;
% the old minErrors=150/maxBlocksFactor=40 defaults capped out at
% nBlocksMin*maxBlocksFactor target subcarriers long before that. Bumped
% nBlocksMin and pass explicit (larger) minErrors/maxBlocksFactor to our
% custom simulators below. This is a real runtime cost -- expect this
% script to take noticeably longer than before, especially at the
% highest SNR points where errors are rare.
nBlocksMin = 1000;
ber_minErrors = 100;
ber_maxBlocksFactor = 2000;   % max target subcarriers per SNR point = nBlocksMin*maxBlocksFactor = 2,000,000
ref = paperReferenceNumbers();
modIdx = find(ref.modulationOrders == M, 1);

N = 256; L = 4; tau = 6; nRealizations = 400; maxEpochs = 60; stride = 8;
% nRealizations bumped 200 -> 400: the previous run extended SNRdB_range
% from 0:1:14 (15 points) to 0:2:20 (11 points) but LEFT nRealizations at
% 200, so each SNR point actually got *less* round-robin coverage than
% before (200/11 ~= 18 realizations/point vs 200/15 ~= 13... actually
% slightly more per point by raw count, but the high-SNR points are also
% now the ones where the network needs the most precision to keep
% improving, and that's exactly where the pooling nets floored out and
% diverged from Original -- see the plot from the 0:2:20 run). Doubling
% nRealizations tests directly whether that floor is a fixable
% data-density problem or a real representational ceiling: if the floor
% moves/lowers with more data, it was density; if it doesn't, it's the
% architecture. This roughly doubles the training-data-build time for
% both pooling arms (though NOT the BER-evaluation time, which is
% unaffected -- see ber_minErrors/ber_maxBlocksFactor above).
Bc = coherenceBandwidth(N, L);
Lseq = round(Bc);
nearFrac = 0.25;
decayScale = Lseq/2;
fprintf('N=%d, L=%d -> Bc=%.1f, Lseq=%d, nearFrac=%.2f, decayScale=%.1f\n', ...
    N, L, Bc, Lseq, nearFrac, decayScale);
fprintf('SNR range: %d:%d:%d dB\n', SNRdB_range(1), SNRdB_range(2)-SNRdB_range(1), SNRdB_range(end));

%% --- Baselines + our reproduced original Deep LAS ---
% Filename tags the SNR range so extending the range can't silently
% reuse flat-MLP/GRU training data built for a narrower one -- the same
% train/test SNR mismatch bug fixed earlier in run_direction1_ablation.m,
% just for the ORIGINAL codebase's own MLP/GRU this time.
dataFileFlat = fullfile(origDir, sprintf('train_%dQAM_snr%dto%d.mat', M, SNRdB_range(1), SNRdB_range(end)));
if ~isfile(dataFileFlat)
    generateTrainingData(M, SNRdB_range, 3000, dataFileFlat);
end
mlpNet = trainMLP(dataFileFlat, 2, 10, 300);
gruNet_orig = trainGRU(dataFileFlat, mlpNet, 2, 100, 40);

fprintf('\n--- Baselines & reproduced original Deep LAS ---\n');
ber_mmse        = simulateBER('mmse-hard',    M, SNRdB_range, nBlocksMin);
ber_convLAS     = simulateBER('convlas-hard', M, SNRdB_range, nBlocksMin);
ber_origDeepLAS = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gruNet_orig);

%% --- Uniform pooling net: reuse cache ONLY if its config (including SNR
% range) matches this run; otherwise retrain, to avoid the exact
% train/test SNR mismatch bug fixed earlier ---
cachedUniform = fullfile(thisDir, 'result', 'best_pooling_net.mat');
uniformCacheOK = false;
if isfile(cachedUniform)
    loadedU = load(cachedUniform);
    if isequal(loadedU.N, N) && isequal(loadedU.L, L) && isequal(loadedU.Lseq, Lseq) && ...
       isfield(loadedU, 'trainSNRrange') && isequal(loadedU.trainSNRrange, SNRdB_range) && ...
       isfield(loadedU, 'nRealizations') && isequal(loadedU.nRealizations, nRealizations)
        fprintf('\nReusing cached uniform pooling net from %s (config matches).\n', cachedUniform);
        poolNetUniform = loadedU.poolNet;
        uniformCacheOK = true;
    else
        fprintf('\nCached uniform pooling net config/SNR-range/nRealizations does not match this run -- retraining.\n');
    end
end
if ~uniformCacheOK
    poolDataUniform = buildPooledContextFeatures(M, SNRdB_range, Lseq, N, L, tau, nRealizations, mlpNet, stride);
    poolNetUniform = trainPoolingNet(poolDataUniform, 128, maxEpochs);
    poolNet = poolNetUniform; %#ok<NASGU> -- saved under the field name run_pooling_vs_original.m also expects
    trainSNRrange = SNRdB_range; %#ok<NASGU>
    save(cachedUniform, 'poolNet', 'mlpNet', 'N', 'L', 'Lseq', 'tau', 'trainSNRrange', 'nRealizations', '-mat');
end
ber_poolUniform = simulateBERPooling(M, SNRdB_range, nBlocksMin, mlpNet, poolNetUniform, Lseq, N, L, tau, ber_minErrors, ber_maxBlocksFactor);

%% --- Distance-aware pooling net (the new arm being tested) ---
poolDataDist = buildDistanceWeightedPooledFeatures(M, SNRdB_range, Lseq, N, L, tau, nRealizations, mlpNet, stride, nearFrac, decayScale);
poolNetDist = trainPoolingNet(poolDataDist, 128, maxEpochs);
ber_poolDist = simulateBERDistancePooling(M, SNRdB_range, nBlocksMin, mlpNet, poolNetDist, Lseq, N, L, tau, nearFrac, decayScale, ber_minErrors, ber_maxBlocksFactor);

%% --- Figure: BER vs SNR comparison ---
figure('Name', 'Distance-aware pooling vs. uniform pooling vs. Original');
semilogy(SNRdB_range, ber_mmse, ':x', 'DisplayName', 'MMSE (hard, sanity)'); hold on; grid on;
semilogy(SNRdB_range, ber_convLAS, '-o', 'DisplayName', 'Conv. LAS (hard)');
semilogy(SNRdB_range, ber_origDeepLAS, '-d', 'DisplayName', 'Original Deep LAS (our reproduction)', 'LineWidth', 1.5);
semilogy(SNRdB_range, ber_poolUniform, '-s', 'Color', [0.13 0.55 0.13], ...
    'DisplayName', 'Uniform pooling net', 'LineWidth', 1.2);
semilogy(SNRdB_range, ber_poolDist, '-^', 'Color', [0.85 0.10 0.10], ...
    'DisplayName', 'Distance-aware pooling net (ours)', 'LineWidth', 1.5);

yline(1e-4, '--k', 'BER=10^{-4} (paper''s Deep-LAS-vs-Conv-LAS target)', 'LabelHorizontalAlignment','left');
yline(1e-5, ':k',  'BER=10^{-5} (paper''s Deep-LAS-vs-SD target)', 'LabelHorizontalAlignment','left');

set(gca, 'YScale', 'log'); ylim([1e-5 1]);
xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location','southwest');
title(sprintf('%d-QAM: distance-aware pooling vs. uniform pooling vs. Original', M));

%% --- SNR-gain table ---
snr_convLAS_at1e4     = snrAtTargetBER(SNRdB_range, ber_convLAS, 1e-4);
snr_origDeepLAS_at1e4 = snrAtTargetBER(SNRdB_range, ber_origDeepLAS, 1e-4);
snr_poolUniform_at1e4 = snrAtTargetBER(SNRdB_range, ber_poolUniform, 1e-4);
snr_poolDist_at1e4    = snrAtTargetBER(SNRdB_range, ber_poolDist, 1e-4);

gain_orig_vs_convLAS = snr_convLAS_at1e4 - snr_origDeepLAS_at1e4;
gain_uniform_vs_orig = snr_origDeepLAS_at1e4 - snr_poolUniform_at1e4;
gain_dist_vs_orig    = snr_origDeepLAS_at1e4 - snr_poolDist_at1e4;
gain_dist_vs_uniform = snr_poolUniform_at1e4 - snr_poolDist_at1e4;

fprintf('\n================= SNR-GAIN SUMMARY (BER=1e-4 target) =================\n');
fprintf('%-50s %10s\n', 'Quantity', 'Value (dB)');
fprintf('%-50s %10.2f\n', 'Paper: Deep LAS gain vs Conv. LAS', ref.deepLAS_vs_convLAS_gainDB(modIdx));
fprintf('%-50s %10.2f\n', 'Ours: Original Deep LAS gain vs Conv. LAS', gain_orig_vs_convLAS);
fprintf('%-50s %10.2f\n', 'Ours: Uniform pooling gain vs Original (previous result)', gain_uniform_vs_orig);
fprintf('%-50s %10.2f\n', 'Ours: Distance-aware pooling gain vs Original (THE novelty claim)', gain_dist_vs_orig);
fprintf('%-50s %10.2f\n', 'Ours: Distance-aware gain vs uniform pooling (isolates THIS change)', gain_dist_vs_uniform);
fprintf('========================================================================\n');

fprintf('\nInterpretation:\n');
fprintf(' - Row 4 vs Row 3: is distance-awareness actually better than uniform pooling?\n');
fprintf(' - Row 4 on its own: does it clearly beat the Original reproduction (the real target)?\n');
fprintf(' - If Row 5 is positive but Row 4 is still <= 0, distance-awareness helped but has not\n');
fprintf('   yet closed the remaining gap to Original -- consider nearFrac/decayScale tuning next.\n');

resultsTable = table( ...
    {'Paper: DeepLAS vs ConvLAS'; 'Ours: OrigDeepLAS vs ConvLAS'; 'Ours: UniformPooling vs Orig'; 'Ours: DistancePooling vs Orig'; 'Ours: DistancePooling vs UniformPooling'}, ...
    [ref.deepLAS_vs_convLAS_gainDB(modIdx); gain_orig_vs_convLAS; gain_uniform_vs_orig; gain_dist_vs_orig; gain_dist_vs_uniform], ...
    'VariableNames', {'Quantity', 'SNRgainDB'});
writetable(resultsTable, fullfile(thisDir, 'result', 'distance_pooling_snr_gain_comparison.csv'));
fprintf('\nSaved comparison table to result/distance_pooling_snr_gain_comparison.csv\n');

save(fullfile(thisDir, 'result', 'best_distance_pooling_net.mat'), ...
    'poolNetDist', 'mlpNet', 'N', 'L', 'Lseq', 'tau', 'nearFrac', 'decayScale');
save(fullfile(thisDir, 'result', 'distance_pooling_vs_original_curves.mat'), ...
    'SNRdB_range', 'ber_mmse', 'ber_convLAS', 'ber_origDeepLAS', 'ber_poolUniform', 'ber_poolDist');
fprintf('Saved the trained distance-aware pooling net and curves to result/.\n');
