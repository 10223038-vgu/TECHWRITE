%% run_quick_demo.m
% Fast smoke test (small sizes) to check the whole pipeline runs
% end-to-end before committing to the full-scale sweeps in the
% other run_*.m scripts. Should complete in well under a minute.

clear; clc;
cfg = getConfig();
M = 4;
qh = qamHelpers();
Es = qh.symEnergy(M);
snrdB = 8;
sigma2 = Es / 10^(snrdB/10);

fprintf('--- Single-symbol sanity check ---\n');
x = qh.mod(randi([0 M-1], cfg.Nt, 1), M);
H = genChannel(cfg.Nr, cfg.Nt);
n = sqrt(sigma2/2)*(randn(cfg.Nr,1)+1i*randn(cfg.Nr,1));
y = H*x + n;

[xhat, LLR, F0] = softOutputLAS(y, H, sigma2, M, 'mmse', cfg.maxLASIter);
disp('True x:'); disp(x.');
disp('Detected xhat:'); disp(xhat.');
disp('LLR matrix:'); disp(LLR);
fprintf('F0 (min cost) = %.4f\n\n', F0);

fprintf('--- Tiny dataset + tiny MLP/GRU training ---\n');
generateTrainingData(M, [4 8], 50, 'quick_demo_data.mat');
mlpNet = trainMLP('quick_demo_data.mat', 1, 6, 30);
gruNet = trainGRU('quick_demo_data.mat', mlpNet, 1, 20, 5);

fprintf('--- Deep LAS single-sample inference ---\n');
LLR_deep = deepLASPredict(y, H, sigma2, M, mlpNet, gruNet);
disp('Deep LAS LLR:'); disp(LLR_deep);

fprintf('--- Tiny BER comparison (few blocks, just to confirm it runs) ---\n');
SNRdB_range = [4 8];
ber_soft = simulateBER('softlas', M, SNRdB_range, 20);
ber_deep = simulateBER('deeplas', M, SNRdB_range, 20, mlpNet, gruNet);

fprintf('Quick demo complete.\n');
