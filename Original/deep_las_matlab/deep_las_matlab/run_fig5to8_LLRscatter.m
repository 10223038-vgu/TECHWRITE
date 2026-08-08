%% run_fig5to8_LLRscatter.m
% Reproduces Figs. 5-8: scatter of actual (model-based Soft-LAS) vs
% approximated LLR (MLP-only = Figs. 5-6; full Deep LAS/GRU = Figs. 7-8)
% versus Re(Rx symbol), per bit.

clear; clc;

cfgList = struct('M', {4, 16}, 'snrdB', {4, 10});   % Fig.5 (4dB) / Fig.6 (10dB)
cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
nTest = 800;

for c = 1:numel(cfgList)
    M = cfgList(c).M;
    snrdB = cfgList(c).snrdB;

    dataFile = sprintf('train_%dQAM.mat', M);
    if ~isfile(dataFile)
        generateTrainingData(M, 0:2:14, 1000, dataFile);
    end
    mlpNet = trainMLP(dataFile, 2, 10, 300);
    gruNet = trainGRU(dataFile, mlpNet, 2, 100, 40);

    qh = qamHelpers(); Es = qh.symEnergy(M);
    sigma2 = Es / 10^(snrdB/10);

    B = log2(M);
    LLR_true_all = zeros(nTest, Nt*B);
    LLR_mlp_all  = zeros(nTest, Nt*B);
    LLR_deep_all = zeros(nTest, Nt*B);
    ReSym_all    = zeros(nTest, Nt);

    for t = 1:nTest
        symIdx = randi([0 M-1], Nt, 1);
        x = qh.mod(symIdx, M);
        H = genChannel(Nr, Nt);
        n = sqrt(sigma2/2)*(randn(Nr,1)+1i*randn(Nr,1));
        y = H*x + n;

        [~, LLR_true] = softOutputLAS(y, H, sigma2, M, 'mmse', cfg.maxLASIter);

        xhat_mmse_soft = initEstimateSoft(y, H, sigma2, M, 'mmse');  % FIX: soft
        xn = [real(xhat_mmse_soft); imag(xhat_mmse_soft)];
        xn = xn ./ max(abs(xn));
        llrMlpVec = mlpNet(xn);
        LLR_mlp = reshape(llrMlpVec, B, Nt).';

        LLR_deep = deepLASPredict(y, H, sigma2, M, mlpNet, gruNet);

        LLR_true_all(t,:) = reshape(LLR_true.', 1, []);
        LLR_mlp_all(t,:)  = reshape(LLR_mlp.',  1, []);
        LLR_deep_all(t,:) = reshape(LLR_deep.', 1, []);
        % FIX: plot the noisy/equalized value (continuous), matching
        % the paper's x-axis, NOT the ideal transmitted symbol (which
        % for e.g. 4-QAM only ever takes the discrete values +-1 and
        % would collapse the scatter to two vertical lines).
        ReSym_all(t,:) = real(xhat_mmse_soft).';
    end

    figure('Name', sprintf('LLR scatter, M=%d, SNR=%ddB', M, snrdB));
    for k = 1:Nt
        subplot(2, ceil(Nt/2), k); hold on; grid on;
        scatter(ReSym_all(:,k), LLR_true_all(:, (k-1)*B+1), 15, 'bo', 'DisplayName','Actual');
        scatter(ReSym_all(:,k), LLR_mlp_all(:,  (k-1)*B+1), 15, 'r.', 'DisplayName','MLP Approx.');
        xlabel(sprintf('\\Re(Rx symbol%d)', k)); ylabel(sprintf('l_{1,%d}', k));
        legend show;
    end
    sgtitle(sprintf('Figs. 5/6 style (MLP-only): M=%d, SNR=%d dB', M, snrdB));

    figure('Name', sprintf('Deep LAS LLR scatter, M=%d, SNR=%ddB', M, snrdB));
    for k = 1:Nt
        subplot(2, ceil(Nt/2), k); hold on; grid on;
        scatter(ReSym_all(:,k), LLR_true_all(:, (k-1)*B+1), 15, 'bo', 'DisplayName','Actual');
        scatter(ReSym_all(:,k), LLR_deep_all(:, (k-1)*B+1), 15, 'r.', 'DisplayName','Deep LAS Approx.');
        xlabel(sprintf('\\Re(Rx symbol%d)', k)); ylabel(sprintf('l_{1,%d}', k));
        legend show;
    end
    sgtitle(sprintf('Figs. 7/8 style (Deep LAS): M=%d, SNR=%d dB', M, snrdB));
end
