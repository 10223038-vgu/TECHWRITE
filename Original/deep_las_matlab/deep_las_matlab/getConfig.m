function cfg = getConfig()
% GETCONFIG Common system parameters used across all scripts.
% Matches the simulation setup described in Section V of the paper
% (Ullah et al., "Soft-Output Deep LAS Detection for Coded MIMO
% Systems: A Learning-Aided LLR Approximation", IEEE TVT 2024).

cfg.Nt = 4;                  % transmit antennas
cfg.Nr = 4;                  % receive antennas
cfg.M_list = [4 16];         % QAM orders (4-QAM, 16-QAM)
cfg.FFT_len = 1024;          % OFDM FFT/IFFT length
cfg.symbolsPerBlock = 8;     % symbols per channel use (per paper)
cfg.turboRate = 1/2;
cfg.turboIterations = 8;
cfg.SNRdB_range = 0:2:16;    % sweep for BER curves
cfg.nBlocksPerSNR = 2000;    % Monte-Carlo blocks per SNR point
                              % (paper uses far more; reduce for quick tests)
cfg.maxLASIter = 50;         % safety cap on LAS inner loop
end
