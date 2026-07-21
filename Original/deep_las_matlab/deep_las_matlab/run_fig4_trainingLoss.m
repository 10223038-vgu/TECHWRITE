%% run_fig4_trainingLoss.m
% Reproduces Fig. 4: training loss vs epoch for (a) MLP with
% 1/2/3 fully-connected hidden layers, (b) GRU block with 1/2/3
% GRU layers.

clear; clc;
M = 4;
dataFile = sprintf('train_%dQAM.mat', M);
if ~isfile(dataFile)
    generateTrainingData(M, 0:2:14, 1000, dataFile);
end

%% (a) MLP: sweep number of FC hidden layers
figure;
subplot(1,2,1); hold on; grid on;
for nLayers = 1:3
    [~, info] = trainMLP(dataFile, nLayers, 10, 300);
    plot(info.perf, 'DisplayName', sprintf('FC layers = %d', nLayers));
end
xlabel('Epoch'); ylabel('Training Loss'); legend show;
title('Fig. 4(a): MLP training loss');

%% (b) GRU: sweep number of GRU layers
mlpNet = trainMLP(dataFile, 2, 10, 300);   % fixed MLP for the GRU sweep

subplot(1,2,2); hold on; grid on;
for nGRU = 1:3
    [~, trainInfo] = trainGRU(dataFile, mlpNet, nGRU, 100, 40);
    plot(trainInfo.TrainingLoss, 'DisplayName', sprintf('GRU layer = %d', nGRU));
end
xlabel('Epoch'); ylabel('Training Loss'); legend show;
title('Fig. 4(b): GRU training loss');
