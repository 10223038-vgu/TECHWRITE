# Recreating the Simulations in "Soft-Output Deep LAS Detection for Coded MIMO Systems"

## 0. Common System Setup (used by almost every figure)

| Parameter | Value |
|---|---|
| Antennas | Nt = Nr = 4 (varied in Fig. 11b: 16, 32, 64, 128, 256) |
| Modulation | 4-QAM and 16-QAM |
| Channel | Rayleigh flat fading per subcarrier, i.i.d. CN(0,1) entries |
| OFDM | IFFT/FFT length = 1024 (varied in Fig. 11a: 512, 1024, 2048) |
| Channel coding | Turbo code, rate 1/2, 8 decoder iterations |
| Interleaver | Random |
| Block size | 8 symbols per channel use |
| Simulator | MATLAB R2023a |

```matlab
% ---- Core system parameters (reuse in every script) ----
Nt = 4; Nr = 4;
M_list = [4 16];              % QAM orders
FFT_len = 1024;
SNRdB_range = 0:2:16;
nBlocksPerSNR = 5000;         % increase for smoother BER curves
turboRate = 1/2;
turboIter = 8;
```

Use `comm.RayleighChannel`/manual generation `H = (randn(Nr,Nt)+1i*randn(Nr,Nt))/sqrt(2)` per subcarrier, `qammod`/`qamdemod` for M-QAM, and `comm.TurboEncoder`/`comm.TurboDecoder` (or `nrTurbo*` if using 5G Toolbox — the paper uses classic turbo codes) for FEC.

---

## 1. Model-Based Two-Step Soft-Output LAS (Algorithms 1 & 2)

**What it does:** Generates the "ground truth" training data (actual LLRs) used later to train Deep LAS, and is itself compared in BER plots.

**Steps to implement:**

1. **Initial estimate** — ZF or MMSE equalizer (Eqs. 5–6):
```matlab
function xhat0 = mmse_init(H, y, EsN0)
    Nt = size(H,2);
    W = (H'*H + (1/EsN0)*eye(Nt)) \ H';
    xhat0 = qam_hard_decision(W*y);
end
```

2. **Step 1 — 1-LAS hypothesis search** (Eqs. 7–13, Algorithm 1): iteratively flip one symbol at a time along `sign(z_n)`, accept if the cost difference ΔΛ (Eq. 10) is negative, repeat until no improvement. Track `F0` = final minimized cost.

3. **Step 2 — modified-LAS counter-hypothesis** (Eqs. 14–18, Algorithm 2): for each bit, flip that bit in the constellation, re-run a constrained LAS search restricted to counter-symbols only, track `F1`.

4. **LLR computation** (Eq. 4b):
```matlab
lij = (1/sigma2) * (F1 - F0);   % sign depends on detected bit value, see Algorithm 2 pseudocode
```

Wrap steps 1–4 in a function `[xhat, LLR] = softOutputLAS(y, H, sigma2, M)` — this is your data generator for Deep LAS training AND your model-based baseline for Fig. 9/10.

---

## 2. Deep LAS Training Data Generation (Section IV-intro, Eq. 19)

For each SNR point and modulation order, generate `{y_hat, xhat_MMSE, LLR_target}` triples:

```matlab
for snr = SNRdB_range
  for b = 1:nBlocksPerSNR
    x = qammod(randi([0 M-1], Nt,1), M, 'UnitAveragePower', true);
    H = (randn(Nr,Nt)+1i*randn(Nr,Nt))/sqrt(2);
    n = sqrt(sigma2/2)*(randn(Nr,1)+1i*randn(Nr,1));
    y = H*x + n;
    [xhat_mmse, LLR_true] = softOutputLAS(y,H,sigma2,M);
    yhat = [real(y); imag(y)];
    xm   = [real(xhat_mmse); imag(xhat_mmse)];
    % store [yhat; xm] -> LLR_true  (Eq. 19 formatting)
  end
end
```
Save as `.mat` files per SNR/modulation — this is the dataset behind **Figs. 5–8**.

---

## 3. Deep LAS Network (MLP + GRU), Section IV-A, Fig. 2

**MLP block** — plain feedforward regressor, `H` layers, `K=10` hidden neurons/layer, ReLU, linear output, trained with Levenberg–Marquardt (`trainlm`, Neural Net Toolbox `fitnet`) — *not* `trainNetwork`, since LM is a classic NN-toolbox algorithm:

```matlab
net = fitnet(10*ones(1,H-2), 'trainlm');   % H-2 hidden layers, 10 neurons each
net.trainParam.epochs = 300;
net.trainParam.max_fail = inf;
net = train(net, Xtrain', Ytrain');        % Xtrain = xhat_mmse (normalized), Ytrain = LLR_true
LLR_mlp = net(Xtest')';
```
Normalize input per Algorithm 3: `xhat_m = xhat_m / max(xhat_m)`.

**GRU block** — sequence regression using Deep Learning Toolbox:

```matlab
inputSize = 2*Nt+1;     % concatenated [y'; LLR_MLP]
numHiddenUnits = 100;   % V = 100 GRU cells
layers = [ ...
    sequenceInputLayer(inputSize)
    gruLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.01)
    reluLayer
    gruLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.01)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',40, 'MiniBatchSize',40, ...
    'Shuffle','every-epoch', 'ValidationData',{XVal,YVal});

gruNet = trainNetwork(XTrainSeq, YTrainSeq, layers, options);
```
Concatenate MLP output with `y'` per Eq. 23 to form the GRU input `U_G`.

Train **each block independently** (80/20 train/validation split), sweeping FC-layer count (1–3) and GRU-layer count (1–3) to reproduce **Fig. 4**.

---

## Mapping Each Figure to a Script

| Figure | What to plot | How to generate |
|---|---|---|
| **Fig. 4** | Training loss vs epoch, for FC layers = 1/2/3 and GRU layers = 1/2/3 | Log `net.trainParam` history (`tr.perf`) for MLP; use `'Plots','training-progress'` or capture `info.TrainingLoss` from `trainNetwork` for GRU. Overlay curves. |
| **Fig. 5–6** | Scatter: actual vs MLP-approximated LLR vs Re(Rx symbol) | Run trained MLP on test set at fixed SNR (4 dB for 4-QAM, 10 dB for 16-QAM), scatter `LLR_true` vs `LLR_mlp` per bit index |
| **Fig. 7–8** | Scatter/line: actual vs GRU(final Deep LAS)-approximated LLR | Same as above but using full Deep LAS output `LLR_hat` at SNR = {0,4} dB (4-QAM) and {7,10} dB (16-QAM) |
| **Fig. 9** | BER vs SNR: Conv. LAS (hard), MLP-only, Prop. two-step Soft-LAS, Prop. Deep LAS | Monte-Carlo BER loop (below) comparing hard-decision LAS bits vs LLR-based soft bits (sign of LLR) fed through turbo decoder |
| **Fig. 10** | BER vs SNR: adds DetNet and optimal SD [9] as baselines, for ZF and MMSE init | Implement DetNet (iterative projected-gradient unfolding, see [35]/[39]) and sphere decoding (`comm.SphereDecoder` won't give soft output — implement soft-SD manually per [9]) as extra baselines in the same BER loop |
| **Fig. 11a** | BER vs SNR for FFT length = 512/1024/2048 | Rerun BER loop with `FFT_len` swept, only affects sequence length fed to GRU (longer FFT ⇒ more OFDM symbols per training sequence) |
| **Fig. 11b** | BER vs SNR for Nt=Nr = 16/32/64/128/256 | Rerun BER loop scaling `Nt,Nr`; note LAS Algorithm 1/2 complexity is O(Nt²), so runtime grows |
| **Fig. 12** | BER vs SNR: MLP-only, LSTM, Bi-LSTM, GRU, Prop. Deep LAS | Swap `gruLayer` for `lstmLayer`/`bilstmLayer` (800 hidden units, dropout 0.2, ReLU, 500 epochs) and compare against the MLP+GRU hybrid |
| **Table I** | Complexity comparison | Not a simulation — analytical; verify empirically by timing each detector (`tic/toc`) per symbol, per Nt, to sanity-check the big-O trend |

---

## Core BER Monte-Carlo Loop (used for Figs. 9–12)

```matlab
for M = M_list
  for snrdB = SNRdB_range
    sigma2 = 10^(-snrdB/10);
    nErr = 0; nBits = 0;
    for blk = 1:nBlocksPerSNR
        bits = randi([0 1], Nt*log2(M), 1);
        % turbo encode -> interleave -> modulate -> IFFT/CP -> channel -> FFT
        x = qammod(bi2de(reshape(bits,[],log2(M))','left-msb'), M, 'UnitAveragePower',true);
        H = (randn(Nr,Nt)+1i*randn(Nr,Nt))/sqrt(2);
        n = sqrt(sigma2/2)*(randn(Nr,1)+1i*randn(Nr,1));
        y = H*x + n;

        % --- choose detector ---
        [~, LLR] = softOutputLAS(y,H,sigma2,M);      % Prop. Soft-LAS
        % or: LLR = deepLAS_predict(y,H,mlpNet,gruNet);  % Prop. Deep LAS
        % or: bits_hat = conv_LAS_hard(y,H,M);            % Conv. LAS (hard)

        % deinterleave -> turbo decode using LLR
        decodedBits = turboDecode(LLR, turboIter);
        nErr = nErr + sum(decodedBits ~= bits);
        nBits = nBits + length(bits);
    end
    BER(snrdB) = nErr/nBits;
  end
end
semilogy(SNRdB_range, BER); grid on;
```

---

## Practical Notes

- **DetNet** and **soft-output SD [9]** are not built-in MATLAB functions — you'll need to implement them from the cited references; DetNet is a straightforward unfolded gradient-descent network (few `fullyConnectedLayer`s per iteration block), and soft SD is a depth-first tree search returning both ML and counter-hypothesis costs.
- Use `parfor` over Monte-Carlo blocks — LAS + turbo decoding is slow in pure MATLAB; consider MEX/vectorizing the inner LAS loop for large Nt (128, 256).
- Reproduce SNR gap annotations (1.9 dB, 2.55 dB, etc.) by reading off SNR at BER = 10⁻⁴/10⁻⁵ from `interp1` on the BER curves.
