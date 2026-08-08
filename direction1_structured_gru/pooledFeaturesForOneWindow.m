function pooled = pooledFeaturesForOneWindow(UG_full, targetIdx, contextIdx, distToTarget, nearFrac, decayScale)
% POOLEDFEATURESFORONEWINDOW Shared feature-construction logic for
% distance-aware pooling, used identically by
% buildDistanceWeightedPooledFeatures.m (training) and
% simulateBERDistancePooling.m (test-time) -- kept in one place so the
% two can never silently drift apart (the exact bug class that caused
% the original GRU sequence-length mismatch).
%
% See buildDistanceWeightedPooledFeatures.m's header for the full
% rationale. Returns an 8*(2Nt+1) x 1 column vector:
%   [target ; nearMean ; nearMax ; nearStd ; farMean ; farMax ; farStd ;
%    distWeightedMean]

targetFeat = UG_full(:, targetIdx);                 % (D x 1)
contextFeat = UG_full(:, contextIdx);                % (D x nContext)

nContext = numel(contextIdx);
nNear = max(1, round(nearFrac * nContext));
[~, distSortIdx] = sort(distToTarget, 'ascend');
nearCols = distSortIdx(1:nNear);
farCols  = distSortIdx(nNear+1:end);

nearFeat = contextFeat(:, nearCols);
if isempty(farCols)
    farFeat = nearFeat;   % degenerate (very small window) fallback: reuse near stats
else
    farFeat = contextFeat(:, farCols);
end

nearMean = mean(nearFeat, 2);
nearMax  = max(nearFeat, [], 2);
nearStd  = std(nearFeat, 0, 2);

farMean = mean(farFeat, 2);
farMax  = max(farFeat, [], 2);
farStd  = std(farFeat, 0, 2);

w = exp(-distToTarget / decayScale);
w = w / sum(w);
distWeightedMean = contextFeat * w(:);   % (D x 1)

pooled = [targetFeat; nearMean; nearMax; nearStd; farMean; farMax; farStd; distWeightedMean];
end
