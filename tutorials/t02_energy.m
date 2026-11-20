%% t02 - Energy terms and the finite-difference baseline
% Evaluates the residual terms at the identity deformation and runs the
% registration WITHOUT the analytic Jacobian (lsqnonlin finite differences).
% Each finite-difference Jacobian needs 6 + 12N residual evaluations, so the
% run is capped (MaxIterations = 3) here; t03 does the same with the analytic J.
clear; close all;
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(repoRoot);
figDir = fullfile(repoRoot, 'results', 'figures');
if ~isfolder(figDir), mkdir(figDir); end
tStart = tic;

src = edreg.loadMesh(fullfile(repoRoot, 'data', 'human_020_f10k.ply'));
tgt = edreg.loadMesh(fullfile(repoRoot, 'data', 'human_022_f10k.ply'));
model = edreg.buildModel(src.V, src.F, Seed=1);
N = model.nNodes;
w = edreg.defaultWeights();
fprintf('weights: rot=%.2f smooth=%.2f lap=%.3f\n', w.rot, w.smooth, w.lap);

%% Residual terms at the identity
P0 = edreg.identityParam(N);
[F0, ~, info] = edreg.residual(P0, model, tgt, w);
terms = string(fieldnames(info.rows))';
fprintf('residual length %d, resnorm = %.4f\n', numel(F0), sum(F0.^2));
for tm = terms
    f = F0(info.rows.(tm));
    fprintf('  %-7s rows %6d  sum of squares %.4e\n', tm, numel(f), sum(f.^2));
end
fprintf('(rot1, rot2 and smooth vanish at the identity; fit and lap do not)\n');

%% Registration without the analytic Jacobian (capped at 3 iterations)
maxIter = 3;
fprintf('Running lsqnonlin with finite-difference Jacobian, %d iterations...\n', maxIter);
[P, out] = edreg.register(model, tgt, UseJacobian=false, MaxIterations=maxIter);
fprintf('finite-difference run: %.1f s, %d iterations, %d residual evaluations\n', ...
    out.elapsed, out.iterations, out.funcCount);
fprintf('resnorm history: %s\n', mat2str(out.resnormHistory', 5));
fprintf('final resnorm = %.4f\n', out.resnorm);

edreg.showMeshes({struct('V', out.Vd, 'F', src.F), tgt}, Visible="off", ...
    Title=sprintf('Finite-difference registration, %d iterations', out.iterations), ...
    SavePath=fullfile(figDir, 't02_baseline_result.png'));
close all;

%% Distance of every source vertex to the target surface points
dBefore = edreg.surfaceDistance(src.V, tgt.V);
dAfter = edreg.surfaceDistance(out.Vd, tgt.V);
fprintf('distance to target   before: mean %.2f  median %.2f  p95 %.2f\n', mean(dBefore), median(dBefore), pct95(dBefore));
fprintf('distance to target   after : mean %.2f  median %.2f  p95 %.2f\n', mean(dAfter), median(dAfter), pct95(dAfter));
edreg.showError(out.Vd, src.F, dAfter, CLim=[0 pct95(dBefore)], Title='Finite-difference result: distance to target', ...
    SavePath=fullfile(figDir, 't02_error_after.png'));
close all;

baseline.elapsed = out.elapsed;
baseline.iterations = out.iterations;
baseline.resnorm = out.resnorm;
baseline.history = out.resnormHistory;
cacheDir = fullfile(repoRoot, 'results', 'cache');
if ~isfolder(cacheDir), mkdir(cacheDir); end
save(fullfile(cacheDir, 't02_baseline.mat'), 'baseline');
fprintf('t02 finished in %.1f s\n', toc(tStart));

function p = pct95(x)
% 95th percentile without the Statistics Toolbox.
    x = sort(x(:));
    p = x(ceil(0.95 * numel(x)));
end
