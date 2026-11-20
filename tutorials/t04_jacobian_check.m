%% t04 - Checking the analytic Jacobian against finite differences
% 1. checkJacobian at the identity and at a random P
% 2. effect of the finite-difference step size
% 3. source -> noisy source sanity test (as in the legacy src2src_noise_* scripts)
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

%% 1a. At the identity (many blocks vanish here: A = I makes rot terms trivial)
fprintf('--- identity ---\n');
edreg.checkJacobian(edreg.identityParam(N), model, tgt, w);

%% 1b. At a random perturbed P (all blocks are exercised)
fprintf('--- random P ---\n');
rng(2);
Prand = edreg.identityParam(N) + 0.05 * randn(6 + 12*N, 1);
Prand(1:3) = 0.2 * randn(3, 1);
edreg.checkJacobian(Prand, model, tgt, w);

%% 2. Step size: the legacy tests compared 1e-6 against 0.1
% Central differences have truncation error ~ h^2 f''' and round-off error
% ~ eps |F| / h.  A subset of columns keeps this fast: R, T, and all
% parameters of the first 5 nodes plus 100 random columns.
cols = unique([1:6, 6 + (1:45), 6 + 9*N + (1:15), 6 + randperm(12*N, 100)]);
steps = [1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1];
worst = zeros(size(steps));
fprintf('--- step size study (%d columns, random P) ---\n', numel(cols));
fprintf('%-10s %-14s\n', 'step', 'worst MaxRelErr');
for i = 1:numel(steps)
    rep = edreg.checkJacobian(Prand, model, tgt, w, Step=steps(i), Columns=cols, Display=false);
    worst(i) = max(rep.MaxRelErr);
    fprintf('%-10.0e %-14.3e\n', steps(i), worst(i));
end
fig = edreg.newFigure(Position=[100 100 700 500]);
loglog(steps, worst, '-o'); grid on;
xlabel('finite-difference step h'); ylabel('worst relative error');
title('Analytic vs central-difference Jacobian');
exportgraphics(fig, fullfile(figDir, 't04_step_size.png'), 'Resolution', 150); close(fig);
fprintf('Step 0.1 detail:\n');
edreg.checkJacobian(Prand, model, tgt, w, Step=0.1, Columns=cols);

%% 3. Source -> noisy source
% The target is the source plus Gaussian noise (sigma = 3), with identical
% connectivity, so the true correspondence is the identity map.
rng(1);
noisy.V = src.V + 3 * randn(size(src.V));
noisy.F = src.F;
fprintf('--- src -> noisy src ---\n');
fprintf('initial RMS vertex distance to noisy target: %.3f\n', sqrt(mean(sum((src.V - noisy.V).^2, 2))));
edreg.checkJacobian(Prand, model, noisy, w, Columns=cols);

[Pn, out] = edreg.register(model, noisy, UseJacobian=true, MaxIterations=30);
rmsAfter = sqrt(mean(sum((out.Vd - noisy.V).^2, 2)));
rmsToClean = sqrt(mean(sum((out.Vd - src.V).^2, 2)));
fprintf('registered: %.1f s, %d iterations, resnorm %.4f\n', out.elapsed, out.iterations, out.resnorm);
fprintf('RMS distance to noisy target: %.3f, RMS displacement from the clean source: %.3f\n', rmsAfter, rmsToClean);

edreg.checkJacobian(Pn, model, noisy, w, Columns=cols);

edreg.showMeshes({struct('V', out.Vd, 'F', src.F), noisy}, Visible="off", ...
    Title="Source registered to noisy source", SavePath=fullfile(figDir, 't04_src2noisy.png'));
close all;
dNoisy = edreg.surfaceDistance(out.Vd, noisy.V);
edreg.showError(out.Vd, src.F, dNoisy, Title='Registered source: distance to noisy target', ...
    SavePath=fullfile(figDir, 't04_error_after.png'));
close all;
fprintf('t04 finished in %.1f s\n', toc(tStart));
