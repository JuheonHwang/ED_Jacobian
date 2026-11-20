%% t05 - Comparing the optimisers
% Runs every solver of edreg.solvers, plus lsqnonlin (edreg.register) as the
% reference, on the f10k pair from the SAME start (identity), with the SAME
% model, weights and analytic Jacobian, then compares
%   1. resnorm vs iteration
%   2. resnorm vs wall-clock time
%   3. a summary table (also saved to results/t05_summary.md) with the distance
%      of the final deformed source to the target, and an error map of the best solver.
% resnorm = sum(F.^2) with the nearest-neighbour correspondences recomputed at
% every evaluation, so all methods minimise the same ICP-style energy.
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
P0 = edreg.identityParam(N);
fun = @(P) edreg.residual(P, model, tgt, w);
fun(P0);                                    % warm-up so that no solver pays for JIT/caching
fprintf('%d vertices, %d ED nodes, %d parameters, resnorm at the start %.3f\n', ...
    model.nVerts, N, numel(P0), sum(fun(P0).^2));

%% Run the solvers
% Newton-type methods get 30 iterations (as the lsqnonlin reference); gradient
% descent is far cheaper per iteration and gets the legacy 100.  Two step sizes
% are shown for gradient descent: the legacy 1e-3, and a smaller one.
solvers = { ...
    "lsqnonlin (reference)",       []; ...
    "gradientDescent (1e-3, legacy)", @(f, P) edreg.solvers.gradientDescent(f, P, MaxIterations=100); ...
    "gradientDescent (2e-4)",      @(f, P) edreg.solvers.gradientDescent(f, P, StepSize=2e-4, MaxIterations=100); ...
    "gaussNewton (normal eq.)",    @(f, P) edreg.solvers.gaussNewton(f, P, MaxIterations=30); ...
    "gaussNewton (lsq, legacy _re)", @(f, P) edreg.solvers.gaussNewton(f, P, LinearSolver="lsq", MaxIterations=30); ...
    "LM hjh2 (direct)",            @(f, P) edreg.solvers.levenbergMarquardt(f, P, MaxIterations=30); ...
    "LM hjh1 (legacy 3 trials)",   @(f, P) edreg.solvers.levenbergMarquardt(f, P, Variant="hjh1", MaxIterations=30); ...
    "LM-CG",                       @(f, P) edreg.solvers.levenbergMarquardtCG(f, P, Preconditioner="none", MaxIterations=30); ...
    "LM-PCG (Jacobi)",             @(f, P) edreg.solvers.levenbergMarquardtCG(f, P, Preconditioner="jacobi", MaxIterations=30)};
nS = size(solvers, 1);
names = strings(nS, 1);
results = struct('P', cell(nS, 1), 'resnorm', [], 'time', [], 'total', [], 'iters', [], 'exit', "", 'nEval', []);

for i = 1:nS
    names(i) = solvers{i, 1};
    fprintf('running %s ...\n', names(i));
    if isempty(solvers{i, 2})
        [P, out] = edreg.register(model, tgt, Weights=w, MaxIterations=30);
        results(i).resnorm = out.resnormHistory(:);
        results(i).time = out.timeHistory(:);
        results(i).total = out.elapsed;
        results(i).iters = out.iterations;
        results(i).exit = "lsqnonlin flag " + out.exitflag;
        results(i).nEval = out.funcCount;
    else
        [P, hst] = solvers{i, 2}(fun, P0);
        results(i).resnorm = hst.resnorm;
        results(i).time = hst.time;
        results(i).total = hst.time(end);
        results(i).iters = hst.iterations;
        results(i).exit = hst.exitReason;
        results(i).nEval = hst.funcCount;
        if isfield(hst, 'cgIterations') && names(i) == "LM-PCG (Jacobi)"
            cgHist = hst;
        end
        if isfield(hst, 'lambda') && names(i) == "LM hjh2 (direct)"
            lmHist = hst;
        end
    end
    results(i).P = P;
end

%% Summary table
% Time is the wall-clock time of the whole solve.  "best" is the smallest
% resnorm visited, which differs from "final" for the non-monotone methods.
% dist = distance from every vertex of the final deformed source to the nearest
% target point (edreg.surfaceDistance), in the units of the scans.
for i = 1:nS
    d = edreg.surfaceDistance(edreg.deform(results(i).P, model), tgt.V);
    results(i).dist = [mean(d), median(d), prctile95(d)];
end
header = ["solver" "iters" "final" "best" "time [s]" "#fun" "dist mean" "dist median" "dist p95" "exit"];
cells = strings(nS, numel(header));
for i = 1:nS
    r = results(i);
    cells(i, :) = [names(i), string(r.iters), sprintf("%.4f", r.resnorm(end)), sprintf("%.4f", min(r.resnorm)), ...
        sprintf("%.2f", r.total), string(r.nEval), compose("%.3f", r.dist), r.exit];
end
fprintf('\n%-32s %6s %12s %12s %9s %6s %10s %11s %9s  %s\n', header);
for i = 1:nS
    fprintf('%-32s %6s %12s %12s %9s %6s %10s %11s %9s  %s\n', cells(i, :));
end
mdFile = fullfile(repoRoot, 'results', 't05_summary.md');
fid = fopen(mdFile, 'w');
fprintf(fid, '| %s |\n|%s\n', strjoin(header, ' | '), repmat('---|', 1, numel(header)));
for i = 1:nS
    fprintf(fid, '| %s |\n', strjoin(cells(i, :), ' | '));
end
fclose(fid);
fprintf('summary table written to %s\n', mdFile);

%% Figure 1: resnorm vs iteration and vs time
colors = [lines(7); 0.35 0.35 0.35; 0 0 0];   % nine distinguishable colours
colors = colors(1:nS, :);
fig = edreg.newFigure(Position=[100 100 1450 520]);
Emin = min(arrayfun(@(r) min(r.resnorm), results));
E0 = results(1).resnorm(1);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact');
ax1 = nexttile(tl); hold(ax1, 'on'); grid(ax1, 'on');
ax2 = nexttile(tl); hold(ax2, 'on'); grid(ax2, 'on');
for i = 1:nS
    r = results(i).resnorm;
    semilogy(ax1, 0:numel(r)-1, r, '-', 'Color', colors(i, :), 'LineWidth', 1.5);
    plot(ax2, results(i).time, r, '-', 'Color', colors(i, :), 'LineWidth', 1.5);
end
set(ax1, 'YScale', 'log'); set(ax2, 'XScale', 'log', 'YScale', 'log');
yl = [0.97 * Emin, 1.2 * E0];
ylim(ax1, yl); ylim(ax2, yl);
xlabel(ax1, 'iteration'); ylabel(ax1, 'resnorm = sum(F.^2)'); title(ax1, 'Resnorm vs iteration');
xlabel(ax2, 'wall-clock time [s]'); ylabel(ax2, 'resnorm = sum(F.^2)'); title(ax2, 'Resnorm vs time');
lg = legend(ax2, names, 'Interpreter', 'none', 'FontSize', 8);
lg.Layout.Tile = 'east';
title(lg, 'curves leaving the axes diverge');
exportgraphics(fig, fullfile(figDir, 't05_convergence.png'), 'Resolution', 150); close(fig);

%% Figure 2: lambda schedule and inner CG iterations of the LM variants
fig = edreg.newFigure(Position=[100 100 1000 400]);
subplot(1, 2, 1);
semilogy(1:numel(lmHist.lambda) - 1, lmHist.lambda(2:end), '-o'); grid on;
xlabel('iteration'); ylabel('\lambda used for the accepted step'); title('LM hjh2: lambda schedule');
subplot(1, 2, 2);
plot(1:numel(cgHist.cgIterations) - 1, cgHist.cgIterations(2:end), '-o'); grid on;
xlabel('LM iteration'); ylabel('inner CG iterations'); title(sprintf('LM-PCG: CG iterations (cap = %d)', numel(P0)));
exportgraphics(fig, fullfile(figDir, 't05_lambda_cg.png'), 'Resolution', 150); close(fig);

%% Figure 3: error map of the lowest-energy method
[~, best] = min(arrayfun(@(r) r.resnorm(end), results));
Vd = edreg.deform(results(best).P, model);
edreg.showError(Vd, src.F, edreg.surfaceDistance(Vd, tgt.V), ...
    Title=sprintf('%s: resnorm %.2f, distance to target', names(best), results(best).resnorm(end)), ...
    Label="distance to nearest target point", SavePath=fullfile(figDir, 't05_error_best.png'));
close all;
fprintf('\nfigures saved to %s\n', figDir);
fprintf('t05 finished in %.1f s\n', toc(tStart));

function p = prctile95(x)
% 95th percentile without the Statistics Toolbox.
    x = sort(x(:));
    p = x(max(1, ceil(0.95 * numel(x))));
end
