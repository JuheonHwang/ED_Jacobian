%% t03 - The analytic Jacobian
% Computes the sparse analytic Jacobian, shows its block structure, and runs
% the registration with it.  Compare with the finite-difference baseline of t02.
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

%% Jacobian at a slightly perturbed parameter vector
rng(0);
P = edreg.identityParam(N) + 0.01 * randn(6 + 12*N, 1);
tic; [F, J, info] = edreg.residual(P, model, tgt, w); tJac = toc;
fprintf('J is %d x %d, nnz = %d (%.3f%% dense)\n', size(J, 1), size(J, 2), nnz(J), 100*nnz(J)/numel(J));
fprintf('residual + analytic Jacobian: %.3f s\n', tJac);
tic; F = edreg.residual(P, model, tgt, w, info.corr); tF = toc; %#ok<NASGU>
fprintf('residual only               : %.3f s\n', tF);

%% Cost of one finite-difference Jacobian (all columns), for comparison
fun = @(x) edreg.residual(x, model, tgt, w, info.corr);
tic; edreg.numericJacobian(fun, P); tFD = toc;
fprintf('numeric Jacobian (%d columns): %.1f s  -> %.0fx slower than the analytic one\n', ...
    numel(P), tFD, tFD / tJac);

%% Sparsity pattern with term (rows) and parameter (columns) blocks
fig = edreg.newFigure(Position=[100 100 1100 800]);
ax = axes(fig);
spy(J, 1); hold(ax, 'on');
axis(ax, 'normal');
ax.Position = [0.09 0.08 0.66 0.80];
ax.YAxis.Exponent = 0;
rowNames = string(fieldnames(info.rows))';
for tm = rowNames
    yline(ax, info.rows.(tm)(end) + 0.5, 'r');
end
thin = [info.rows.rot1(1), info.rows.smooth(end)];       % rot1, rot2, smooth are short
labels = {"fit (" + numel(info.rows.fit) + " rows)", mean([info.rows.fit(1) info.rows.fit(end)]);
          "rot1 + rot2 + smooth (" + (thin(2) - thin(1) + 1) + " rows)", mean(thin);
          "lap (" + numel(info.rows.lap) + " rows)", mean([info.rows.lap(1) info.rows.lap(end)])};
for i = 1:size(labels, 1)
    text(ax, size(J, 2) * 1.02, labels{i, 2}, labels{i, 1}, 'Color', 'r', 'FontWeight', 'bold', 'Clipping', 'off');
end
for b = ["R,T", "A", "t"]
    switch b
        case "R,T", c = info.cols.R(1):info.cols.T(end);
        case "A",   c = info.cols.A;
        case "t",   c = info.cols.t;
    end
    xline(ax, c(end) + 0.5, 'b');
    text(ax, mean(c([1 end])), -0.03*size(J, 1), b, 'Color', 'b', 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'Clipping', 'off');
end
title(ax, sprintf('Sparsity of J (%d x %d). Rows: residual terms; columns: R,T | A | t', size(J, 1), size(J, 2)), 'Color', 'k');
ax.Title.Units = 'normalized'; ax.Title.Position = [0.5 1.06];
xlabel(ax, 'parameter index'); ylabel(ax, 'residual row');
exportgraphics(fig, fullfile(figDir, 't03_jacobian_sparsity.png'), 'Resolution', 150);
close(fig);

%% Zoom: rot1 / rot2 / smooth rows x (A, t) columns of the first 5 nodes
% Each rot1 row touches only the 9 A entries of its own node (block diagonal),
% rot2 likewise; smooth rows touch A_k plus the t of node k and its neighbours.
nz = 5;
zRows = [info.rows.rot1(1:9*nz), info.rows.rot2(1:nz), info.rows.smooth(1:3*nz)];
zCols = [info.cols.A(1:9*nz), info.cols.t(1:3*nz)];
Jz = J(zRows, zCols);
zEnds = cumsum([9*nz, nz, 3*nz]);                        % row ends of rot1 / rot2 / smooth
zStarts = [1, zEnds(1:2) + 1];
fig = edreg.newFigure(Position=[100 100 1100 650]);
ax = axes(fig);
spy(Jz, 10); hold(ax, 'on');
axis(ax, 'normal');
ax.Position = [0.08 0.12 0.70 0.74];
zNames = ["rot1" "rot2" "smooth"];
for i = 1:3
    yline(ax, zEnds(i) + 0.5, 'r');
    text(ax, size(Jz, 2) + 1, mean([zStarts(i) zEnds(i)]), zNames(i), 'Color', 'r', 'FontWeight', 'bold', 'Clipping', 'off');
end
for k = 1:nz - 1                              % thin grey lines separate the nodes
    xline(ax, 9*k + 0.5, ':', 'Color', [0.5 0.5 0.5]);
end
xline(ax, 9*nz + 0.5, 'b');
xline(ax, 9*nz + 3*(1:nz-1) + 0.5, ':', 'Color', [0.5 0.5 0.5]);
yTop = -0.04 * size(Jz, 1);
text(ax, 9*nz/2, yTop, sprintf('A_1 ... A_%d (9 columns per node)', nz), 'Color', 'b', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Clipping', 'off');
text(ax, 9*nz + 3*nz/2, yTop, sprintf('t_1 ... t_%d', nz), 'Color', 'b', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Clipping', 'off');
title(ax, sprintf('Zoom: rows of rot1, rot2, smooth for the first %d nodes; columns A_k and t_k of those nodes', nz), 'Color', 'k');
ax.Title.Units = 'normalized'; ax.Title.Position = [0.5 1.08];
xlabel(ax, 'column within the shown (A | t) block'); ylabel(ax, 'row within the shown rot1 / rot2 / smooth rows');
exportgraphics(fig, fullfile(figDir, 't03_jacobian_sparsity_zoom.png'), 'Resolution', 150);
close(fig);

%% Registration with the analytic Jacobian: same iteration cap as t02
[~, out3] = edreg.register(model, tgt, UseJacobian=true, MaxIterations=3);
fprintf('analytic J, MaxIterations=3 : %.1f s, %d iterations (%.2f s/iter), resnorm %.4f\n', ...
    out3.elapsed, out3.iterations, out3.elapsed / out3.iterations, out3.resnorm);
baselineFile = fullfile(repoRoot, 'results', 'cache', 't02_baseline.mat');
if isfile(baselineFile)
    b = load(baselineFile).baseline;
    fprintf(['t02 finite differences, %d iterations: %.1f s (%.2f s/iter), resnorm %.4f\n' ...
             '  -> %.1fx faster per iteration with the analytic Jacobian\n'], ...
        b.iterations, b.elapsed, b.elapsed / b.iterations, b.resnorm, ...
        (b.elapsed / b.iterations) / (out3.elapsed / out3.iterations));
    fprintf(['  (resnorms differ because the finite-difference gradient also differentiates through the\n' ...
             '   changing nearest-neighbour correspondences, which the analytic Jacobian holds fixed)\n']);
else
    fprintf('(run t02 first to print the speed-up against the finite-difference baseline)\n');
end

%% Full registration (30 iterations) with the analytic Jacobian
[Pfull, out] = edreg.register(model, tgt, UseJacobian=true, MaxIterations=30);
fprintf('analytic J, 30 iterations: %.1f s, %d iterations, final resnorm %.4f\n', ...
    out.elapsed, out.iterations, out.resnorm);
fprintf('resnorm: start %.4f -> end %.4f\n', out.resnormHistory(1), out.resnormHistory(end));
[~, ~, infoEnd] = edreg.residual(Pfull, model, tgt, w);
Fend = edreg.residual(Pfull, model, tgt, w, infoEnd.corr);
for tm = rowNames
    fprintf('  final %-7s sum of squares %.4e\n', tm, sum(Fend(infoEnd.rows.(tm)).^2));
end

fig = edreg.newFigure(Position=[100 100 700 500]);
semilogy(0:numel(out.resnormHistory)-1, out.resnormHistory, '-o');
xlabel('iteration'); ylabel('resnorm'); grid on; title('Registration with analytic Jacobian');
exportgraphics(fig, fullfile(figDir, 't03_resnorm_history.png'), 'Resolution', 150); close(fig);

edreg.showMeshes({struct('V', out.Vd, 'F', src.F), tgt}, Visible="off", ...
    Title=sprintf('Analytic-Jacobian registration, %d iterations', out.iterations), ...
    SavePath=fullfile(figDir, 't03_result.png'));
close all;

%% Fit quality: distance of every source vertex to the nearest target point
dBefore = edreg.surfaceDistance(src.V, tgt.V);
dAfter = edreg.surfaceDistance(out.Vd, tgt.V);
fprintf('distance to target   before: mean %.2f  median %.2f  p95 %.2f\n', mean(dBefore), median(dBefore), pct95(dBefore));
fprintf('distance to target   after : mean %.2f  median %.2f  p95 %.2f\n', mean(dAfter), median(dAfter), pct95(dAfter));
lims = [0 pct95(dBefore)];
allV = [src.V; out.Vd];                                  % identical axis limits -> identical scale
axLims = [min(allV, [], 1)' max(allV, [], 1)'];
fig = edreg.newFigure(Position=[100 100 1200 800]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact');
[~, ax1] = edreg.showError(src.V, src.F, dBefore, CLim=lims, Limits=axLims, Colorbar=false, ...
    Title='Before (identity)', Axes=nexttile(tl));
[~, ax2] = edreg.showError(out.Vd, src.F, dAfter, CLim=lims, Limits=axLims, Colorbar=false, ...
    Title='After (analytic Jacobian, 30 iterations)', Axes=nexttile(tl));
cb = colorbar(ax2);                                      % one colourbar for both panels
cb.Layout.Tile = 'south';
cb.Label.String = 'distance to nearest target point';
exportgraphics(fig, fullfile(figDir, 't03_error_before_after.png'), 'Resolution', 150);
close all;
fprintf('t03 finished in %.1f s\n', toc(tStart));

function p = pct95(x)
% 95th percentile without the Statistics Toolbox.
    x = sort(x(:));
    p = x(ceil(0.95 * numel(x)));
end
