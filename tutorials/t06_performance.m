%% t06 - Cost of the analytic Jacobian
% (a) one analytic Jacobian vs one finite-difference Jacobian (serial and parfor
%     over columns, as the legacy EDRegisterSparse3__J_parfor_Show_iter.m did)
% (b) scaling with the mesh size (f10k vs f20k, and f80k if present) and with
%     the number of ED nodes (SampleRate)
% (c) size, number of non-zeros and fill ratio of J
% All timings are wall-clock medians (timeit) at a fixed perturbed parameter
% vector, with the correspondences recomputed inside edreg.residual as usual.
clear; close all;
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(repoRoot);
figDir = fullfile(repoRoot, 'results', 'figures');
if ~isfolder(figDir), mkdir(figDir); end
tStart = tic;
w = edreg.defaultWeights();
dataDir = fullfile(repoRoot, 'data');

%% (a) Analytic vs finite-difference Jacobian on the f10k pair
src = edreg.loadMesh(fullfile(dataDir, 'human_020_f10k.ply'));
tgt = edreg.loadMesh(fullfile(dataDir, 'human_022_f10k.ply'));
model = edreg.buildModel(src.V, src.F, Seed=1);
N = model.nNodes;
nP = 6 + 12*N;
rng(0);
P = edreg.identityParam(N) + 0.01 * randn(nP, 1);
[F0, J, info] = edreg.residual(P, model, tgt, w);

tF = timeit(@() edreg.residual(P, model, tgt, w));
tJ = timeit(@() jacobianOnly(P, model, tgt, w));
fprintf('f10k: %d vertices, %d nodes, %d parameters, %d residuals\n', model.nVerts, N, nP, numel(F0));
fprintf('  residual only          : %8.4f s\n', tF);
fprintf('  residual + analytic J  : %8.4f s\n', tJ);

% Finite differences with FIXED correspondences, forward difference with the
% legacy step 1e-5 (a central difference would double the cost).
fixedFun = @(x) edreg.residual(x, model, tgt, w, info.corr);
h = 1e-5;
tic; Jfd = fdJacobian(fixedFun, P, F0, h, false); tFDserial = toc;
fprintf('  finite-difference J, serial (%d columns): %8.2f s  = %.0fx the analytic one\n', nP, tFDserial, tFDserial / tJ);
errRel = full(max(abs(J - Jfd), [], 'all')) / full(max(abs(J), [], 'all'));
fprintf('  max |J_analytic - J_fd| / max|J| = %.2e (forward difference, h = %g)\n', errRel, h);

tFDpar = NaN;
nWorkers = 0;
try
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf('  starting a parallel pool ...\n');
        pool = parpool;
    end
    nWorkers = pool.NumWorkers;
    fdJacobian(fixedFun, P, F0, h, true);          % warm-up: data transfer, JIT on workers
    tic; fdJacobian(fixedFun, P, F0, h, true); tFDpar = toc;
    fprintf('  finite-difference J, parfor (%d workers): %8.2f s  = %.0fx the analytic one\n', nWorkers, tFDpar, tFDpar / tJ);
catch err
    fprintf('  parfor timing skipped (no parallel pool available): %s\n', err.message);
end

%% (b1) Scaling with the mesh size
% The FD cost per Jacobian is (6+12N) residual evaluations; for the larger
% meshes it is ESTIMATED from the time of 20 columns, which is exact up to the
% constant per-column cost (validated against the full measurement above).
meshes = struct('name', {"f10k", "f20k", "f80k"}, ...
    'src', {'human_020_f10k.ply', 'human_020_f20k.ply', 'human_020_f80k.ply'}, ...
    'tgt', {'human_022_f10k.ply', 'human_022_f20k.ply', 'human_022_f80k.ply'});
rows = struct('label', {}, 'nVerts', {}, 'nNodes', {}, 'nP', {}, 'nRows', {}, 'nnz', {}, ...
    'fill', {}, 'tF', {}, 'tJ', {}, 'tFDest', {});
for m = 1:numel(meshes)
    fs = fullfile(dataDir, meshes(m).src);
    ft = fullfile(dataDir, meshes(m).tgt);
    if ~isfile(fs) || ~isfile(ft)
        fprintf('(b) %s skipped: data not found. Put %s and %s in %s to include it.\n', ...
            meshes(m).name, meshes(m).src, meshes(m).tgt, dataDir);
        continue
    end
    s = edreg.loadMesh(fs); t = edreg.loadMesh(ft);
    rows(end+1) = measure(char(meshes(m).name), s, t, 0.01, w); %#ok<SAGROW>
end

%% (b2) Scaling with the number of ED nodes (f10k)
rates = [0.0025 0.005 0.01 0.02 0.04];
nodeRows = struct('label', {}, 'nVerts', {}, 'nNodes', {}, 'nP', {}, 'nRows', {}, 'nnz', {}, ...
    'fill', {}, 'tF', {}, 'tJ', {}, 'tFDest', {});
for r = rates
    nodeRows(end+1) = measure(sprintf('rate %.4g', r), src, tgt, r, w); %#ok<SAGROW>
end

%% (c) Size, non-zeros and fill of J
printTable('Mesh size (SampleRate 0.01)', rows);
printTable('Number of ED nodes (f10k)', nodeRows);
fprintf('\nJacobian blocks at the f10k default (nnz per term):\n');
for tm = string(fieldnames(info.rows))'
    rr = info.rows.(tm);
    fprintf('  %-7s %6d rows, nnz %8d\n', tm, numel(rr), nnz(J(rr, :)));
end

%% Figure: cost vs number of parameters
fig = edreg.newFigure(Position=[100 100 1100 420]);
subplot(1, 2, 1);
loglog([nodeRows.nP], [nodeRows.tJ], '-o', [nodeRows.nP], [nodeRows.tFDest], '-s', ...
    [nodeRows.nP], [nodeRows.tF], '--^', 'LineWidth', 1.4); grid on;
xlabel('number of parameters 6 + 12N'); ylabel('time per Jacobian [s]');
legend('analytic (F + J)', 'finite differences (serial, est.)', 'residual only', 'Location', 'northwest');
title('f10k, varying the number of ED nodes');
subplot(1, 2, 2);
loglog([nodeRows.nP], [nodeRows.nnz], '-o', [nodeRows.nP], [nodeRows.nP] .* [nodeRows.nRows], '--', 'LineWidth', 1.4); grid on;
xlabel('number of parameters 6 + 12N'); ylabel('entries');
legend('nnz(J)', 'size of a dense J', 'Location', 'northwest');
title('Sparsity of J');
exportgraphics(fig, fullfile(figDir, 't06_scaling.png'), 'Resolution', 150); close(fig);
fprintf('\nfigure saved to %s\n', fullfile(figDir, 't06_scaling.png'));

%% Markdown summary (exactly the numbers measured in this run)
mdFile = fullfile(repoRoot, 'results', 't06_summary.md');
fid = fopen(mdFile, 'w');
fprintf(fid, '### (a) Cost of one Jacobian, f10k (%d vertices, %d nodes, %d parameters)\n\n', model.nVerts, N, nP);
fprintf(fid, '| method | time [s] | x analytic F+J |\n|---|---|---|\n');
fprintf(fid, '| residual only | %.4f | %.2f |\n', tF, tF / tJ);
fprintf(fid, '| analytic F + J | %.4f | 1 |\n', tJ);
fprintf(fid, '| finite differences, serial (%d residual evaluations, measured) | %.3f | %.1f |\n', nP, tFDserial, tFDserial / tJ);
if isnan(tFDpar)
    fprintf(fid, '| finite differences, parfor | n/a (no parallel pool could be started) | n/a |\n');
else
    fprintf(fid, '| finite differences, parfor (%d workers, measured, pool already started) | %.3f | %.1f |\n', nWorkers, tFDpar, tFDpar / tJ);
end
fprintf(fid, '\nForward differences, step 1e-5, fixed correspondences; max abs(J - J_fd) / max abs(J) = %.2e.\n\n', errRel);
writeTable(fid, '### (b) Mesh size (SampleRate 0.01)', rows, "mesh");
fprintf(fid, '\nf80k not measured: data not present in data/.\n\n');
writeTable(fid, '### (b, c) Number of ED nodes (f10k)', nodeRows, "SampleRate");
fprintf(fid, '\nFD serial values in these tables are ESTIMATED: the time of 20 columns scaled to all columns.\n');
fclose(fid);
fprintf('summary written to %s\n', mdFile);
fprintf('t06 finished in %.1f s\n', toc(tStart));

%% ----------------------------------------------------------------------
function Jd = fdJacobian(fun, P, F0, h, useParfor)
% Forward-difference Jacobian, one residual evaluation per column.
    nP = numel(P);
    Jd = zeros(numel(F0), nP);
    if useParfor
        parfor i = 1:nP
            Pt = P; Pt(i) = Pt(i) + h;
            Jd(:, i) = (fun(Pt) - F0) / h;
        end
    else
        for i = 1:nP
            Pt = P; Pt(i) = Pt(i) + h;
            Jd(:, i) = (fun(Pt) - F0) / h;
        end
    end
end

function J = jacobianOnly(P, model, tgt, w)
    [~, J] = edreg.residual(P, model, tgt, w);
end

function r = measure(label, s, t, rate, w)
% Build a model at the given SampleRate and time F, F+J and (estimated) FD.
    model = edreg.buildModel(s.V, s.F, SampleRate=rate, Seed=1);
    N = model.nNodes;
    nP = 6 + 12*N;
    rng(0);
    P = edreg.identityParam(N) + 0.01 * randn(nP, 1);
    [F0, J, info] = edreg.residual(P, model, t, w);
    r.label = label;
    r.nVerts = model.nVerts;
    r.nNodes = N;
    r.nP = nP;
    r.nRows = size(J, 1);
    r.nnz = nnz(J);
    r.fill = nnz(J) / numel(J);
    r.tF = timeit(@() edreg.residual(P, model, t, w));
    r.tJ = timeit(@() jacobianOnly(P, model, t, w));
    fixedFun = @(x) edreg.residual(x, model, t, w, info.corr);
    cols = 1:20;
    tic;
    for i = cols
        Pt = P; Pt(i) = Pt(i) + 1e-5;
        fixedFun(Pt);
    end
    r.tFDest = toc / numel(cols) * nP;
    fprintf('  measured %s (%d vertices, %d nodes)\n', label, r.nVerts, N);
    assert(numel(F0) == r.nRows);
end

function writeTable(fid, heading, rows, firstCol)
    fprintf(fid, '%s\n\n| %s | vertices | nodes | parameters | rows | nnz(J) | fill [%%] | residual [s] | F+J [s] | FD serial [s] (estimated) | FD / (F+J) |\n|%s\n', ...
        heading, firstCol, repmat('---|', 1, 11));
    for r = rows
        fprintf(fid, '| %s | %d | %d | %d | %d | %d | %.3f | %.4f | %.4f | %.2f | %.0f |\n', r.label, r.nVerts, r.nNodes, ...
            r.nP, r.nRows, r.nnz, 100 * r.fill, r.tF, r.tJ, r.tFDest, r.tFDest / r.tJ);
    end
end

function printTable(titleText, rows)
    fprintf('\n%s\n', titleText);
    fprintf('%-10s %7s %6s %7s %9s %10s %9s %9s %9s %11s %9s\n', 'case', 'verts', 'nodes', 'params', ...
        'rows', 'nnz(J)', 'fill [%]', 'F [s]', 'F+J [s]', 'FD est [s]', 'FD/J');
    for r = rows
        fprintf('%-10s %7d %6d %7d %9d %10d %9.3f %9.4f %9.4f %11.2f %9.0f\n', r.label, r.nVerts, r.nNodes, ...
            r.nP, r.nRows, r.nnz, 100 * r.fill, r.tF, r.tJ, r.tFDest, r.tFDest / r.tJ);
    end
end
