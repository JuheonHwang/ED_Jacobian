%% t01 - The embedded deformation (ED) model
% Builds the ED model of a source scan, visualises the ED nodes and applies
% hand-made deformations to show what each block of the parameter vector
%   P = [r(3); T(3); A(:) (9N); t(:) (3N)]
% does.  Run from anywhere:  matlab -batch "run('tutorials/t01_ed_model.m')"
clear; close all;
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(repoRoot);
figDir = fullfile(repoRoot, 'results', 'figures');
if ~isfolder(figDir), mkdir(figDir); end
tStart = tic;

%% Load the source mesh and build the model
src = edreg.loadMesh(fullfile(repoRoot, 'data', 'human_020_f10k.ply'));
model = edreg.buildModel(src.V, src.F, SampleRate=0.01, NumVertexNeighbors=4, ...
    NumNodeNeighbors=6, Seed=1);
fprintf('vertices            : %d\n', model.nVerts);
fprintf('faces               : %d\n', size(src.F, 1));
fprintf('ED nodes            : %d\n', model.nNodes);
fprintf('parameters          : %d  (6 global + 9*%d affine + 3*%d translation)\n', ...
    6 + 12*model.nNodes, model.nNodes, model.nNodes);
fprintf('invalid vertices    : %d (no reachable ED node; they stay in place)\n', nnz(model.invalidVertex));
fprintf('weights row sum     : min %.6f max %.6f (valid vertices)\n', ...
    min(sum(model.wNeighbor(~model.invalidVertex, :), 2)), max(sum(model.wNeighbor(~model.invalidVertex, :), 2)));

%% Visualise the ED nodes
[fig, ax] = edreg.showMeshes({src}, Colors=[0.8 0.8 0.8 0.6], Visible="off", ...
    Title="Source mesh and ED nodes");
scatter3(ax, model.nodeV(:, 1), model.nodeV(:, 2), model.nodeV(:, 3), 40, 'r', 'filled');
exportgraphics(fig, fullfile(figDir, 't01_nodes.png'), 'Resolution', 150);
close(fig);

%% The identity parameters leave the mesh untouched
N = model.nNodes;
P0 = edreg.identityParam(N);
Vd0 = edreg.deform(P0, model);
fprintf('identity deformation: max |Vd - V| = %.2e\n', max(abs(Vd0 - src.V), [], 'all'));

%% Global rotation r (about the centroid G0) and translation T
% Pick the longest bounding-box axis as the body's "up" axis.
[~, up] = max(max(src.V) - min(src.V));
axisUp = double((1:3)' == up);
P = P0;
P(1:3) = deg2rad(10) * axisUp;               % 10 degree rotation about the long axis
P(4:6) = [0; 0; 0];
Vd = edreg.deform(P, model);
fprintf('global rotation 10 deg : max displacement %.1f\n', max(vecnorm(Vd - src.V, 2, 2)));
edreg.showMeshes({src, struct('V', Vd, 'F', src.F)}, Visible="off", ...
    Title="Global rotation r (10 deg)", SavePath=fullfile(figDir, 't01_global_rotation.png'));

%% Per-node translation t: lift one node, neighbouring vertices follow
% The node nearest to the top of the body is moved by 30 along the up axis;
% each vertex follows with weight w_ik of its (up to K) neighbouring nodes.
[~, kTop] = max(model.nodeV(:, up));
P = P0;
[~, ~, A, t] = edreg.decodeParam(P, N);
t(:, kTop) = 30 * axisUp;
P = edreg.encodeParam(eye(3), zeros(3, 1), A, t);
Vd = edreg.deform(P, model);
moved = vecnorm(Vd - src.V, 2, 2) > 1e-9;
fprintf('node translation 30   : %d vertices moved, max displacement %.1f\n', nnz(moved), max(vecnorm(Vd - src.V, 2, 2)));
[fig, ax] = edreg.showMeshes({src, struct('V', Vd, 'F', src.F)}, Visible="off", ...
    Title="Node translation t (one node, 30 units)");
scatter3(ax, model.nodeV(kTop, 1), model.nodeV(kTop, 2), model.nodeV(kTop, 3), 80, 'k', 'filled');
exportgraphics(fig, fullfile(figDir, 't01_node_translation.png'), 'Resolution', 150); close(fig);

%% Per-node affine A: a local rotation/scale about the node position g_k
% A_k acts on (v - g_k), so the node itself stays fixed while its
% neighbourhood is rotated by 40 degrees about the up axis and scaled by 1.1.
rotUp = edreg.rodrigues(deg2rad(40) * axisUp);
P = P0;
[~, ~, A, t] = edreg.decodeParam(P, N);
A(:, :, kTop) = 1.1 * rotUp;
P = edreg.encodeParam(eye(3), zeros(3, 1), A, t);
Vd = edreg.deform(P, model);
fprintf('node affine (40 deg)  : %d vertices moved, max displacement %.1f\n', ...
    nnz(vecnorm(Vd - src.V, 2, 2) > 1e-9), max(vecnorm(Vd - src.V, 2, 2)));
[fig, ax] = edreg.showMeshes({src, struct('V', Vd, 'F', src.F)}, Visible="off", ...
    Title="Node affine A (one node, 40 deg rotation, scale 1.1)");
scatter3(ax, model.nodeV(kTop, 1), model.nodeV(kTop, 2), model.nodeV(kTop, 3), 80, 'k', 'filled');
exportgraphics(fig, fullfile(figDir, 't01_node_affine.png'), 'Resolution', 150); close(fig);

fprintf('t01 finished in %.1f s\n', toc(tStart));
