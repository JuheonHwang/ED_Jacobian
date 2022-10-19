clear; close all; fclose all;
DIR = 'Dataset7CResult';
Src = 'mat_mesh_020';
Tar = 'mat_mesh_025';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_vertex_neighbor = 4;
EDSampleRate = 0.01;
w_rot = 1.0; %100;
w_smooth = 0.1; %1000;
w_lap = 0.05;
distanceOnMesh = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_node_neighbor = round(1 / EDSampleRate);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assert(n_vertex_neighbor > 1);
assert(n_node_neighbor > 1);

opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'UseParallel', true); %off

SrcMesh = load(fullfile(DIR, Src));
TarMesh = load(fullfile(DIR, Tar));

pntSrc = pointCloud(SrcMesh.Vpoi);
pntTar = pointCloud(TarMesh.Vpoi);

n_verts = size(SrcMesh.Vpoi, 1);
n_nodes = round(n_verts * EDSampleRate);

nodeIdx = randperm(n_verts, n_nodes);

pntEDNodes = pointCloud(SrcMesh.Vpoi(nodeIdx, :));

fprintf("The number of points of source : %d\n", n_verts);
fprintf("The number of points of target : %d\n", size(TarMesh.Vpoi, 1));
fprintf("The number of points of ED Nodes : %d\n", n_nodes);

SrcMesh.Npoi = vertexNormal(triangulation(SrcMesh.Fpoi, SrcMesh.Vpoi));
TarMesh.Npoi = vertexNormal(triangulation(TarMesh.Fpoi, TarMesh.Vpoi));


R0 = eye(3); T0 = [0, 0, 0];
A0 = repmat(reshape(eye(3), [3 3 1]), [1 1 n_nodes]);
t0 = zeros(3, n_nodes);
P0 = encodeParam(R0, T0, A0, t0);
G0 = mean(SrcMesh.Vpoi, 1);

if distanceOnMesh
    s = [SrcMesh.Fpoi(:, 1); SrcMesh.Fpoi(:, 2); SrcMesh.Fpoi(:, 3)];
    t = [SrcMesh.Fpoi(:, 2); SrcMesh.Fpoi(:, 3); SrcMesh.Fpoi(:, 1)];
    w = sqrt(sum((SrcMesh.Vpoi(s, :) - SrcMesh.Vpoi(t, :)).^2, 2));
    G = graph(s, t, w);
    G = simplify(G);
    Lap = laplacian(G);
    fprintf('Get Distances... ');
    D = distances(G);
    D = D(:, nodeIdx);
    fprintf('Get Nearests... ');
    [distNeighbor, idxNeighbor] = mink(D, n_vertex_neighbor+1, 2);
    distNeighbor = distNeighbor'; idxNeighbor = idxNeighbor';
    D = D(nodeIdx, :);
    [distNodeNeighbor, idxNodeNeighbor] = mink(D, n_node_neighbor, 2);
    idxNodeNeighbor = idxNodeNeighbor';
    assert(~any(isnan(distNodeNeighbor(:))));
    
    distNeighbor = 1.0 - (distNeighbor ./ distNeighbor(n_vertex_neighbor+1, :));
    idxNeighbor = idxNeighbor(1:n_vertex_neighbor, :); distNeighbor = distNeighbor(1:n_vertex_neighbor, :);
else
    [idxNeighbor, distNeighbor] = multiQueryKNNSearchImpl(pntEDNodes, pntSrc.Location, n_vertex_neighbor+2); %#ok<*UNRCH>
    idxNodeNeighbor = multiQueryKNNSearchImpl(pntEDNodes, pntEDNodes.Location, n_node_neighbor+1);
    distNeighbor = sqrt(distNeighbor);
    
    distNeighbor = 1.0 - (distNeighbor ./ distNeighbor(n_vertex_neighbor+2, :));
    idxNeighbor = idxNeighbor(2:(1+n_vertex_neighbor), :); distNeighbor = distNeighbor(2:(1+n_vertex_neighbor), :);
    idxNodeNeighbor = idxNodeNeighbor(2:(1+n_node_neighbor), :);
end

distNeighbor = distNeighbor ./ sum(distNeighbor, 1); % Normalize Dist Weights

invalidIdx = any(isnan(distNeighbor), 1);
distNeighbor(:, invalidIdx) = 1 / n_node_neighbor;
idxNeighbor(:, invalidIdx) = NaN;


dispMesh(SrcMesh.Vpoi, SrcMesh.Fpoi, -90, 0);
dispMesh(TarMesh.Vpoi, TarMesh.Fpoi, -90, 0);
gpuSrcMesh = SrcMesh;
gpuSrcMesh.Vpoi = gpuArray(gpuSrcMesh.Vpoi);
gpuSrcMesh.Npoi = gpuArray(gpuSrcMesh.Npoi);
pntTar.Normal = TarMesh.Npoi;


P0 = lsqnonlin(@(P)EfuncLocal(P, gpuSrcMesh, pntTar, gpuArray(G0), nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, gpuArray(Lap), w_rot, w_smooth, w_lap), P0, [], [], opts);

[R0, T0, A0, t0] = decodeParam(P0, n_nodes);

v0 = SrcMesh.Vpoi';
g0 = v0(:, nodeIdx);
n0 = SrcMesh.Npoi';

deformedV = deformED(v0, n0, g0, A0, t0, G0, R0, T0, idxNeighbor, distNeighbor);


dispMesh(deformedV, SrcMesh.Fpoi, -90, 0);
dispMesh(SrcMesh.Vpoi, SrcMesh.Fpoi, -90, 0);
dispMesh(TarMesh.Vpoi, TarMesh.Fpoi, -90, 0);

jwutils.saveMesh(deformedV, SrcMesh.Fpoi, -90, 0, 'Front.png');
jwutils.saveMesh(deformedV, SrcMesh.Fpoi, -90, 180, 'Back.png');



function F = EfuncLocal(P0, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap)  
    n_nodes = length(nodeIdx);
    
    [R0, T0, A0, t0] = decodeParam(P0, n_nodes); 
    
    v0 = SrcMesh.Vpoi';
    n0 = SrcMesh.Npoi';
    g0 = v0(:, nodeIdx);

    %[deformedV, deformedN] = deformED(v0, n0, g0, A0, t0, G0, R0, T0, idxNeighbor, distNeighbor);
    deformedV = deformED(v0, n0, g0, A0, t0, G0, R0, T0, idxNeighbor, distNeighbor);
    
    % [idxNearest, dists] = multiQueryKNNSearchImpl(pntTar, gather(deformedV), 1); %idxNearest
    % Efit = sqrt(dists ./ size(deformedV, 1));
    % Efit = gather(min(Efit .* dot(deformedN, pntTar.Normal(idxNearest, :), 2)', 0));
    
    % Efit = (pntTar.Location(idxNearest, :) - deformedV) ./ sqrt(size(deformedV, 1));
    
    %[~, dists1] = multiQueryKNNSearchImpl(pntTar, gather(deformedV), 1); %idxNearest
    [idxNearest, dists2] = multiQueryKNNSearchImpl(pointCloud(gather(deformedV)), pntTar.Location, 1); %idxNearest
    Efit = sqrt(dists2 ./ size(pntTar.Location, 1));
    %Efit = gather(dot(pntTar.Location - deformedV(idxNearest, :), pntTar.Normal, 2) ./ sqrt(size(pntTar.Location, 1)));
    
    
    %Efit = [sqrt(dists1 ./ size(deformedV, 1)) sqrt(dists2 ./ size(pntTar.Location, 1))];
    %Efit = gather(min(Efit .* dot(deformedN(idxNearest, :), pntTar.Normal, 2)', 0));
    Elap = gather((Lap * deformedV) ./ full(diag(Lap)));
    
    
    
    
    % Efit = gather(min(Efit .* dot(deformedN, pntTar.Normal(idxNearest, :), 2)', 0));
    
    Erot1 = (pagemtimes(A0, 'transpose', A0, 'none') - reshape(eye(3), 3, 3, 1)) ./ sqrt(n_nodes);
    Erot2 = (pagedet(A0) - 1.0) ./ sqrt(n_nodes);
    
    g1 = reshape(g0(:, idxNodeNeighbor'), [size(g0, 1) size(idxNodeNeighbor')]);
    t1 = reshape(t0(:, idxNodeNeighbor'), [size(t0, 1) size(idxNodeNeighbor')]);
    v1 = reshape(g1 - g0, [size(g1, 1) 1 size(g1, 2:3)]);
    Esmooth = gather(sqrt(sum((squeeze(pagemtimes(A0, v1)) + g0 + t0 - g1 - t1).^2, [1 3]) ./ (3 * n_nodes)));
    
    F = [Efit(:); w_rot*Erot1(:); w_rot*Erot2(:); w_smooth*Esmooth(:); w_lap* Elap(:)];
end

function [deformedV, deformedN] = deformED(v0, n0, g0, A0, t0, G0, R0, T0, idxNeighbor, distNeighbor)
    n_verts = size(distNeighbor, 2);
    n_neighbor = size(distNeighbor, 1);
    assert(n_verts > n_neighbor);
    
    distNeighbor = reshape(distNeighbor', [1 size(distNeighbor')]);
    invalidNeibor = isnan(idxNeighbor);
    idxNeighbor(invalidNeibor) = 1;
    
    g1 = reshape(g0(:, idxNeighbor'), [size(g0, 1) size(idxNeighbor')]);
    t1 = reshape(t0(:, idxNeighbor'), [size(t0, 1) size(idxNeighbor')]);
    A1 = reshape(A0(:, :, idxNeighbor'), [size(A0, [1 2]) size(idxNeighbor')]);
    v1 = reshape(v0 - g1, [size(g1, 1) 1 size(g1, 2:3)]);
    v1 = squeeze(pagemtimes(A1, v1)) + g1 + t1;
    
    n0 = gpuArray(reshape(n0, [size(n0, 1) 1 size(n0, 2)]));
    
    if nargout > 1
        deformedN = squeeze(sum(squeeze(pagemtimes(pagefun(@transpose, pagefun(@inv, gpuArray(A1))), n0)) .* distNeighbor, 3))';
    end
    
    deformedV = squeeze(sum(distNeighbor .* v1, 3))'; % Local Transform
    deformedV = (deformedV - G0(:)') * R0' + T0(:)' + G0(:)'; % Global Transform
    deformedV(any(invalidNeibor, 1), :) = v0(:, any(invalidNeibor, 1))';
end

function D = pagedet(X)
    D = X(1, 1, :).*X(2, 2, :).*X(3, 3, :) + X(1, 2, :).*X(2, 3, :).*X(3, 1, :) + X(1, 3, :).*X(2, 1, :).*X(3, 2, :) ...
        - X(1, 3, :).*X(2, 2, :).*X(3, 1, :) - X(1, 2, :).*X(2, 1, :).*X(3, 3, :) - X(1, 1, :).*X(2, 3, :).*X(3, 2, :);
end

function dispMesh(V, F, rot1, rot2)
    figure; 
    jwutils.dispMesh(V, F, [0.8 0.8 0.8 1.0]);
    camorbit(rot1, 0, 'data', [0 0 1]);
    camorbit(rot2, 0, 'data ', [1 0 0]);
    axis off;
end

function P = encodeParam(R, T, A, t)
    n_verts = size(A, 3);
    assert(size(t, 2) == n_verts);
    P = [rotationMatrixToVector(R) T(:)', A(:)', t(:)']';
end

function [R, T, A, t] = decodeParam(P, n_verts)
    P = P(:);
    R = rotationVectorToMatrix(P(1:3)');
    T = P(4:6)';
    nA = 9*n_verts;
    A = P(7:(7+nA-1));
    t = P((7+nA):end);
    A = reshape(A, [3, 3, n_verts]);
    t = reshape(t, [3, n_verts]);
end