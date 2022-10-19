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
n_node_neighbor = 6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assert(n_vertex_neighbor > 1);
assert(n_node_neighbor > 1);

%opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'UseParallel', true); %off
%opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'Algorithm', 'trust-region-reflective',...
%    'SpecifyObjectiveGradient', true, 'FunctionTolerance', 1e-10);

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

fprintf('Run Optim... ');
%P0 = lsqnonlin(@(P)EfuncLocal(P, gpuSrcMesh, pntTar, gpuArray(G0), nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, gpuArray(Lap), w_rot, w_smooth, w_lap), P0, [], [], opts);
P0 = runOptim(P0, gpuSrcMesh, pntTar, gpuArray(G0), nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, gpuArray(Lap), w_rot, w_smooth, w_lap, true, false);

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

function P0 = runOptim(P0, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, show_iteration, show_final)
%     opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'Algorithm', 'levenberg-marquardt', 'SpecifyObjectiveGradient', true, ...
%     'FunctionTolerance', 1e-10, 'MaxIterations', PARAM.max_iterations, 'OutputFcn',@outfun);

    %opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'Algorithm', 'trust-region-reflective', 'SpecifyObjectiveGradient', true, ...
    %'FunctionTolerance', 1e-10, 'OutputFcn',@outfun);
    opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'SpecifyObjectiveGradient', true, ...
    'FunctionTolerance', 1e-10, 'OutputFcn',@outfun);
    
    P0 = lsqnonlin(@(P)EfuncLocal(P, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap), P0, [], [], opts);
    disp('done!');
    
    function stop = outfun(x, optimValues, state)
        figshow = false;
        stop = false;
        switch state
            case 'iter'
                if show_iteration
                    figshow = true;
                end
                
            case 'interrupt'
                  % Probably no action here. Check conditions to see  
                  % whether optimization should quit.
            case 'init'
                  % Setup for plots or guis
            case 'done'
                if show_final
                    figshow = true;
                end
                  % Cleanup of plots, guis, or final plot
            otherwise
        end
        
        if figshow
            [R0, T0, A0, t0] = decodeParam(x, size(idxNodeNeighbor, 2));
            
            v0 = SrcMesh.Vpoi';
            n0 = SrcMesh.Npoi';
            g0 = v0(:, nodeIdx);
            deformedV = deformED(v0, n0, g0, A0, t0, G0, R0, T0, idxNeighbor, distNeighbor);
            
            dispMesh(gather(deformedV), gather(SrcMesh.Fpoi), -90, 0);
        end
    end
end



function [F, J] = EfuncLocal(P0, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap)  
    n_nodes = length(nodeIdx);
    
    [R0, T0, A0, t0] = decodeParam(P0, n_nodes);
    
    v0 = SrcMesh.Vpoi';
    %n0 = SrcMesh.Npoi';
    g0 = v0(:, nodeIdx);
    
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
    v1_trans = squeeze(pagemtimes(A1, v1)) + g1 + t1;
    
    %n0 = gpuArray(reshape(n0, [size(n0, 1) 1 size(n0, 2)]));
    %deformedN = squeeze(sum(squeeze(pagemtimes(pagefun(@transpose, pagefun(@inv, gpuArray(A1))), n0)) .* distNeighbor, 3))';
        
    deformedV_local = squeeze(sum(distNeighbor .* v1_trans, 3))'; % Local Transform
    deformedV = (deformedV_local - G0(:)') * R0' + T0(:)' + G0(:)'; % Global Transform
    deformedV(any(invalidNeibor, 1), :) = v0(:, any(invalidNeibor, 1))';
    
    %[idxNearest, dists2] = multiQueryKNNSearchImpl(pointCloud(gather(deformedV)), pntTar.Location, 1); %idxNearest
    idxNearest = knnsearch(gather(deformedV), pntTar.Location, 'K', 1);
    Efit = (gather(deformedV(idxNearest, :)) - pntTar.Location) ./ sqrt(size(pntTar.Location, 1));
    %Efit = sqrt(sum(Efit.^2, 2));
    %Efit = sqrt(dists2 ./ size(pntTar.Location, 1));
    
    Elap = gather((Lap * deformedV) ./ full(diag(Lap)));
    
    Erot1 = (pagemtimes(A0, 'transpose', A0, 'none') - reshape(eye(3), 3, 3, 1)) ./ sqrt(n_nodes);
    Erot2 = (pagedet(A0) - 1.0) ./ sqrt(n_nodes);
    
    g_node1 = reshape(g0(:, idxNodeNeighbor'), [size(g0, 1) size(idxNodeNeighbor')]);
    t_node1 = reshape(t0(:, idxNodeNeighbor'), [size(t0, 1) size(idxNodeNeighbor')]);
    v_node1 = reshape(g_node1 - g0, [size(g_node1, 1) 1 size(g_node1, 2:3)]);
    Esmooth = gather(sum(squeeze(pagemtimes(A0, v_node1)) + g0 + t0 - g_node1 - t_node1, 3)) ./ sqrt(3 * n_nodes);
    %Esmooth = gather(sqrt(sum((squeeze(pagemtimes(A0, v_node1)) + g0 + t0 - g_node1 - t_node1).^2, [1 3]) ./ (3 * n_nodes)));
    
    F = [Efit(:); w_rot*Erot1(:); w_rot*Erot2(:); w_smooth*Esmooth(:); w_lap* Elap(:)];
    
    if nargout > 1
        nA = 9 * n_nodes;
        n_target = length(pntTar.Location);
        Px = (rotationVectorToMatrix(P0(1:3)+[1; 0; 0])-rotationVectorToMatrix(P0(1:3)));
        Py = (rotationVectorToMatrix(P0(1:3)+[0; 1; 0])-rotationVectorToMatrix(P0(1:3)));
        Pz = (rotationVectorToMatrix(P0(1:3)+[0; 0; 1])-rotationVectorToMatrix(P0(1:3)));
        
        Jfit = sparse(numel(Efit), numel(P0));
        Jrot1 = sparse(numel(Erot1), numel(P0));
        Jrot2 = sparse(numel(Erot2), numel(P0));
        Jsmooth = sparse(numel(Esmooth), numel(P0));
        Jlap = sparse(numel(Elap), numel(P0));
        
            Jfit(:, 1) = reshape(gather((deformedV_local(idxNearest, :) - G0(:)') * Px), [], 1);
            Jfit(:, 2) = reshape(gather((deformedV_local(idxNearest, :) - G0(:)') * Py), [], 1);
            Jfit(:, 3) = reshape(gather((deformedV_local(idxNearest, :) - G0(:)') * Pz), [], 1);
            Jfit(1:n_target, 4) = 1;
            Jfit((n_target + 1):(2 * n_target), 5) = 1;
            Jfit((2 * n_target + 1):(3 * n_target), 6) = 1;
            
            Jlap(:, 1) = reshape(gather(Lap * (deformedV_local - G0(:)') * Px), [], 1);
            Jlap(:, 2) = reshape(gather(Lap * (deformedV_local - G0(:)') * Py), [], 1);
            Jlap(:, 3) = reshape(gather(Lap * (deformedV_local - G0(:)') * Pz), [], 1);
            
            parfor i = 1:(numel(P0)-6)               
                if (i <= nA)
                    new_A0 = zeros(3, 3, n_nodes);
                    new_A0(i) = 1;
                    new_A1 = reshape(new_A0(:, :, idxNeighbor'), [size(new_A0, [1 2]) size(idxNeighbor')]);
                    new_v1_trans = squeeze(pagemtimes(new_A1, v1));
                    new_deformedV_local = squeeze(sum(distNeighbor .* new_v1_trans, 3))';
                    
                    Jfit(:, 6 + i) = reshape(gather(new_deformedV_local(idxNearest,:) * R0'), [], 1);
                    Jsmooth(:, 6 + i) = reshape(gather(sum(squeeze(pagemtimes(new_A0, v_node1)) + g0 + t0 - g_node1 - t_node1, 3)), [], 1);
                    Jlap(:, 6 + i) = reshape(gather(Lap * new_deformedV_local * R0'), [], 1);
                else
                    new_t0 = zeros(3, n_nodes);
                    new_t0(i - nA) = 1;
                    new_t1 = reshape(new_t0(:, idxNeighbor'), [size(new_t0, 1) size(idxNeighbor')]);
                    new_deformedV_local = squeeze(sum(distNeighbor .* new_t1, 3))';
                    
                    new_t_node1 = reshape(new_t0(:, idxNodeNeighbor'), [size(new_t0, 1) size(idxNodeNeighbor')]);
                    
                    Jfit(:, 6 + i) = reshape(gather(new_deformedV_local(idxNearest,:) * R0'), [], 1);
                    Jsmooth(:, 6 + i) = reshape(sum(new_t0 - new_t_node1, 3), [], 1);
                    Jlap(:, 6 + i) = reshape(gather(Lap * new_deformedV_local * R0'), [], 1);
                end
            end
            
            rot1_diff = zeros(9, 9, 700);
                rot1_diff(1, 1, :) = 2 * A0(1, 1, :); rot1_diff(2, 2, :) = A0(1, 1, :); rot1_diff(3, 3, :) = A0(1, 1, :); rot1_diff(4, 2, :) = A0(1, 1, :); rot1_diff(7, 3, :) = A0(1, 1, :);
                rot1_diff(2, 1, :) = A0(2, 2, :); rot1_diff(4, 1, :) = A0(2, 2, :); rot1_diff(5, 2, :) = 2 * A0(2, 2, :); rot1_diff(6, 3, :) = A0(2, 2, :); rot1_diff(8, 3, :) = A0(2, 2, :);
                rot1_diff(3, 1, :) = A0(3, 3, :); rot1_diff(6, 2, :) = A0(3, 3, :); rot1_diff(7, 1, :) = A0(3, 3, :); rot1_diff(8, 2, :) = A0(3, 3, :); rot1_diff(9, 3, :) = 2 * A0(3, 3, :);
                rot1_diff(1, 4, :) = 2 * A0(4, 4, :); rot1_diff(2, 5, :) = A0(4, 4, :); rot1_diff(3, 6, :) = A0(4, 4, :); rot1_diff(4, 5, :) = A0(4, 4, :); rot1_diff(7, 6, :) = A0(4, 4, :);
                rot1_diff(2, 4, :) = A0(5, 5, :); rot1_diff(4, 4, :) = A0(5, 5, :); rot1_diff(5, 5, :) = 2 * A0(5, 5, :); rot1_diff(6, 6, :) = A0(5, 5, :); rot1_diff(8, 6, :) = A0(5, 5, :);
                rot1_diff(3, 4, :) = A0(6, 6, :); rot1_diff(6, 5, :) = A0(6, 6, :); rot1_diff(7, 4, :) = A0(6, 6, :); rot1_diff(8, 5, :) = A0(6, 6, :); rot1_diff(9, 6, :) = 2 * A0(6, 6, :);
                rot1_diff(1, 7, :) = 2 * A0(7, 7, :); rot1_diff(2, 8, :) = A0(7, 7, :); rot1_diff(3, 9, :) = A0(7, 7, :); rot1_diff(4, 8, :) = A0(7, 7, :); rot1_diff(7, 9, :) = A0(7, 7, :);
                rot1_diff(2, 7, :) = A0(8, 8, :); rot1_diff(4, 7, :) = A0(8, 8, :); rot1_diff(5, 8, :) = 2 * A0(8, 8, :); rot1_diff(6, 9, :) = A0(8, 8, :); rot1_diff(8, 9, :) = A0(8, 8, :);
                rot1_diff(3, 7, :) = A0(9, 9, :); rot1_diff(6, 8, :) = A0(9, 9, :); rot1_diff(7, 7, :) = A0(9, 9, :); rot1_diff(8, 8, :) = A0(9, 9, :); rot1_diff(9, 9, :) = 2 * A0(9, 9, :);
            
            Jrot1(:, 7:(6 + nA)) = blkdiag(rot1_diff(:, :, 1), rot1_diff(:, :, 2), rot1_diff(:, :, 3), rot1_diff(:, :, 4), rot1_diff(:, :, 5),...
                                       rot1_diff(:, :, 6), rot1_diff(:, :, 7), rot1_diff(:, :, 8), rot1_diff(:, :, 9), rot1_diff(:, :, 10),...
                                       rot1_diff(:, :, 11), rot1_diff(:, :, 12), rot1_diff(:, :, 13), rot1_diff(:, :, 14), rot1_diff(:, :, 15),...
                                       rot1_diff(:, :, 16), rot1_diff(:, :, 17), rot1_diff(:, :, 18), rot1_diff(:, :, 19), rot1_diff(:, :, 20),...
                                       rot1_diff(:, :, 21), rot1_diff(:, :, 22), rot1_diff(:, :, 23), rot1_diff(:, :, 24), rot1_diff(:, :, 25),...
                                       rot1_diff(:, :, 26), rot1_diff(:, :, 27), rot1_diff(:, :, 28), rot1_diff(:, :, 29), rot1_diff(:, :, 30),...
                                       rot1_diff(:, :, 31), rot1_diff(:, :, 32), rot1_diff(:, :, 33), rot1_diff(:, :, 34), rot1_diff(:, :, 35),...
                                       rot1_diff(:, :, 36), rot1_diff(:, :, 37), rot1_diff(:, :, 38), rot1_diff(:, :, 39), rot1_diff(:, :, 40),...
                                       rot1_diff(:, :, 41), rot1_diff(:, :, 42), rot1_diff(:, :, 43), rot1_diff(:, :, 44), rot1_diff(:, :, 45),...
                                       rot1_diff(:, :, 46), rot1_diff(:, :, 47), rot1_diff(:, :, 48), rot1_diff(:, :, 49), rot1_diff(:, :, 50),...
                                       rot1_diff(:, :, 51), rot1_diff(:, :, 52), rot1_diff(:, :, 53), rot1_diff(:, :, 54), rot1_diff(:, :, 55),...
                                       rot1_diff(:, :, 56), rot1_diff(:, :, 57), rot1_diff(:, :, 58), rot1_diff(:, :, 59), rot1_diff(:, :, 60),...
                                       rot1_diff(:, :, 61), rot1_diff(:, :, 62), rot1_diff(:, :, 63), rot1_diff(:, :, 64), rot1_diff(:, :, 65),...
                                       rot1_diff(:, :, 66), rot1_diff(:, :, 67), rot1_diff(:, :, 68), rot1_diff(:, :, 69), rot1_diff(:, :, 70),...
                                       rot1_diff(:, :, 71), rot1_diff(:, :, 72), rot1_diff(:, :, 73), rot1_diff(:, :, 74), rot1_diff(:, :, 75),...
                                       rot1_diff(:, :, 76), rot1_diff(:, :, 77), rot1_diff(:, :, 78), rot1_diff(:, :, 79), rot1_diff(:, :, 80),...
                                       rot1_diff(:, :, 81), rot1_diff(:, :, 82), rot1_diff(:, :, 83), rot1_diff(:, :, 84), rot1_diff(:, :, 85),...
                                       rot1_diff(:, :, 86), rot1_diff(:, :, 87), rot1_diff(:, :, 88), rot1_diff(:, :, 89), rot1_diff(:, :, 90),...
                                       rot1_diff(:, :, 91), rot1_diff(:, :, 92), rot1_diff(:, :, 93), rot1_diff(:, :, 94), rot1_diff(:, :, 95),...
                                       rot1_diff(:, :, 96), rot1_diff(:, :, 97), rot1_diff(:, :, 98), rot1_diff(:, :, 99), rot1_diff(:, :, 100),...
                                       rot1_diff(:, :, 101), rot1_diff(:, :, 102), rot1_diff(:, :, 103), rot1_diff(:, :, 104), rot1_diff(:, :, 105),...
                                       rot1_diff(:, :, 106), rot1_diff(:, :, 107), rot1_diff(:, :, 108), rot1_diff(:, :, 109), rot1_diff(:, :, 110),...
                                       rot1_diff(:, :, 111), rot1_diff(:, :, 112), rot1_diff(:, :, 113), rot1_diff(:, :, 114), rot1_diff(:, :, 115),...
                                       rot1_diff(:, :, 116), rot1_diff(:, :, 117), rot1_diff(:, :, 118), rot1_diff(:, :, 119), rot1_diff(:, :, 120),...
                                       rot1_diff(:, :, 121), rot1_diff(:, :, 122), rot1_diff(:, :, 123), rot1_diff(:, :, 124), rot1_diff(:, :, 125),...
                                       rot1_diff(:, :, 126), rot1_diff(:, :, 127), rot1_diff(:, :, 128), rot1_diff(:, :, 129), rot1_diff(:, :, 130),...
                                       rot1_diff(:, :, 131), rot1_diff(:, :, 132), rot1_diff(:, :, 133), rot1_diff(:, :, 134), rot1_diff(:, :, 135),...
                                       rot1_diff(:, :, 136), rot1_diff(:, :, 137), rot1_diff(:, :, 138), rot1_diff(:, :, 139), rot1_diff(:, :, 140),...
                                       rot1_diff(:, :, 141), rot1_diff(:, :, 142), rot1_diff(:, :, 143), rot1_diff(:, :, 144), rot1_diff(:, :, 145),...
                                       rot1_diff(:, :, 146), rot1_diff(:, :, 147), rot1_diff(:, :, 148), rot1_diff(:, :, 149), rot1_diff(:, :, 150),...
                                       rot1_diff(:, :, 151), rot1_diff(:, :, 152), rot1_diff(:, :, 153), rot1_diff(:, :, 154), rot1_diff(:, :, 155),...
                                       rot1_diff(:, :, 156), rot1_diff(:, :, 157), rot1_diff(:, :, 158), rot1_diff(:, :, 159), rot1_diff(:, :, 160),...
                                       rot1_diff(:, :, 161), rot1_diff(:, :, 162), rot1_diff(:, :, 163), rot1_diff(:, :, 164), rot1_diff(:, :, 165),...
                                       rot1_diff(:, :, 166), rot1_diff(:, :, 167), rot1_diff(:, :, 168), rot1_diff(:, :, 169), rot1_diff(:, :, 170),...
                                       rot1_diff(:, :, 171), rot1_diff(:, :, 172), rot1_diff(:, :, 173), rot1_diff(:, :, 174), rot1_diff(:, :, 175),...
                                       rot1_diff(:, :, 176), rot1_diff(:, :, 177), rot1_diff(:, :, 178), rot1_diff(:, :, 179), rot1_diff(:, :, 180),...
                                       rot1_diff(:, :, 181), rot1_diff(:, :, 182), rot1_diff(:, :, 183), rot1_diff(:, :, 184), rot1_diff(:, :, 185),...
                                       rot1_diff(:, :, 186), rot1_diff(:, :, 187), rot1_diff(:, :, 188), rot1_diff(:, :, 189), rot1_diff(:, :, 190),...
                                       rot1_diff(:, :, 191), rot1_diff(:, :, 192), rot1_diff(:, :, 193), rot1_diff(:, :, 194), rot1_diff(:, :, 195),...
                                       rot1_diff(:, :, 196), rot1_diff(:, :, 197), rot1_diff(:, :, 198), rot1_diff(:, :, 199), rot1_diff(:, :, 200),...
                                       rot1_diff(:, :, 201), rot1_diff(:, :, 202), rot1_diff(:, :, 203), rot1_diff(:, :, 204), rot1_diff(:, :, 205),...
                                       rot1_diff(:, :, 206), rot1_diff(:, :, 207), rot1_diff(:, :, 208), rot1_diff(:, :, 209), rot1_diff(:, :, 210),...
                                       rot1_diff(:, :, 211), rot1_diff(:, :, 212), rot1_diff(:, :, 213), rot1_diff(:, :, 214), rot1_diff(:, :, 215),...
                                       rot1_diff(:, :, 216), rot1_diff(:, :, 217), rot1_diff(:, :, 218), rot1_diff(:, :, 219), rot1_diff(:, :, 220),...
                                       rot1_diff(:, :, 221), rot1_diff(:, :, 222), rot1_diff(:, :, 223), rot1_diff(:, :, 224), rot1_diff(:, :, 225),...
                                       rot1_diff(:, :, 226), rot1_diff(:, :, 227), rot1_diff(:, :, 228), rot1_diff(:, :, 229), rot1_diff(:, :, 230),...
                                       rot1_diff(:, :, 231), rot1_diff(:, :, 232), rot1_diff(:, :, 233), rot1_diff(:, :, 234), rot1_diff(:, :, 235),...
                                       rot1_diff(:, :, 236), rot1_diff(:, :, 237), rot1_diff(:, :, 238), rot1_diff(:, :, 239), rot1_diff(:, :, 240),...
                                       rot1_diff(:, :, 241), rot1_diff(:, :, 242), rot1_diff(:, :, 243), rot1_diff(:, :, 244), rot1_diff(:, :, 245),...
                                       rot1_diff(:, :, 246), rot1_diff(:, :, 247), rot1_diff(:, :, 248), rot1_diff(:, :, 249), rot1_diff(:, :, 250),...
                                       rot1_diff(:, :, 251), rot1_diff(:, :, 252), rot1_diff(:, :, 253), rot1_diff(:, :, 254), rot1_diff(:, :, 255),...
                                       rot1_diff(:, :, 256), rot1_diff(:, :, 257), rot1_diff(:, :, 258), rot1_diff(:, :, 259), rot1_diff(:, :, 260),...
                                       rot1_diff(:, :, 261), rot1_diff(:, :, 262), rot1_diff(:, :, 263), rot1_diff(:, :, 264), rot1_diff(:, :, 265),...
                                       rot1_diff(:, :, 266), rot1_diff(:, :, 267), rot1_diff(:, :, 268), rot1_diff(:, :, 269), rot1_diff(:, :, 270),...
                                       rot1_diff(:, :, 271), rot1_diff(:, :, 272), rot1_diff(:, :, 273), rot1_diff(:, :, 274), rot1_diff(:, :, 275),...
                                       rot1_diff(:, :, 276), rot1_diff(:, :, 277), rot1_diff(:, :, 278), rot1_diff(:, :, 279), rot1_diff(:, :, 280),...
                                       rot1_diff(:, :, 281), rot1_diff(:, :, 282), rot1_diff(:, :, 283), rot1_diff(:, :, 284), rot1_diff(:, :, 285),...
                                       rot1_diff(:, :, 286), rot1_diff(:, :, 287), rot1_diff(:, :, 288), rot1_diff(:, :, 289), rot1_diff(:, :, 290),...
                                       rot1_diff(:, :, 291), rot1_diff(:, :, 292), rot1_diff(:, :, 293), rot1_diff(:, :, 294), rot1_diff(:, :, 295),...
                                       rot1_diff(:, :, 296), rot1_diff(:, :, 297), rot1_diff(:, :, 298), rot1_diff(:, :, 299), rot1_diff(:, :, 300),...
                                       rot1_diff(:, :, 301), rot1_diff(:, :, 302), rot1_diff(:, :, 303), rot1_diff(:, :, 304), rot1_diff(:, :, 305),...
                                       rot1_diff(:, :, 306), rot1_diff(:, :, 307), rot1_diff(:, :, 308), rot1_diff(:, :, 309), rot1_diff(:, :, 310),...
                                       rot1_diff(:, :, 311), rot1_diff(:, :, 312), rot1_diff(:, :, 313), rot1_diff(:, :, 314), rot1_diff(:, :, 315),...
                                       rot1_diff(:, :, 316), rot1_diff(:, :, 317), rot1_diff(:, :, 318), rot1_diff(:, :, 319), rot1_diff(:, :, 320),...
                                       rot1_diff(:, :, 321), rot1_diff(:, :, 322), rot1_diff(:, :, 323), rot1_diff(:, :, 324), rot1_diff(:, :, 325),...
                                       rot1_diff(:, :, 326), rot1_diff(:, :, 327), rot1_diff(:, :, 328), rot1_diff(:, :, 329), rot1_diff(:, :, 330),...
                                       rot1_diff(:, :, 331), rot1_diff(:, :, 332), rot1_diff(:, :, 333), rot1_diff(:, :, 334), rot1_diff(:, :, 335),...
                                       rot1_diff(:, :, 336), rot1_diff(:, :, 337), rot1_diff(:, :, 338), rot1_diff(:, :, 339), rot1_diff(:, :, 340),...
                                       rot1_diff(:, :, 341), rot1_diff(:, :, 342), rot1_diff(:, :, 343), rot1_diff(:, :, 344), rot1_diff(:, :, 345),...
                                       rot1_diff(:, :, 346), rot1_diff(:, :, 347), rot1_diff(:, :, 348), rot1_diff(:, :, 349), rot1_diff(:, :, 350),...
                                       rot1_diff(:, :, 351), rot1_diff(:, :, 352), rot1_diff(:, :, 353), rot1_diff(:, :, 354), rot1_diff(:, :, 355),...
                                       rot1_diff(:, :, 356), rot1_diff(:, :, 357), rot1_diff(:, :, 358), rot1_diff(:, :, 359), rot1_diff(:, :, 360),...
                                       rot1_diff(:, :, 361), rot1_diff(:, :, 362), rot1_diff(:, :, 363), rot1_diff(:, :, 364), rot1_diff(:, :, 365),...
                                       rot1_diff(:, :, 366), rot1_diff(:, :, 367), rot1_diff(:, :, 368), rot1_diff(:, :, 369), rot1_diff(:, :, 370),...
                                       rot1_diff(:, :, 371), rot1_diff(:, :, 372), rot1_diff(:, :, 373), rot1_diff(:, :, 374), rot1_diff(:, :, 375),...
                                       rot1_diff(:, :, 376), rot1_diff(:, :, 377), rot1_diff(:, :, 378), rot1_diff(:, :, 379), rot1_diff(:, :, 380),...
                                       rot1_diff(:, :, 381), rot1_diff(:, :, 382), rot1_diff(:, :, 383), rot1_diff(:, :, 384), rot1_diff(:, :, 385),...
                                       rot1_diff(:, :, 386), rot1_diff(:, :, 387), rot1_diff(:, :, 388), rot1_diff(:, :, 389), rot1_diff(:, :, 390),...
                                       rot1_diff(:, :, 391), rot1_diff(:, :, 392), rot1_diff(:, :, 393), rot1_diff(:, :, 394), rot1_diff(:, :, 395),...
                                       rot1_diff(:, :, 396), rot1_diff(:, :, 397), rot1_diff(:, :, 398), rot1_diff(:, :, 399), rot1_diff(:, :, 400),...
                                       rot1_diff(:, :, 401), rot1_diff(:, :, 402), rot1_diff(:, :, 403), rot1_diff(:, :, 404), rot1_diff(:, :, 405),...
                                       rot1_diff(:, :, 406), rot1_diff(:, :, 407), rot1_diff(:, :, 408), rot1_diff(:, :, 409), rot1_diff(:, :, 410),...
                                       rot1_diff(:, :, 411), rot1_diff(:, :, 412), rot1_diff(:, :, 413), rot1_diff(:, :, 414), rot1_diff(:, :, 415),...
                                       rot1_diff(:, :, 416), rot1_diff(:, :, 417), rot1_diff(:, :, 418), rot1_diff(:, :, 419), rot1_diff(:, :, 420),...
                                       rot1_diff(:, :, 421), rot1_diff(:, :, 422), rot1_diff(:, :, 423), rot1_diff(:, :, 424), rot1_diff(:, :, 425),...
                                       rot1_diff(:, :, 426), rot1_diff(:, :, 427), rot1_diff(:, :, 428), rot1_diff(:, :, 429), rot1_diff(:, :, 430),...
                                       rot1_diff(:, :, 431), rot1_diff(:, :, 432), rot1_diff(:, :, 433), rot1_diff(:, :, 434), rot1_diff(:, :, 435),...
                                       rot1_diff(:, :, 436), rot1_diff(:, :, 437), rot1_diff(:, :, 438), rot1_diff(:, :, 439), rot1_diff(:, :, 440),...
                                       rot1_diff(:, :, 441), rot1_diff(:, :, 442), rot1_diff(:, :, 443), rot1_diff(:, :, 444), rot1_diff(:, :, 445),...
                                       rot1_diff(:, :, 446), rot1_diff(:, :, 447), rot1_diff(:, :, 448), rot1_diff(:, :, 449), rot1_diff(:, :, 450),...
                                       rot1_diff(:, :, 451), rot1_diff(:, :, 452), rot1_diff(:, :, 453), rot1_diff(:, :, 454), rot1_diff(:, :, 455),...
                                       rot1_diff(:, :, 456), rot1_diff(:, :, 457), rot1_diff(:, :, 458), rot1_diff(:, :, 459), rot1_diff(:, :, 460),...
                                       rot1_diff(:, :, 461), rot1_diff(:, :, 462), rot1_diff(:, :, 463), rot1_diff(:, :, 464), rot1_diff(:, :, 465),...
                                       rot1_diff(:, :, 466), rot1_diff(:, :, 467), rot1_diff(:, :, 468), rot1_diff(:, :, 469), rot1_diff(:, :, 470),...
                                       rot1_diff(:, :, 471), rot1_diff(:, :, 472), rot1_diff(:, :, 473), rot1_diff(:, :, 474), rot1_diff(:, :, 475),...
                                       rot1_diff(:, :, 476), rot1_diff(:, :, 477), rot1_diff(:, :, 478), rot1_diff(:, :, 479), rot1_diff(:, :, 480),...
                                       rot1_diff(:, :, 481), rot1_diff(:, :, 482), rot1_diff(:, :, 483), rot1_diff(:, :, 484), rot1_diff(:, :, 485),...
                                       rot1_diff(:, :, 486), rot1_diff(:, :, 487), rot1_diff(:, :, 488), rot1_diff(:, :, 489), rot1_diff(:, :, 490),...
                                       rot1_diff(:, :, 491), rot1_diff(:, :, 492), rot1_diff(:, :, 493), rot1_diff(:, :, 494), rot1_diff(:, :, 495),...
                                       rot1_diff(:, :, 496), rot1_diff(:, :, 497), rot1_diff(:, :, 498), rot1_diff(:, :, 499), rot1_diff(:, :, 500),...
                                       rot1_diff(:, :, 501), rot1_diff(:, :, 502), rot1_diff(:, :, 503), rot1_diff(:, :, 504), rot1_diff(:, :, 505),...
                                       rot1_diff(:, :, 506), rot1_diff(:, :, 507), rot1_diff(:, :, 508), rot1_diff(:, :, 509), rot1_diff(:, :, 510),...
                                       rot1_diff(:, :, 511), rot1_diff(:, :, 512), rot1_diff(:, :, 513), rot1_diff(:, :, 514), rot1_diff(:, :, 515),...
                                       rot1_diff(:, :, 516), rot1_diff(:, :, 517), rot1_diff(:, :, 518), rot1_diff(:, :, 519), rot1_diff(:, :, 520),...
                                       rot1_diff(:, :, 521), rot1_diff(:, :, 522), rot1_diff(:, :, 523), rot1_diff(:, :, 524), rot1_diff(:, :, 525),...
                                       rot1_diff(:, :, 526), rot1_diff(:, :, 527), rot1_diff(:, :, 528), rot1_diff(:, :, 529), rot1_diff(:, :, 530),...
                                       rot1_diff(:, :, 531), rot1_diff(:, :, 532), rot1_diff(:, :, 533), rot1_diff(:, :, 534), rot1_diff(:, :, 535),...
                                       rot1_diff(:, :, 536), rot1_diff(:, :, 537), rot1_diff(:, :, 538), rot1_diff(:, :, 539), rot1_diff(:, :, 540),...
                                       rot1_diff(:, :, 541), rot1_diff(:, :, 542), rot1_diff(:, :, 543), rot1_diff(:, :, 544), rot1_diff(:, :, 545),...
                                       rot1_diff(:, :, 546), rot1_diff(:, :, 547), rot1_diff(:, :, 548), rot1_diff(:, :, 549), rot1_diff(:, :, 550),...
                                       rot1_diff(:, :, 551), rot1_diff(:, :, 552), rot1_diff(:, :, 553), rot1_diff(:, :, 554), rot1_diff(:, :, 555),...
                                       rot1_diff(:, :, 556), rot1_diff(:, :, 557), rot1_diff(:, :, 558), rot1_diff(:, :, 559), rot1_diff(:, :, 560),...
                                       rot1_diff(:, :, 561), rot1_diff(:, :, 562), rot1_diff(:, :, 563), rot1_diff(:, :, 564), rot1_diff(:, :, 565),...
                                       rot1_diff(:, :, 566), rot1_diff(:, :, 567), rot1_diff(:, :, 568), rot1_diff(:, :, 569), rot1_diff(:, :, 570),...
                                       rot1_diff(:, :, 571), rot1_diff(:, :, 572), rot1_diff(:, :, 573), rot1_diff(:, :, 574), rot1_diff(:, :, 575),...
                                       rot1_diff(:, :, 576), rot1_diff(:, :, 577), rot1_diff(:, :, 578), rot1_diff(:, :, 579), rot1_diff(:, :, 580),...
                                       rot1_diff(:, :, 581), rot1_diff(:, :, 582), rot1_diff(:, :, 583), rot1_diff(:, :, 584), rot1_diff(:, :, 585),...
                                       rot1_diff(:, :, 586), rot1_diff(:, :, 587), rot1_diff(:, :, 588), rot1_diff(:, :, 589), rot1_diff(:, :, 590),...
                                       rot1_diff(:, :, 591), rot1_diff(:, :, 592), rot1_diff(:, :, 593), rot1_diff(:, :, 594), rot1_diff(:, :, 595),...
                                       rot1_diff(:, :, 596), rot1_diff(:, :, 597), rot1_diff(:, :, 598), rot1_diff(:, :, 599), rot1_diff(:, :, 600),...
                                       rot1_diff(:, :, 601), rot1_diff(:, :, 602), rot1_diff(:, :, 603), rot1_diff(:, :, 604), rot1_diff(:, :, 605),...
                                       rot1_diff(:, :, 606), rot1_diff(:, :, 607), rot1_diff(:, :, 608), rot1_diff(:, :, 609), rot1_diff(:, :, 610),...
                                       rot1_diff(:, :, 611), rot1_diff(:, :, 612), rot1_diff(:, :, 613), rot1_diff(:, :, 614), rot1_diff(:, :, 615),...
                                       rot1_diff(:, :, 616), rot1_diff(:, :, 617), rot1_diff(:, :, 618), rot1_diff(:, :, 619), rot1_diff(:, :, 620),...
                                       rot1_diff(:, :, 621), rot1_diff(:, :, 622), rot1_diff(:, :, 623), rot1_diff(:, :, 624), rot1_diff(:, :, 625),...
                                       rot1_diff(:, :, 626), rot1_diff(:, :, 627), rot1_diff(:, :, 628), rot1_diff(:, :, 629), rot1_diff(:, :, 630),...
                                       rot1_diff(:, :, 631), rot1_diff(:, :, 632), rot1_diff(:, :, 633), rot1_diff(:, :, 634), rot1_diff(:, :, 635),...
                                       rot1_diff(:, :, 636), rot1_diff(:, :, 637), rot1_diff(:, :, 638), rot1_diff(:, :, 639), rot1_diff(:, :, 640),...
                                       rot1_diff(:, :, 641), rot1_diff(:, :, 642), rot1_diff(:, :, 643), rot1_diff(:, :, 644), rot1_diff(:, :, 645),...
                                       rot1_diff(:, :, 646), rot1_diff(:, :, 647), rot1_diff(:, :, 648), rot1_diff(:, :, 649), rot1_diff(:, :, 650),...
                                       rot1_diff(:, :, 651), rot1_diff(:, :, 652), rot1_diff(:, :, 653), rot1_diff(:, :, 654), rot1_diff(:, :, 655),...
                                       rot1_diff(:, :, 656), rot1_diff(:, :, 657), rot1_diff(:, :, 658), rot1_diff(:, :, 659), rot1_diff(:, :, 660),...
                                       rot1_diff(:, :, 661), rot1_diff(:, :, 662), rot1_diff(:, :, 663), rot1_diff(:, :, 664), rot1_diff(:, :, 665),...
                                       rot1_diff(:, :, 666), rot1_diff(:, :, 667), rot1_diff(:, :, 668), rot1_diff(:, :, 669), rot1_diff(:, :, 670),...
                                       rot1_diff(:, :, 671), rot1_diff(:, :, 672), rot1_diff(:, :, 673), rot1_diff(:, :, 674), rot1_diff(:, :, 675),...
                                       rot1_diff(:, :, 676), rot1_diff(:, :, 677), rot1_diff(:, :, 678), rot1_diff(:, :, 679), rot1_diff(:, :, 680),...
                                       rot1_diff(:, :, 681), rot1_diff(:, :, 682), rot1_diff(:, :, 683), rot1_diff(:, :, 684), rot1_diff(:, :, 685),...
                                       rot1_diff(:, :, 686), rot1_diff(:, :, 687), rot1_diff(:, :, 688), rot1_diff(:, :, 689), rot1_diff(:, :, 690),...
                                       rot1_diff(:, :, 691), rot1_diff(:, :, 692), rot1_diff(:, :, 693), rot1_diff(:, :, 694), rot1_diff(:, :, 695),...
                                       rot1_diff(:, :, 696), rot1_diff(:, :, 697), rot1_diff(:, :, 698), rot1_diff(:, :, 699), rot1_diff(:, :, 700));
                
            for i = 1:n_nodes
                A = A0(:, :, i);
                A = A';
                
                Jrot2(i, (7 + 9 * (i - 1))) = A(5) * A(9) - A(6) * A(8);
                Jrot2(i, (8 + 9 * (i - 1))) = A(6) * A(7) - A(4) * A(9);
                Jrot2(i, (9 + 9 * (i - 1))) = A(4) * A(8) - A(5) * A(7);
                Jrot2(i, (10 + 9 * (i - 1))) = A(3) * A(8) - A(2) * A(9);
                Jrot2(i, (11 + 9 * (i - 1))) = A(1) * A(9) - A(3) * A(7);
                Jrot2(i, (12 + 9 * (i - 1))) = A(2) * A(7) - A(1) * A(8);
                Jrot2(i, (13 + 9 * (i - 1))) = A(2) * A(6) - A(3) * A(5);
                Jrot2(i, (14 + 9 * (i - 1))) = A(3) * A(4) - A(1) * A(6);
                Jrot2(i, (15 + 9 * (i - 1))) = A(1) * A(5) - A(2) * A(4);
            end
        
           
        J = [Jfit ./ sqrt(size(pntTar.Location, 1));
            w_rot * Jrot1 ./ sqrt(n_nodes);
            w_rot * Jrot2 ./ sqrt(n_nodes);
            w_smooth * Jsmooth ./ sqrt(3 * n_nodes);
            w_lap * Jlap ./ reshape(repmat(full(diag(gather(Lap))), [1 3]), [], 1)];
    end
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