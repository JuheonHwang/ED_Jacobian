clear; close all; fclose all;
DIR = 'Dataset7CResult';
Src = 'mat_mesh_020';
Tar = 'mat_mesh_022';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_vertex_neighbor = 4;
EDSampleRate = 0.01;
w_rot = 1.0; %100;
w_smooth = 0.1; %1000;
w_lap = 0.01;
distanceOnMesh = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_node_neighbor = 6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assert(n_vertex_neighbor > 1);
assert(n_node_neighbor > 1);

%opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'UseParallel', true); %off
%opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'Algorithm', 'trust-region-reflective',...
%    'SpecifyObjectiveGradient', true, 'FunctionTolerance', 1e-10);

[src_face, src_pts] = jwutils.plyread('simple_poisson_mesh_020.ply', 'tri');
[tar_face, tar_pts] = jwutils.plyread('simple_poisson_mesh_022.ply', 'tri');
SrcMesh.Vpoi = src_pts; SrcMesh.Fpoi = src_face;
TarMesh.Vpoi = tar_pts; TarMesh.Fpoi = tar_face;

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
P0 = runOptim(P0, gpuSrcMesh, pntTar, gpuArray(G0), nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, gpuArray(Lap), w_rot, w_smooth, w_lap, R0, T0, A0, t0, true, false);

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

function P0 = runOptim(P0, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, R_, T_, A_, t_, show_iteration, show_final)
%     opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'Algorithm', 'levenberg-marquardt', 'SpecifyObjectiveGradient', true, ...
%     'FunctionTolerance', 1e-10, 'MaxIterations', PARAM.max_iterations, 'OutputFcn',@outfun);

    opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'SpecifyObjectiveGradient', true, ...
    'FunctionTolerance', 1e-10, 'MaxIter', 10000, 'OutputFcn',@outfun);
    %opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'UseParallel', true, 'OutputFcn', @outfun); %off
    
    P0 = lsqnonlin(@(P)EfuncLocal(P, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, R_, T_, A_, t_), P0, [], [], opts);
    disp('done!');
    
    function stop = outfun(x, optimValues, state)
        figshow = false;
        stop = false;
        switch state
            case 'iter'
                if show_iteration
%                     if (rem(optimValues.funccount, 25) == 0)
%                         figshow = true;
%                     end
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
            pause(1)
        end
    end
end



function [F, J] = EfuncLocal(P0, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, R_, T_, A_, t_, indices)  
    n_nodes = length(nodeIdx);
    
    [R0, T0, A0, t0] = decodeParam(P0, n_nodes);
    [R0, T0, A0, t0, finds_nearests] = replaceParam(R0, T0, A0, t0, R_, T_, A_, t_);
    
    v0 = SrcMesh.Vpoi';
    %n0 = SrcMesh.Npoi';
    g0 = v0(:, nodeIdx);
    
    n_verts = size(distNeighbor, 2);
    n_neighbor = size(distNeighbor, 1);
    assert(n_verts > n_neighbor);
    
    distNeighbor_ = reshape(distNeighbor', [1 size(distNeighbor')]);
    invalidNeibor = isnan(idxNeighbor);
    idxNeighbor(invalidNeibor) = 1;
    
    g1 = reshape(g0(:, idxNeighbor'), [size(g0, 1) size(idxNeighbor')]);
    t1 = reshape(t0(:, idxNeighbor'), [size(t0, 1) size(idxNeighbor')]);
    A1 = reshape(A0(:, :, idxNeighbor'), [size(A0, [1 2]) size(idxNeighbor')]);
    v1 = reshape(v0 - g1, [size(g1, 1) 1 size(g1, 2:3)]);
    v1_trans = squeeze(pagemtimes(A1, v1)) + g1 + t1;
    
    %n0 = gpuArray(reshape(n0, [size(n0, 1) 1 size(n0, 2)]));
    %deformedN = squeeze(sum(squeeze(pagemtimes(pagefun(@transpose, pagefun(@inv, gpuArray(A1))), n0)) .* distNeighbor, 3))';
        
    deformedV_local = squeeze(sum(distNeighbor_ .* v1_trans, 3))'; % Local Transform
    deformedV = (deformedV_local - G0(:)') * R0' + T0(:)' + G0(:)'; % Global Transform
    deformedV(any(invalidNeibor, 1), :) = v0(:, any(invalidNeibor, 1))';
    
    %[idxNearest, dists2] = multiQueryKNNSearchImpl(pointCloud(gather(deformedV)), pntTar.Location, 1); %idxNearest
    if ~exist('indices', 'var')
        indices = knnsearch(gather(deformedV), pntTar.Location, 'K', 1);
    end
    
    Efit = (gather(deformedV(indices, :)) - pntTar.Location) ./ sqrt(size(pntTar.Location, 1));
    %Efit = sqrt(sum(Efit.^2, 2));
    %Efit = sqrt(dists2 ./ size(pntTar.Location, 1));
    %Efit (3, size(pntTar.Location, 1))
    
    Elap = gather((Lap * deformedV) ./ full(diag(Lap)));
    %Elap (3, n_verts)
    
    Erot1 = (pagemtimes(A0, 'transpose', A0, 'none') - reshape(eye(3), 3, 3, 1)) ./ sqrt(n_nodes);
    Erot2 = (pagedet(A0) - 1.0) ./ sqrt(n_nodes);
    %Erot1 (3, 3, n_nodes)
    %Erot2 (1, 1, n_nodes)
    
    g_node1 = reshape(g0(:, idxNodeNeighbor'), [size(g0, 1) size(idxNodeNeighbor')]);
    t_node1 = reshape(t0(:, idxNodeNeighbor'), [size(t0, 1) size(idxNodeNeighbor')]);
    v_node1 = reshape(g_node1 - g0, [size(g_node1, 1) 1 size(g_node1, 2:3)]);
    Esmooth = gather(sum(squeeze(pagemtimes(A0, v_node1)) + g0 + t0 - g_node1 - t_node1, 3)) ./ sqrt(3 * n_nodes);
    %Esmooth = gather(sqrt(sum((squeeze(pagemtimes(A0, v_node1)) + g0 + t0 - g_node1 - t_node1).^2, [1 3]) ./ (3 * n_nodes)));
    %Esmooth (3, n_nodes)
    
    F = [Efit(:); w_rot*Erot1(:); w_rot*Erot2(:); w_smooth*Esmooth(:); w_lap* Elap(:)];
    
    if nargout > 1
        jacobian_delta = 0.000001;
        %jacobian_delta = 0.1;
        
        Jfit = sparse(numel(Efit), numel(P0));
        Jrot1 = sparse(numel(Erot1), numel(P0));
        Jrot2 = sparse(numel(Erot2), numel(P0));
        Jsmooth = sparse(numel(Esmooth), numel(P0));
        Jlap = sparse(numel(Elap), numel(P0));
        
        J_ = [Jfit; Jrot1; Jrot2; Jsmooth; Jlap];
        
%             idx = 1:numel(P0);
%             for i = 1:length(idx)
%                 Pt = P0;
%                 Pt(idx(i)) = Pt(idx(i)) + jacobian_delta;
%                 if finds_nearests(idx(i))
%                     Fp = EfuncLocal(Pt, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, R_, T_, A_, t_);
%                 else
%                     Fp = EfuncLocal(Pt, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, R_, T_, A_, t_, indices);
%                 end
%                 J(:, idx(i)) = (Fp(:) - F(:)) / jacobian_delta;
%             end

            n_target = length(pntTar.Location);
            nA = 9 * n_nodes;
            
            parfor i = 1:(12 * n_nodes + 6)
                Pt = P0;
                Pt(i) = Pt(i) + jacobian_delta;
                if finds_nearests(i)
                    Fp = EfuncLocal(Pt, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, R_, T_, A_, t_);
                else
                    Fp = EfuncLocal(Pt, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, R_, T_, A_, t_, indices);
                end
                J_(:, i) = (Fp(:) - F(:)) / jacobian_delta;
                
                if ((i > 6) && (i <= (nA + 6)))
                    new_A0 = zeros(3, 3, n_nodes);
                    new_A0(i - 6) = 1;
                    new_A1 = reshape(new_A0(:, :, idxNeighbor'), [size(new_A0, [1 2]) size(idxNeighbor')]);
                    new_v1_trans = squeeze(pagemtimes(new_A1, v1));
                    new_deformedV_local = squeeze(sum(distNeighbor_ .* new_v1_trans, 3))';
                    
                    %Jfit(:, i) = reshape(gather(new_deformedV_local(indices,:) * R0'), [], 1);
                    %Jsmooth(:, i) = reshape(gather(sum(squeeze(pagemtimes(new_A0, v_node1)), 3)), [], 1);
                    Jlap(:, i) = reshape(gather(Lap * new_deformedV_local * R0'), [], 1);
                elseif (i > (nA + 6))
                    new_t0 = zeros(3, n_nodes);
                    new_t0(i - nA - 6) = 1;
                    new_t1 = reshape(new_t0(:, idxNeighbor'), [size(new_t0, 1) size(idxNeighbor')]);
                    new_deformedV_local = squeeze(sum(distNeighbor_ .* new_t1, 3))';
                    
                    %new_t_node1 = reshape(new_t0(:, idxNodeNeighbor'), [size(new_t0, 1) size(idxNodeNeighbor')]);
                    
                    %Jfit(:, i) = reshape(gather(new_deformedV_local(indices,:) * R0'), [], 1);
                    %Jsmooth(:, i) = reshape(sum(new_t0 - new_t_node1, 3), [], 1);
                    Jlap(:, i) = reshape(gather(Lap * new_deformedV_local * R0'), [], 1);
                end
            end
            
%             rot1_row =[1 2 3 4 7; 2 4 5 6 8; 3 6 7 8 9; 1 2 3 4 7; 2 4 5 6 8; 3 6 7 8 9; 1 2 3 4 7; 2 4 5 6 8; 3 6 7 8 9]';
%             rot1_rows = reshape(permute(repmat(rot1_row, [1 1 n_nodes]), [1 3 2]) + repmat(9 * ((1:n_nodes) - 1), [5 1 9]), [] ,1);
%             
%             rot1_cols = reshape(reshape(repmat((1:9) + 6, [5 n_nodes]), [], n_nodes) + repmat(9 * ((1:n_nodes) - 1), [45 1]), [], 1);
%             
%             A = reshape(permute(A0, [2 1 3]), [] ,size(A0, 3));
%             rot1_vals = [2 * A(1, :), A(2, :), A(3, :), A(2, :), A(3, :),...
%                     A(1, :), A(1, :), 2 * A(2, :), A(3, :), A(3, :),...
%                     A(1, :), A(2, :), A(1, :), A(2, :), 2 * A(3, :),...
%                     2 * A(4, :), A(5, :), A(6, :), A(5, :), A(4, :),...
%                     A(4, :), A(4, :), 2 * A(5, :), A(6, :), A(6, :),...
%                     A(4, :), A(5, :), A(4, :), A(5, :), 2 * A(6, :),...
%                     2 * A(7, :), A(8, :), A(9, :), A(8, :), A(7, :),...
%                     A(7, :), A(7, :), 2 * A(8, :), A(9, :), A(9, :),...
%                     A(7, :), A(8, :), A(7, :), A(8, :), 2 * A(9, :)]';
%                 
%             Jrot1 = sparse(rot1_rows, rot1_cols, rot1_vals, numel(Erot1), numel(P0));
%             
%             rot2_rows = reshape(repmat(1:n_nodes, [9 1]), [], 1);
%             
%             rot2_cols = reshape(reshape(repmat((1:9) + 6, [1 n_nodes]), [], n_nodes) + repmat(9 * ((1:n_nodes) - 1), [9 1]), [], 1);
%             
%             rot2_vals = [A(5, :) .* A(9, :) - A(6, :) .* A(8, :),...
%                          A(6, :) .* A(7, :) - A(4, :) .* A(9, :),...
%                          A(4, :) .* A(8, :) - A(5, :) .* A(7, :),...
%                          A(3, :) .* A(8, :) - A(2, :) .* A(9, :),...
%                          A(1, :) .* A(9, :) - A(3, :) .* A(7, :),...
%                          A(2, :) .* A(7, :) - A(1, :) .* A(8, :),...
%                          A(2, :) .* A(6, :) - A(3, :) .* A(5, :),...
%                          A(3, :) .* A(4, :) - A(1, :) .* A(6, :),...
%                          A(1, :) .* A(5, :) - A(2, :) .* A(4, :)];
%                      
%             Jrot2 = sparse(rot2_rows, rot2_cols, rot2_vals, numel(Erot2), numel(P0));
            
            
        Jfit_ = sparse(J_(1:(3 * n_target), 7:end) ./ sqrt(n_target));
        Jrot1_ = sparse(w_rot *  J_((1 + 3 * n_target):(3 * 3 * n_nodes + 3 * n_target), 7:end) ./ sqrt(n_nodes));
        Jrot2_ = sparse(w_rot *  J_((1 + 3 * 3 * n_nodes + 3 * n_target):(n_nodes + 3 * 3 * n_nodes + 3 * n_target), 7:end) ./ sqrt(n_nodes));
        Jsmooth_ = sparse( w_smooth * J_((1 + n_nodes + 3 * 3 * n_nodes + 3 * n_target):(3 * n_nodes + n_nodes + 3 * 3 * n_nodes + 3 * n_target), 7:end) ./ sqrt(3 * n_nodes));
        Jlap_ = sparse(w_lap * Jlap(:, 7:end) ./ reshape(repmat(full(diag(gather(Lap))), [1 3]), [], 1));
        J_7toEnd = [Jfit_; Jrot1_; Jrot2_; Jsmooth_; Jlap_];
        
        J_(1:(3 * n_target), 1:6) = J_(1:(3 * n_target), 1:6) ./ sqrt(n_target);
        J_((1 + 3 * n_nodes + n_nodes + 3 * 3 * n_nodes + 3 * n_target):end, 1:6) = w_lap * J_((1 + 3 * n_nodes + n_nodes + 3 * 3 * n_nodes + 3 * n_target):end, 1:6) ./ reshape(repmat(full(diag(gather(Lap))), [1 3]), [], 1);
        
        J = [sparse(J_(:, 1:6)), J_7toEnd];
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

function [R0, T0, A0, t0, finds_nearests] = replaceParam(R0, T0, A0, t0, R_, T_, A_, t_)
    finds_nearests = [];
    if isempty(R0)
        R0 = R_;
    else
        finds_nearests = [finds_nearests, zeros(1, numel(R0), 'logical')];
    end
    
    if isempty(T0)
        T0 = T_;
    else
        finds_nearests = [finds_nearests, zeros(1, numel(T0), 'logical')];
    end
    if isempty(A0)
        A0 = A_;
    else
        finds_nearests = [finds_nearests, zeros(1, numel(A0), 'logical')];
    end
    if isempty(t0)
        t0 = t_;
    else
        finds_nearests = [finds_nearests, zeros(1, numel(t0), 'logical')];
    end
end