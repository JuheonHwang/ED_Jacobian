function model = buildModel(V, F, opts)
%BUILDMODEL  Build the embedded-deformation (ED) model of a source mesh.
%   model = edreg.buildModel(V, F)
%   model = edreg.buildModel(V, F, SampleRate=0.01, NumVertexNeighbors=4, ...
%                            NumNodeNeighbors=6, Seed=42)
%
%   Inputs
%     V : n-by-3 source vertices
%     F : m-by-3 triangle indices
%   Options
%     SampleRate         fraction of vertices used as ED nodes (N = round(n*rate))
%     NumVertexNeighbors K, number of nodes that move each vertex
%     NumNodeNeighbors   number of neighbours per node in the smoothness term,
%                        INCLUDING the node itself
%     Seed               if given, nodes are sampled with a private random stream
%                        (the global rng is untouched); otherwise global rng is used
%
%   Node sampling: nodes are drawn with randperm(n, N). On scans with floating
%   fragments a node can land in a small connected component. A node is "bad"
%   if its component (conncomp of the mesh graph) holds fewer than
%   NumNodeNeighbors of the sampled nodes. Bad nodes are replaced, in place,
%   by further draws from the same random stream, restricted to vertices of
%   components that already hold at least NumNodeNeighbors good nodes (the
%   largest component if there is none) and not yet used as nodes. Draws that
%   needed no replacement are identical to plain randperm(n, N), so seeds that
%   worked before give the same nodes. Components without any node remain
%   unreachable: their vertices are flagged invalidVertex and pinned.
%
%   Output: struct with fields
%     V, F            source mesh
%     nVerts, nNodes  n and N
%     nodeIdx         N-by-1 vertex index of every ED node
%     nodeV           N-by-3 node positions (g_k = V(nodeIdx(k),:))
%     G0              1-by-3 centroid of V (centre of the global rotation)
%     idxNeighbor     n-by-K node indices influencing each vertex
%     wNeighbor       n-by-K weights (rows sum to 1 for valid vertices)
%     nReplacedNodes  number of sampled nodes replaced by the rule above
%     invalidVertex   n-by-1 logical; vertices whose weights are undefined
%                     (NaN, e.g. K+1-th node at distance 0 or unreachable).
%                     They are kept at their original position by edreg.deform.
%     idxNodeNeighbor N-by-Kn node neighbours; column 1 is the node itself
%     Lap             n-by-n sparse graph Laplacian (edge-length weighted)
%     LapDiag         n-by-1 diagonal of Lap
%     options         the options used
%
%   Distances are geodesic (shortest paths along mesh edges, weighted by edge
%   length).  The weight of node k for a vertex with K+1 nearest nodes at
%   distances d_1..d_{K+1} is proportional to 1 - d_k/d_{K+1}.
    arguments
        V (:,3) double
        F (:,3) double
        opts.SampleRate (1,1) double {mustBePositive, mustBeLessThanOrEqual(opts.SampleRate, 1)} = 0.01
        opts.NumVertexNeighbors (1,1) double {mustBeInteger, mustBePositive} = 4
        opts.NumNodeNeighbors (1,1) double {mustBeInteger, mustBeGreaterThan(opts.NumNodeNeighbors, 1)} = 6
        opts.Seed double {mustBeScalarOrEmpty} = []
    end
    n = size(V, 1);
    nNodes = round(n * opts.SampleRate);
    K = opts.NumVertexNeighbors;
    Kn = opts.NumNodeNeighbors;
    assert(nNodes >= Kn && nNodes > K, 'edreg:buildModel:tooFewNodes', ...
        ['Only %d ED nodes; need more than NumVertexNeighbors (%d) and at least ' ...
         'NumNodeNeighbors (%d). Increase SampleRate.'], nNodes, K, Kn);

    % --- mesh graph with edge-length weights and its Laplacian
    s = [F(:, 1); F(:, 2); F(:, 3)];
    e = [F(:, 2); F(:, 3); F(:, 1)];
    len = sqrt(sum((V(s, :) - V(e, :)).^2, 2));
    G = simplify(graph(s, e, len, n), 'first');   % each undirected edge once
    Lap = laplacian(G);
    lapDiag = full(diag(Lap));
    assert(all(lapDiag > 0), 'edreg:buildModel:isolatedVertex', ...
        'The mesh has vertices without edges (zero Laplacian diagonal).');

    % --- node sampling (see the help text for the replacement rule)
    if isempty(opts.Seed)
        stream = RandStream.getGlobalStream;
    else
        stream = RandStream('twister', Seed=opts.Seed);
    end
    nodeIdx = randperm(stream, n, nNodes)';
    comp = conncomp(G)';
    nComp = max(comp);
    perComp = accumarray(comp(nodeIdx), 1, [nComp 1]);
    bad = perComp(comp(nodeIdx)) < Kn;
    nReplaced = nnz(bad);
    if nReplaced > 0
        goodComps = find(perComp >= Kn);
        if isempty(goodComps)
            [~, goodComps] = max(accumarray(comp, 1, [nComp 1]));
        end
        eligible = find(ismember(comp, goodComps));
        eligible = setdiff(eligible, nodeIdx(~bad));
        assert(numel(eligible) >= nReplaced, 'edreg:buildModel:noReplacement', ...
            'Not enough vertices in well-populated components to replace %d nodes.', nReplaced);
        nodeIdx(bad) = eligible(randperm(stream, numel(eligible), nReplaced));
    end

    % --- geodesic distances from every node to every vertex (N-by-n)
    Dn = distances(G, nodeIdx);

    % vertex -> nearest nodes (K+1 needed for the weight normalisation)
    [dist, idx] = mink(Dn', K + 1, 2);                  % n-by-(K+1)
    w = 1 - dist ./ dist(:, K + 1);
    idxNeighbor = idx(:, 1:K);
    w = w(:, 1:K);
    w = w ./ sum(w, 2);
    invalid = any(~isfinite(w), 2);
    w(invalid, :) = 0;
    idxNeighbor(invalid, :) = 1;                         % any valid index; unused

    % node -> neighbouring nodes, self first (the smoothness Jacobian
    % documentation and tests rely on this invariant)
    Dnn = Dn(:, nodeIdx);
    [dnn, inn] = mink(Dnn, Kn + 1, 2);
    idxNodeNeighbor = zeros(nNodes, Kn);
    for k = 1:nNodes
        others = inn(k, inn(k, :) ~= k);
        idxNodeNeighbor(k, :) = [k, others(1:Kn - 1)];
    end
    assert(all(isfinite(dnn(:, 1:Kn)), 'all'), 'edreg:buildModel:disconnected', ...
        ['Some ED nodes have fewer than NumNodeNeighbors reachable neighbours even after node ' ...
         'replacement; the mesh has too few vertices in its connected components.']);
    assert(all(idxNodeNeighbor(:, 1) == (1:nNodes)'));

    model.V = V;
    model.F = F;
    model.nVerts = n;
    model.nNodes = nNodes;
    model.nodeIdx = nodeIdx;
    model.nReplacedNodes = nReplaced;
    model.nodeV = V(nodeIdx, :);
    model.G0 = mean(V, 1);
    model.idxNeighbor = idxNeighbor;
    model.wNeighbor = w;
    model.invalidVertex = invalid;
    model.idxNodeNeighbor = idxNodeNeighbor;
    model.Lap = Lap;
    model.LapDiag = lapDiag;
    model.options = opts;
end
