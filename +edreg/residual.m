function [F, J, info] = residual(P, model, target, w, corr)
%RESIDUAL  ED registration residual vector and (optionally) its analytic Jacobian.
%   [F, J, info] = edreg.residual(P, model, target, w, corr)
%
%   Inputs
%     P      : parameter vector [r(3); T(3); A(:); t(:)] (see edreg.encodeParam)
%     model  : struct from edreg.buildModel
%     target : mesh struct with field V (nT-by-3)
%     w      : struct with fields rot, smooth, lap (see edreg.defaultWeights);
%              [] or missing fields use the defaults
%     corr   : optional nT-by-1 fixed correspondences (indices into the
%              deformed source); computed from the current Vd if omitted/empty
%   Outputs
%     F    : stacked residual
%              [ Efit; w.rot*Erot1; w.rot*Erot2; w.smooth*Esmooth; w.lap*Elap ]
%     J    : sparse Jacobian dF/dP (only computed when nargout > 1)
%     info : .rows.{fit,rot1,rot2,smooth,lap}  row index ranges of F / J
%            .cols.{R,T,A,t}                    column index ranges of P / J
%            .corr                              correspondences used
%            .Vd                                deformed vertices
%
%   Terms (N nodes, n vertices, nT target points, Vd = edreg.deform(P, model)):
%     Efit    = (Vd(corr,:) - targetV) / sqrt(nT)                        nT x 3
%     Erot1   = (A_k' A_k - I) / sqrt(N)           per node, 3 x 3       N x 9
%     Erot2   = (det(A_k) - 1) / sqrt(N)           per node              N x 1
%     Esmooth = sum_{j in nbr(k)} ( A_k (g_j - g_k) + g_k + t_k - g_j - t_j )
%               / sqrt(3N)                         per node              N x 3
%     Elap    = (Lap * Vd) ./ diag(Lap)                                  n x 3
%   Each block is flattened column-major into F.
%
%   Jacobian layout: every term function below returns a sparse matrix with
%   P-many columns, ordered [R(3) | T(3) | A(9N) | t(3N)], where the R block is
%   the derivative with respect to the rotation VECTOR r (via d vec(R)/d r).
    arguments
        P (:,1) double
        model struct
        target struct
        w = []
        corr = []
    end
    w = edreg.defaultWeights(w);
    n = model.nVerts;
    N = model.nNodes;
    nT = size(target.V, 1);

    [Vd, aux] = edreg.deform(P, model);
    if isempty(corr)
        corr = edreg.findCorrespondence(Vd, target.V);
    end
    corr = corr(:);
    A = aux.A;
    g = model.nodeV';                                   % 3-by-N

    Efit = (Vd(corr, :) - target.V) / sqrt(nT);
    Erot1 = (pagemtimes(A, 'transpose', A, 'none') - eye(3)) / sqrt(N);
    Erot2 = (pagedet(A) - 1) / sqrt(N);
    Esmooth = zeros(3, N);
    for m = 1:size(model.idxNodeNeighbor, 2)
        j = model.idxNodeNeighbor(:, m)';
        Ad = pagemtimes(A, reshape(g(:, j) - g, [3 1 N]));
        Esmooth = Esmooth + reshape(Ad, 3, N) + g + aux.t - g(:, j) - aux.t(:, j);
    end
    Esmooth = Esmooth / sqrt(3*N);
    Elap = (model.Lap * Vd) ./ model.LapDiag;

    F = [Efit(:); w.rot*Erot1(:); w.rot*Erot2(:); w.smooth*Esmooth(:); w.lap*Elap(:)];

    if nargout < 2
        return
    end

    nP = numel(P);
    dV = deformJacobian(model, aux);                    % 3n-by-nP
    J = [ jacFit(dV, corr, n, nT);
          w.rot    * jacRot1(A, nP);
          w.rot    * jacRot2(A, nP);
          w.smooth * jacSmooth(model, nP);
          w.lap    * jacLap(dV, model) ];

    if nargout > 2
        sizes = [3*nT, 9*N, N, 3*N, 3*n];
        ends = cumsum(sizes);
        starts = ends - sizes + 1;
        names = ["fit" "rot1" "rot2" "smooth" "lap"];
        for i = 1:numel(names)
            info.rows.(names(i)) = starts(i):ends(i);
        end
        info.cols = columnRanges(N);
        info.corr = corr;
        info.Vd = Vd;
    end
end

% ======================================================================
%  Column layout
% ======================================================================
function c = columnRanges(N)
    c.R = 1:3;
    c.T = 4:6;
    c.A = 6 + (1:9*N);
    c.t = 6 + 9*N + (1:3*N);
end

% ======================================================================
%  Shared block: d(deformed vertices) / d(R, T, A, t)
% ======================================================================
function dV = deformJacobian(model, aux)
% Jacobian of vec(Vd) (n-by-3, column-major: all x, then all y, then all z)
% with respect to P, size 3n-by-nP.  With u_i = sum_k w_ik(A_k(v_i-g_k)+g_k+t_k) - G0
% and vd_i = R u_i + T + G0, for output coordinate c:
%   d vd_ic / d R(c,j)   = u_ij                  (then chain rule dvec(R)/dr)
%   d vd_ic / d T(c)     = 1
%   d vd_ic / d A_k(a,b) = w_ik R(c,a) (v_i - g_k)_b        (k a neighbour of i)
%   d vd_ic / d t_k(b)   = w_ik R(c,b)                       (k a neighbour of i)
% Invalid vertices are pinned to their original position in edreg.deform, so
% all their rows are zero (this keeps J consistent with F).
    n = model.nVerts;
    N = model.nNodes;
    K = size(model.idxNeighbor, 2);
    R = aux.R;
    valid = find(~model.invalidVertex);
    nv = numel(valid);
    nP = 6 + 12*N;
    cOff = reshape(0:2, 1, 1, 1, 1, 3) * n;             % coordinate row offset

    % --- global rotation: S(row (c,i), vec-index c+3(j-1)) = u_ij, then S * dR
    [ii, jj, cc] = ndgrid(valid, 1:3, 1:3);
    S = sparse((cc(:) - 1)*n + ii(:), cc(:) + 3*(jj(:) - 1), aux.U(sub2ind([n 3], ii(:), jj(:))), 3*n, 9);
    JR = S * sparse(aux.dR);

    % --- global translation
    [ii, cc] = ndgrid(valid, 1:3);
    JT = sparse((cc(:) - 1)*n + ii(:), cc(:), 1, 3*n, 3);

    % --- node affine matrices A: array dims [vertex, neighbour slot, a, b, c]
    k = model.idxNeighbor(valid, :);                    % nv-by-K
    wk = model.wNeighbor(valid, :);
    d = reshape(model.V(valid, :), [nv 1 3]) - reshape(model.nodeV(k(:), :), [nv K 3]);   % v_i - g_k
    sz = [nv K 3 3 3];
    Rca = reshape(R', [1 1 3 1 3]);                     % R(c,a) at dims (a, c)
    valA = wk .* Rca .* reshape(d, [nv K 1 3]);         % nv x K x a x b x c
    rowA = reshape(valid, [nv 1 1 1 1]) + cOff;
    colA = 9*(k - 1) + reshape(1:3, 1, 1, 3) + 3*reshape(0:2, 1, 1, 1, 3);
    JA = sparse(expandTo(rowA, sz), expandTo(colA, sz), valA(:), 3*n, 9*N);

    % --- node translations t: dims [vertex, neighbour slot, b, c]
    szt = [nv K 3 3];
    valt = wk .* reshape(R', [1 1 3 3]);                % element (b,c) = R(c,b)
    rowt = reshape(valid, [nv 1 1 1]) + reshape(0:2, 1, 1, 1, 3) * n;
    colt = 3*(k - 1) + reshape(1:3, 1, 1, 3);
    Jt = sparse(expandTo(rowt, szt), expandTo(colt, szt), valt(:), 3*n, 3*N);

    dV = [JR, JT, JA, Jt];
    assert(size(dV, 2) == nP);
end

function x = expandTo(x, sz)
% Broadcast x to size sz and flatten to a column.
    x = x + zeros(sz);
    x = x(:);
end

% ======================================================================
%  Fit term:  Efit = (Vd(corr,:) - targetV) / sqrt(nT)
% ======================================================================
function Jf = jacFit(dV, corr, n, nT)
% Row (c, q) of Efit is the c-th coordinate of vertex corr(q), so its
% gradient is the corresponding row of dV, scaled by 1/sqrt(nT).
    rows = [corr; n + corr; 2*n + corr];
    Jf = dV(rows, :) / sqrt(nT);
end

% ======================================================================
%  Rotation term 1:  Erot1_k = (A_k' A_k - I) / sqrt(N)
% ======================================================================
function Jr = jacRot1(A, nP)
% (A'A)_pq = sum_r A(r,p) A(r,q), so
%   d (A'A)_pq / d A(r,p) += A(r,q)   and   d (A'A)_pq / d A(r,q) += A(r,p)
% (both contributions add up when p == q, giving 2 A(r,p)).
% Row of entry (p,q) of node k: 9(k-1) + p + 3(q-1); column of A_k(r,s): 6 + 9(k-1) + r + 3(s-1).
    N = size(A, 3);
    sz = [N 3 3 3];                                     % dims (k, p, q, r)
    Ap = permute(A, [3 1 2]);                           % Ap(k, r, c) = A(r, c, k)
    v1 = permute(Ap, [1 4 3 2]);                        % A(r,q,k) at dims (k,.,q,r)
    v2 = permute(Ap, [1 3 4 2]);                        % A(r,p,k) at dims (k,p,.,r)
    kOff = reshape(0:N-1, N, 1);
    rows = 9*kOff + reshape(1:3, 1, 3) + 3*reshape(0:2, 1, 1, 3);
    col1 = 6 + 9*kOff + reshape(1:3, 1, 1, 1, 3) + 3*reshape(0:2, 1, 3);   % A(r,p)
    col2 = 6 + 9*kOff + reshape(1:3, 1, 1, 1, 3) + 3*reshape(0:2, 1, 1, 3); % A(r,q)
    Jr = sparse([expandTo(rows, sz); expandTo(rows, sz)], ...
                [expandTo(col1, sz); expandTo(col2, sz)], ...
                [expandTo(v1, sz); expandTo(v2, sz)], 9*N, nP) / sqrt(N);
end

% ======================================================================
%  Rotation term 2:  Erot2_k = (det(A_k) - 1) / sqrt(N)
% ======================================================================
function Jr = jacRot2(A, nP)
% d det(A) / d A = cofactor matrix; its columns are a2 x a3, a3 x a1, a1 x a2
% for A = [a1 a2 a3].
    N = size(A, 3);
    a1 = reshape(A(:, 1, :), 3, N);
    a2 = reshape(A(:, 2, :), 3, N);
    a3 = reshape(A(:, 3, :), 3, N);
    C = [cross(a2, a3, 1); cross(a3, a1, 1); cross(a1, a2, 1)];        % 9-by-N, col-major
    rows = repmat(1:N, 9, 1);
    cols = 6 + 9*(0:N-1) + (1:9)';
    Jr = sparse(rows(:), cols(:), C(:), N, nP) / sqrt(N);
end

% ======================================================================
%  Smoothness term
% ======================================================================
function Js = jacSmooth(model, nP)
% Esmooth_k = sum_{j in nbr(k)} (A_k (g_j - g_k) + g_k + t_k - g_j - t_j) / sqrt(3N)
%  => d/dA_k(a,b) of component a = s_k(b),  s_k = sum_j (g_j - g_k)
%     d/dt_k = Kn * I,   d/dt_j = -I   for every neighbour slot j of k
% (when the node itself is among its neighbours, as buildModel guarantees,
% its slot contributes +1 and -1 to t_k, leaving the net Kn - 1).
    N = model.nNodes;
    nb = model.idxNodeNeighbor;
    Kn = size(nb, 2);
    gk = model.nodeV;
    s = squeeze(sum(reshape(gk(nb(:), :), [N Kn 3]), 2)) - Kn*gk;       % N-by-3

    [k, a, b] = ndgrid(1:N, 1:3, 1:3);
    rowsA = 3*(k(:) - 1) + a(:);
    colsA = 6 + 9*(k(:) - 1) + a(:) + 3*(b(:) - 1);
    valsA = s(sub2ind([N 3], k(:), b(:)));

    [k, c] = ndgrid(1:N, 1:3);
    rowsT = 3*(k(:) - 1) + c(:);
    colsSelf = 6 + 9*N + 3*(k(:) - 1) + c(:);
    rI = []; cI = []; vI = [];
    for m = 1:Kn
        j = nb(:, m);
        colsNb = 6 + 9*N + 3*(j(k(:)) - 1) + c(:);
        rI = [rI; rowsT; rowsT]; %#ok<AGROW>
        cI = [cI; colsSelf; colsNb]; %#ok<AGROW>
        vI = [vI; ones(numel(rowsT), 1); -ones(numel(rowsT), 1)]; %#ok<AGROW>
    end
    Js = sparse([rowsA; rI], [colsA; cI], [valsA; vI], 3*N, nP) / sqrt(3*N);
end

% ======================================================================
%  Laplacian term:  Elap = diag(Lap)^-1 * Lap * Vd
% ======================================================================
function Jl = jacLap(dV, model)
% Linear in Vd, so Jlap = blkdiag(Lap,Lap,Lap) * dV, rows scaled by 1/diag(Lap).
% This includes the global R block (Lap kills T only where no vertex is
% pinned, so the T block is computed exactly as well, not assumed zero).
    n = model.nVerts;
    L3 = kron(speye(3), model.Lap);
    Jl = spdiags(1 ./ repmat(model.LapDiag, 3, 1), 0, 3*n, 3*n) * (L3 * dV);
end

function D = pagedet(X)
% Determinant of every 3-by-3 page of X, returned as an N-by-1 vector.
    D = X(1,1,:).*X(2,2,:).*X(3,3,:) + X(1,2,:).*X(2,3,:).*X(3,1,:) + X(1,3,:).*X(2,1,:).*X(3,2,:) ...
      - X(1,3,:).*X(2,2,:).*X(3,1,:) - X(1,2,:).*X(2,1,:).*X(3,3,:) - X(1,1,:).*X(2,3,:).*X(3,2,:);
    D = D(:);
end
