function [Vd, aux] = deform(P, model)
%DEFORM  Apply the embedded deformation to the source vertices.
%   [Vd, aux] = edreg.deform(P, model)
%
%   Inputs
%     P     : parameter vector [r; T; A(:); t(:)]
%     model : struct from edreg.buildModel
%   Outputs
%     Vd    : n-by-3 deformed vertices
%     aux   : intermediates reused by the Jacobian
%               .R, .T, .A, .t, .dR   decoded parameters
%               .U     n-by-3 local-transform result minus G0 (before global R)
%
%   For vertex v_i with neighbour nodes k (weights w_ik, node position g_k):
%       u_i  = sum_k w_ik ( A_k (v_i - g_k) + g_k + t_k ) - G0
%       vd_i = R u_i + T + G0
%   Vertices flagged model.invalidVertex keep their original position.
    n = model.nVerts;
    [R, T, A, t, dR] = edreg.decodeParam(P, model.nNodes);
    Vloc = zeros(n, 3);
    for m = 1:size(model.idxNeighbor, 2)
        k = model.idxNeighbor(:, m);
        gk = model.nodeV(k, :);                                    % n-by-3
        Ad = pagemtimes(A(:, :, k), reshape((model.V - gk)', [3 1 n]));
        Vloc = Vloc + model.wNeighbor(:, m) .* (reshape(Ad, 3, n)' + gk + t(:, k)');
    end
    U = Vloc - model.G0;
    Vd = U * R' + T' + model.G0;
    Vd(model.invalidVertex, :) = model.V(model.invalidVertex, :);
    aux = struct('R', R, 'T', T, 'A', A, 't', t, 'dR', dR, 'U', U);
end
