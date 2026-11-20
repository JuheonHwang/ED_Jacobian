function [R, T, A, t, dR] = decodeParam(P, nNodes)
%DECODEPARAM  Unpack the ED parameter vector (inverse of edreg.encodeParam).
%   [R, T, A, t, dR] = edreg.decodeParam(P, nNodes)
%
%   Inputs
%     P      : (6 + 12*nNodes)-vector, layout [r(3); T(3); A(:); t(:)]
%     nNodes : number of ED nodes N
%   Outputs
%     R  : 3-by-3 rotation matrix from the rotation vector r = P(1:3)
%     T  : 3-by-1 translation
%     A  : 3-by-3-by-N affine matrices
%     t  : 3-by-N node translations
%     dR : 9-by-3, d vec(R) / d r (see edreg.rodrigues)
    arguments
        P (:,1) double
        nNodes (1,1) double {mustBeInteger, mustBePositive}
    end
    assert(numel(P) == 6 + 12*nNodes, 'edreg:decodeParam:size', ...
        'Expected %d parameters for %d nodes, got %d.', 6 + 12*nNodes, nNodes, numel(P));
    [R, dR] = edreg.rodrigues(P(1:3));
    T = P(4:6);
    nA = 9 * nNodes;
    A = reshape(P(7:6+nA), [3 3 nNodes]);
    t = reshape(P(7+nA:end), [3 nNodes]);
end
