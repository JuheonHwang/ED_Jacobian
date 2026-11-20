function P = encodeParam(R, T, A, t)
%ENCODEPARAM  Pack ED parameters into one column vector.
%   P = edreg.encodeParam(R, T, A, t)
%
%   Inputs
%     R : 3-by-3 global rotation matrix
%     T : 3-vector global translation
%     A : 3-by-3-by-N per-node affine matrices
%     t : 3-by-N per-node translations
%   Output
%     P : (6 + 12N)-by-1,  P = [r(3); T(3); A(:); t(:)]  with r the rotation
%         vector of R, A(:) column-major per node, t(:) node by node.
    arguments
        R (3,3) double
        T (:,1) double
        A (3,3,:) double
        t (3,:) double
    end
    assert(numel(T) == 3 && size(t, 2) == size(A, 3), 'edreg:encodeParam:size', ...
        'T must have 3 entries and t must be 3-by-N with N = size(A,3).');
    P = [edreg.rotationToVector(R); T(:); A(:); t(:)];
end
