function [F, J, h] = evaluate(fun, P, needJacobian, h)
%EVALUATE  Call the user function and count the evaluations.
%   [F, J, h] = evaluate(fun, P, needJacobian, h)
%
%   With needJacobian = false, fun is called with ONE output so that
%   edreg.residual skips assembling J (trial points of a line search or of
%   an LM accept/reject test only need the residual).  F is returned as a
%   column vector.  fun recomputes the nearest-neighbour correspondences
%   at P itself, so F and J always belong to the same, self-consistent
%   correspondences.
    if needJacobian
        [F, J] = fun(P);
        h.jacCount = h.jacCount + 1;
    else
        F = fun(P);
        J = [];
    end
    h.funcCount = h.funcCount + 1;
    F = F(:);
end
