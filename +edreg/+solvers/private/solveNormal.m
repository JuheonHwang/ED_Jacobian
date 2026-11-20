function x = solveNormal(A, b)
%SOLVENORMAL  Solve the sparse symmetric system A x = b without densifying A.
%   x = solveNormal(A, b)
%
%   A is a (damped) normal-equation matrix J'J + lambda*D.  If it is
%   positive definite a sparse Cholesky factorisation is used (the fast path
%   for every damped system).  An undamped J'J of the ED problem is only
%   positive SEMI-definite (global rotation/translation are redundant with
%   the per-node A and t), in which case the minimum-norm solution is
%   returned by lsqminnorm, as the legacy Gauss-Newton script did.
    [R, flag, s] = chol(A, 'lower', 'vector');
    if flag == 0
        y = R \ b(s);
        x = zeros(size(b));
        x(s) = R' \ y;
    else
        x = lsqminnorm(A, b);
    end
end
