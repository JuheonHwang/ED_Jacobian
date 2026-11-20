function [x, nIter] = conjugateGradient(A, b, M, tol, maxIter)
%CONJUGATEGRADIENT  (Preconditioned) conjugate gradient for A x = b, x0 = 0.
%   [x, nIter] = conjugateGradient(A, b, M, tol, maxIter)
%
%   Inputs
%     A       sparse symmetric positive (semi-)definite matrix
%     b       right-hand side
%     M       diagonal of the Jacobi preconditioner (the preconditioned
%             residual is z = r ./ M), or [] for plain CG
%     tol     absolute tolerance on the 2-norm of the (unpreconditioned)
%             residual r = b - A x, as in the legacy scripts (1e-10)
%     maxIter iteration cap (legacy: length(b))
%   Outputs
%     x      approximate solution
%     nIter  number of CG iterations performed
%
%   Standard Hestenes-Stiefel recurrences: alpha = (r'z)/(p'Ap),
%   x += alpha p, r -= alpha Ap, beta = (r'z)_new/(r'z)_old, p = z + beta p.
%   Stops early if the curvature p'Ap is not positive (singular direction).
    x = zeros(size(b));
    r = b;
    if isempty(M), z = r; else, z = r ./ M; end
    p = z;
    rz = r' * z;
    nIter = 0;
    if norm(r) < tol
        return
    end
    for k = 1:maxIter
        Ap = A * p;
        curvature = p' * Ap;
        if ~(curvature > 0)
            break
        end
        alpha = rz / curvature;
        x = x + alpha * p;
        r = r - alpha * Ap;
        nIter = k;
        if norm(r) < tol
            break
        end
        if isempty(M), z = r; else, z = r ./ M; end
        rzNew = r' * z;
        p = z + (rzNew / rz) * p;
        rz = rzNew;
    end
end
