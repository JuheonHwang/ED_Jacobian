function [P, hist] = levenbergMarquardtCG(fun, P0, varargin)
%LEVENBERGMARQUARDTCG  Levenberg-Marquardt whose damped normal equations are solved by (preconditioned) CG.
%   [P, hist] = edreg.solvers.levenbergMarquardtCG(fun, P0)
%   [P, hist] = edreg.solvers.levenbergMarquardtCG(fun, P0, Preconditioner="none")
%
%   Same outer loop, lambda schedule and accept/reject logic as
%   edreg.solvers.levenbergMarquardt (Variant "hjh2"); only the linear
%   solve  (J'J + lambda*diag(diag(J'J))) d = -J'F  is done iteratively with
%   conjugate gradient started from d = 0, so the factorisation is avoided
%   and every iteration costs only sparse matrix-vector products with the
%   sparse J'J.  hist.cgIterations records the inner iterations.
%
%   Preconditioner (option, default "jacobi")
%     "jacobi" : z = r ./ diag(A) with A the damped matrix; legacy
%                EDRegisterSparse_Show_iter_PCG_hjh.m (preconjgrad)
%     "none"   : plain CG; legacy EDRegisterSparse_Show_iter_CG_hjh.m
%   Legacy CG stopping rule kept as default: |r|_2 < 1e-10 (absolute) or
%   length(b) iterations.  Because the tolerance is absolute and tiny the
%   cap is usually what ends the inner loop; use CGTolerance / CGMaxIterations
%   to trade accuracy for speed (inexact Newton).
%
%   All other options and both outputs are those of
%   edreg.solvers.levenbergMarquardt (Variant, Damping, Lambda*, tolerances,
%   Display, Callback, ...); LinearSolver is fixed to "cg".
%   Legacy Damping modes of the PCG script: 'diag' (default), 'eye'
%   (= Damping="identity") and 'none'.
    p = inputParser;
    p.KeepUnmatched = true;
    addParameter(p, 'Preconditioner', "jacobi");
    parse(p, varargin{:});
    pre = string(p.Results.Preconditioner);
    passed = namedargs2cell(p.Unmatched);
    [P, hist] = edreg.solvers.levenbergMarquardt(fun, P0, passed{:}, ...
        LinearSolver="cg", Preconditioner=pre, SolverName="levenbergMarquardtCG");
end
