function [P, hist] = gaussNewton(fun, P0, opts)
%GAUSSNEWTON  Undamped Gauss-Newton with full steps for the ED registration energy.
%   [P, hist] = edreg.solvers.gaussNewton(fun, P0)
%   [P, hist] = edreg.solvers.gaussNewton(fun, P0, LinearSolver="lsq", MaxIterations=50)
%
%   Each iteration linearises F(P + d) ~ F + J d and takes the full step
%   d = argmin |F + J d|^2 (minimum-norm solution, since J has a null space:
%   the global rotation/translation are redundant with the per-node A and t):
%       "lsq"    d = lsqminnorm(J, -F)            legacy ..._Gauss_Newton_hjh_re.m
%       "normal" d = lsqminnorm(J'J, -J'F)        legacy ..._Gauss_Newton_hjh.m
%   The legacy files differ only in this line (and in the Jacobian code, which
%   is replaced here by edreg.residual).  "normal" forms the SPARSE product
%   J'J (never a dense matrix of our own making).  It is the default because
%   "lsq" diverged on the f10k body-scan pair (tutorials/t05_solvers.m: the
%   second full step already raised the energy from 198 to 982 and it then
%   blew up).  Probable cause: J is numerically rank deficient, lsqminnorm
%   truncates singular values below a tolerance relative to the largest one,
%   and for J itself that keeps directions with tiny singular values that the
%   squared singular values of J'J drop; the minimum-norm step then becomes
%   huge along them.  Both options stay available to reproduce the legacy files.
%   There is no damping and no line search, so the energy may rise
%   (legacy behaviour).  Use edreg.solvers.levenbergMarquardt for a
%   safeguarded method.
%   Legacy: EDRegisterSparse_Show_iter_Gauss_Newton_hjh.m, ..._hjh_re.m and
%   ..._Gauss_Newton_delta_hjh.m (the latter replaces the analytic J by a
%   parfor finite-difference J with fixed correspondences; any such J can be
%   used here by supplying it through fun (see edreg.numericJacobian).
%
%   Inputs
%     fun : handle, [F, J] = fun(P) (see edreg.solvers.gradientDescent);
%           correspondences are recomputed at every call
%     P0  : starting parameter vector
%   Options
%     LinearSolver      "normal" (default) | "lsq", see above
%     MaxIterations     iteration cap (default 100, legacy)
%     FunctionTolerance stop when |E - Eprev| < tol*(1+E)        (default 1e-6, legacy)
%     StepTolerance     stop when |d|_inf < tol*(1+|d|_inf)      (default 1e-3, legacy)
%     GradientTolerance stop when |2J'F|_inf < tol*(1+E)         (default 1e-2, legacy)
%                       Any of them can be disabled with 0.  The legacy code
%                       evaluated these on the residual VECTOR, which only
%                       worked by accident; here E = sum(F.^2) is used.
%     Display           "off" (default) | "iter" | "final"
%     Callback          callback(P, hist) after every iteration (time excluded)
%   Outputs
%     P    : final parameter vector
%     hist : struct with iteration, resnorm, time, stepNorm, gradNorm (column
%            vectors, entry 1 = starting point) and the scalars iterations,
%            funcCount, jacCount, exitReason.  If a step produces a
%            non-finite residual the solver stops and returns the last finite
%            iterate (exitReason "NonFiniteResidual").
    arguments
        fun (1,1) function_handle
        P0 (:,1) double
        opts.LinearSolver (1,1) string {mustBeMember(opts.LinearSolver, ["lsq" "normal"])} = "normal"
        opts.MaxIterations (1,1) double {mustBeInteger, mustBeNonnegative} = 100
        opts.FunctionTolerance (1,1) double {mustBeNonnegative} = 1e-6
        opts.StepTolerance (1,1) double {mustBeNonnegative} = 1e-3
        opts.GradientTolerance (1,1) double {mustBeNonnegative} = 1e-2
        opts.Display (1,1) string {mustBeMember(opts.Display, ["off" "iter" "final"])} = "off"
        opts.Callback = []
    end
    clk = tic;
    h = newHistory("gaussNewton");

    P = P0(:);
    [F, J, h] = evaluate(fun, P, true, h);
    E = sum(F.^2);
    assert(isfinite(E), 'edreg:solvers:nonfinite', 'The residual is not finite at the starting point.');
    g = 2 * (J' * F);
    h = record(h, clk, P, E, NaN, norm(g), [], opts.Display, opts.Callback);

    reason = "MaxIterations";
    for k = 1:opts.MaxIterations
        switch opts.LinearSolver
            case "lsq"
                step = lsqminnorm(J, -F);
            case "normal"
                step = lsqminnorm(J' * J, -(J' * F));
        end
        Pnew = P + step;
        [Fnew, Jnew, h] = evaluate(fun, Pnew, true, h);
        Enew = sum(Fnew.^2);
        if ~isfinite(Enew)
            reason = "NonFiniteResidual";
            break
        end
        Eprev = E;
        P = Pnew; F = Fnew; J = Jnew; E = Enew;
        g = 2 * (J' * F);
        h = record(h, clk, P, E, norm(step), norm(g), [], opts.Display, opts.Callback);
        stop = checkStop(E, Eprev, norm(step, inf), norm(g, inf), ...
            opts.FunctionTolerance, opts.StepTolerance, opts.GradientTolerance);
        if stop ~= ""
            reason = stop;
            break
        end
    end
    hist = finish(h, reason, opts.Display);
end
