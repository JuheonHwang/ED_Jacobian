function [P, hist] = levenbergMarquardt(fun, P0, opts)
%LEVENBERGMARQUARDT  Levenberg-Marquardt with the legacy lambda schedule for ED registration.
%   [P, hist] = edreg.solvers.levenbergMarquardt(fun, P0)
%   [P, hist] = edreg.solvers.levenbergMarquardt(fun, P0, Variant="hjh1", LinearSolver="cg")
%
%   Each iteration solves the damped normal equations
%       (J'J + lambda*D) d = -J'F,       D = diag(diag(J'J))   (Marquardt scaling)
%   and tests the trial point P + d.  J'J is a SPARSE matrix; the legacy
%   dense inv(...) is replaced by a sparse Cholesky solve (or CG, see
%   LinearSolver).  Energies are compared as E = sum(F.^2) where both the
%   current and the trial residual come from fun, i.e. each with its own
%   freshly computed nearest-neighbour correspondences (consistent ICP
%   energies; the accepted point is re-evaluated, never compared against a
%   stale value).
%
%   Variant "hjh2" (default; legacy EDRegisterSparse_Show_iter_LM_hjh_2.m)
%     lambda starts at 0.01.  Repeat: solve with lambda, evaluate the trial.
%       E_trial < E : accept the step.  If this was the first trial of the
%                     iteration, lambda <- lambda / 10 (afterwards); a
%                     success after at least one rejection keeps the
%                     increased lambda.
%       otherwise   : reject, lambda <- 10*lambda; if lambda > 1e5 give up
%                     (exitReason "LambdaExceeded", P stays at the last
%                     accepted point, as in the legacy loop).
%     The energy therefore never increases.
%   Variant "hjh1" (legacy EDRegisterSparse_Show_iter_LM_hjh.m)
%     lambda starts at 0.1 and at most 3 trials are made per iteration:
%       trial 1 with E_trial < E : accept, lambda <- lambda / 10, next iteration.
%       otherwise lambda <- 10*lambda (reset to 100 once it reaches 1000) and
%       the next trial is made.  A success on trial 2 or 3 does NOT stop the
%       loop, and whatever the third trial gives is taken even if the energy
%       rose.  This legacy logic is reproduced for comparison only: it can
%       accept uphill steps.  There is no termination other than MaxIterations
%       (legacy: 30).
%
%   Inputs
%     fun : handle, [F, J] = fun(P); it is called with ONE output for trial
%           points (so J is not assembled for rejected steps) and with two
%           outputs at accepted points.
%     P0  : starting parameter vector
%   Options
%     Variant           "hjh2" (default) | "hjh1", see above
%     LinearSolver      "direct" (default, sparse Cholesky; lsqminnorm if the
%                       matrix is only semi-definite) | "cg" (conjugate
%                       gradient, see edreg.solvers.levenbergMarquardtCG)
%     Damping           "diag" (default, lambda*diag(diag(J'J)), legacy) |
%                       "identity" (lambda*I) | "none" (lambda is ignored in
%                       the system but the schedule still runs; legacy
%                       'none' mode of ..._PCG_hjh.m)
%     Preconditioner    CG only: "none" (default, plain CG as ..._CG_hjh.m) |
%                       "jacobi" (diag of the damped matrix, ..._PCG_hjh.m)
%     CGTolerance       CG: absolute residual tolerance (default 1e-10, legacy)
%     CGMaxIterations   CG: iteration cap (default: number of parameters, legacy)
%     LambdaInit        starting lambda (default 0.01 for hjh2, 0.1 for hjh1)
%     LambdaIncrease    factor on rejection (default 10)
%     LambdaDecrease    divisor on acceptance (default 10)
%     MaxLambda         hjh2: give up above this lambda (default 1e5)
%     MaxIterations     accepted steps (default 100 for hjh2, 30 for hjh1; [] = variant default)
%     FunctionTolerance, StepTolerance, GradientTolerance
%                       as in edreg.solvers.gaussNewton but all disabled (0)
%                       by default: the legacy LM scripts had the same tests
%                       commented out
%     Display           "off" (default) | "iter" | "final"
%     Callback          callback(P, hist) after every iteration (time excluded)
%   Outputs
%     P    : final parameter vector
%     hist : struct with iteration, resnorm, time, stepNorm, gradNorm and the
%            method-specific columns
%              lambda        lambda used for the step taken (NaN at iteration 0)
%              nTrials       trial points evaluated in that iteration
%              cgIterations  inner CG iterations (NaN for the direct solver)
%            plus iterations, funcCount, jacCount, exitReason.
    arguments
        fun (1,1) function_handle
        P0 (:,1) double
        opts.Variant (1,1) string {mustBeMember(opts.Variant, ["hjh2" "hjh1"])} = "hjh2"
        opts.LinearSolver (1,1) string {mustBeMember(opts.LinearSolver, ["direct" "cg"])} = "direct"
        opts.Damping (1,1) string {mustBeMember(opts.Damping, ["diag" "identity" "none"])} = "diag"
        opts.Preconditioner (1,1) string {mustBeMember(opts.Preconditioner, ["none" "jacobi"])} = "none"
        opts.CGTolerance (1,1) double {mustBePositive} = 1e-10
        opts.CGMaxIterations double {mustBeScalarOrEmpty} = []
        opts.LambdaInit double {mustBeScalarOrEmpty} = []
        opts.LambdaIncrease (1,1) double {mustBeGreaterThan(opts.LambdaIncrease, 1)} = 10
        opts.LambdaDecrease (1,1) double {mustBeGreaterThan(opts.LambdaDecrease, 1)} = 10
        opts.MaxLambda (1,1) double {mustBePositive} = 1e5
        opts.MaxIterations double {mustBeScalarOrEmpty} = []
        opts.FunctionTolerance (1,1) double {mustBeNonnegative} = 0
        opts.StepTolerance (1,1) double {mustBeNonnegative} = 0
        opts.GradientTolerance (1,1) double {mustBeNonnegative} = 0
        opts.Display (1,1) string {mustBeMember(opts.Display, ["off" "iter" "final"])} = "off"
        opts.Callback = []
        opts.SolverName (1,1) string = "levenbergMarquardt"   % internal: label used by levenbergMarquardtCG
    end
    legacy1 = opts.Variant == "hjh1";
    lambda = opts.LambdaInit;
    maxIter = opts.MaxIterations;
    if isempty(lambda), lambda = 0.1 * legacy1 + 0.01 * ~legacy1; end
    if isempty(maxIter), maxIter = 30 * legacy1 + 100 * ~legacy1; end
    nP = numel(P0);
    cgMax = opts.CGMaxIterations;
    if isempty(cgMax), cgMax = nP; end

    clk = tic;
    h = newHistory(opts.SolverName, ["lambda" "nTrials" "cgIterations"]);
    P = P0(:);
    [F, J, h] = evaluate(fun, P, true, h);
    E = sum(F.^2);
    assert(isfinite(E), 'edreg:solvers:nonfinite', 'The residual is not finite at the starting point.');
    g = 2 * (J' * F);
    h = record(h, clk, P, E, NaN, norm(g), [], opts.Display, opts.Callback);

    reason = "MaxIterations";
    for k = 1:maxIter
        JtJ = J' * J;
        b = J' * F;
        if legacy1
            [Pnew, Enew, lambda, lambdaUsed, nTrials, cgIt, h] = stepHjh1(fun, P, E, JtJ, b, lambda, opts, cgMax, h);
            accepted = true;
        else
            [Pnew, Enew, lambda, lambdaUsed, nTrials, cgIt, accepted, h] = stepHjh2(fun, P, E, JtJ, b, lambda, opts, cgMax, h);
        end
        if ~accepted
            reason = "LambdaExceeded";
            break
        end
        step = Pnew - P;
        Eprev = E;
        P = Pnew;
        [F, J, h] = evaluate(fun, P, true, h);
        E = sum(F.^2);
        if ~isfinite(E)
            reason = "NonFiniteResidual";
            break
        end
        assert(abs(E - Enew) <= 1e-8 * (1 + E), 'edreg:solvers:inconsistentResidual', ...
            'fun returned different residuals for the same P; correspondences must be deterministic.');
        g = 2 * (J' * F);
        h = record(h, clk, P, E, norm(step), norm(g), ...
            struct('lambda', lambdaUsed, 'nTrials', nTrials, 'cgIterations', cgIt), ...
            opts.Display, opts.Callback);
        stop = checkStop(E, Eprev, norm(step, inf), norm(g, inf), ...
            opts.FunctionTolerance, opts.StepTolerance, opts.GradientTolerance);
        if stop ~= ""
            reason = stop;
            break
        end
    end
    hist = finish(h, reason, opts.Display);
end

% ======================================================================
function [Pnew, Enew, lambda, lambdaUsed, nTrials, cgIt, accepted, h] = stepHjh2(fun, P, E, JtJ, b, lambda, opts, cgMax, h)
% Legacy LM_hjh_2: retry with 10x lambda until the energy drops; lambda is
% divided by 10 only when the very first trial succeeded.
    firstTrial = true;
    nTrials = 0;
    accepted = false;
    Pnew = P; Enew = E; lambdaUsed = lambda; cgIt = NaN;
    while true
        nTrials = nTrials + 1;
        [d, cgIt] = solveDamped(JtJ, b, lambda, opts, cgMax);
        [Ft, ~, h] = evaluate(fun, P + d, false, h);
        Et = sum(Ft.^2);
        if Et < E
            accepted = true;
            Pnew = P + d; Enew = Et; lambdaUsed = lambda;
            if firstTrial
                lambda = lambda / opts.LambdaDecrease;
            end
            return
        end
        firstTrial = false;
        lambda = lambda * opts.LambdaIncrease;
        if lambda > opts.MaxLambda
            return
        end
    end
end

function [Pnew, Enew, lambda, lambdaUsed, nTrials, cgIt, h] = stepHjh1(fun, P, E, JtJ, b, lambda, opts, cgMax, h)
% Legacy LM_hjh: exactly three trials; only a success on trial 1 ends the
% loop early; the last trial is taken whether or not it lowered the energy.
    nTrials = 0;
    for i = 1:3
        nTrials = i;
        lambdaUsed = lambda;
        [d, cgIt] = solveDamped(JtJ, b, lambda, opts, cgMax);
        Pnew = P + d;
        [Ft, ~, h] = evaluate(fun, Pnew, false, h);
        Enew = sum(Ft.^2);
        if Enew < E && i == 1
            lambda = lambda / opts.LambdaDecrease;
            return
        end
        lambda = lambda * opts.LambdaIncrease;
        if lambda >= 1000
            lambda = 100;
        end
    end
end

function [d, cgIt] = solveDamped(JtJ, b, lambda, opts, cgMax)
% Solve (J'J + lambda*D) d = -b with D from opts.Damping.
    switch opts.Damping
        case "diag"
            A = JtJ + lambda * spdiags(diag(JtJ), 0, size(JtJ, 1), size(JtJ, 1));
        case "identity"
            A = JtJ + lambda * speye(size(JtJ, 1));
        case "none"
            A = JtJ;
    end
    cgIt = NaN;
    switch opts.LinearSolver
        case "direct"
            d = solveNormal(A, -b);
        case "cg"
            M = [];
            if opts.Preconditioner == "jacobi"
                M = diag(A);
                M(M <= 0) = 1;
            end
            [d, cgIt] = conjugateGradient(A, -b, M, opts.CGTolerance, cgMax);
    end
end
