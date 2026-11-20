function [P, hist] = gradientDescent(fun, P0, opts)
%GRADIENTDESCENT  Fixed-step gradient descent on the ED registration energy.
%   [P, hist] = edreg.solvers.gradientDescent(fun, P0)
%   [P, hist] = edreg.solvers.gradientDescent(fun, P0, StepSize=1e-3, MaxIterations=100)
%
%   Minimises E(P) = sum(F(P).^2) with the plain update
%       g = 2 J' F,      P <- P - StepSize * s .* g,
%   where s = RigidScale for the six global parameters (r, T) and s = 1 for
%   the per-node parameters (A, t).  The global parameters move every vertex
%   at once, so the legacy code damped their step by a factor 0.01.  No line
%   search, no accept/reject test: the energy can rise.
%   Legacy: EDRegisterSparse_Show_iter_gradient_hjh.m, last lines of
%   EfuncLocal (step_size = 0.001, 100 iterations, rigid step 0.01*step_size).
%
%   Inputs
%     fun : handle, [F, J] = fun(P) with F the residual vector and J its
%           sparse Jacobian, e.g. @(P) edreg.residual(P, model, target, w).
%           fun recomputes the nearest-neighbour correspondences at every
%           call (ICP style), so F and J are consistent at each P.
%     P0  : starting parameter vector
%   Options
%     StepSize          gradient step (default 1e-3, legacy)
%     RigidScale        extra factor on the step of the first six parameters
%                       (default 0.01, legacy)
%     MaxIterations     iteration cap (default 100, legacy)
%     FunctionTolerance stop when |E - Eprev| < tol*(1+E); 0 disables
%     StepTolerance     stop when |step|_inf < tol*(1+|step|_inf); 0 disables
%     GradientTolerance stop when |g|_inf < tol*(1+E); 0 disables
%                       (all three default to 0: the legacy loop ran a fixed
%                       number of iterations)
%     Display           "off" (default) | "iter" | "final"
%     Callback          callback(P, hist) called after every iteration
%                       (and once for the starting point), e.g. to draw meshes;
%                       its run time is excluded from hist.time
%   Outputs
%     P    : final parameter vector
%     hist : struct, see the common fields below
%       iteration, resnorm (= E), time (cumulative seconds), stepNorm and
%       gradNorm (2-norms) as column vectors, entry 1 = starting point;
%       iterations, funcCount, jacCount, exitReason (scalars).
    arguments
        fun (1,1) function_handle
        P0 (:,1) double
        opts.StepSize (1,1) double {mustBePositive} = 1e-3
        opts.RigidScale (1,1) double {mustBePositive} = 0.01
        opts.MaxIterations (1,1) double {mustBeInteger, mustBeNonnegative} = 100
        opts.FunctionTolerance (1,1) double {mustBeNonnegative} = 0
        opts.StepTolerance (1,1) double {mustBeNonnegative} = 0
        opts.GradientTolerance (1,1) double {mustBeNonnegative} = 0
        opts.Display (1,1) string {mustBeMember(opts.Display, ["off" "iter" "final"])} = "off"
        opts.Callback = []
    end
    clk = tic;
    h = newHistory("gradientDescent");
    scale = [opts.RigidScale * ones(6, 1); ones(numel(P0) - 6, 1)];

    P = P0(:);
    [F, J, h] = evaluate(fun, P, true, h);
    E = sum(F.^2);
    assert(isfinite(E), 'edreg:solvers:nonfinite', 'The residual is not finite at the starting point.');
    g = 2 * (J' * F);
    h = record(h, clk, P, E, NaN, norm(g), [], opts.Display, opts.Callback);

    reason = "MaxIterations";
    for k = 1:opts.MaxIterations
        step = -opts.StepSize * scale .* g;
        Pnew = P + step;
        [Fnew, Jnew, h] = evaluate(fun, Pnew, true, h);
        Enew = sum(Fnew.^2);
        if ~isfinite(Enew)
            reason = "NonFiniteResidual";
            break
        end
        Eprev = E;
        P = Pnew; E = Enew;
        g = 2 * (Jnew' * Fnew);
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
