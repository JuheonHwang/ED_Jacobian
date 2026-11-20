function h = record(h, clk, P, E, stepNorm, gradNorm, extras, display, callback)
%RECORD  Append one iteration to the history, then print and call the callback.
%   h = record(h, clk, P, E, stepNorm, gradNorm, extras, display, callback)
%
%   Inputs
%     h        history from newHistory
%     clk      tic id taken at the start of the solver
%     P        current parameter vector (passed to the callback)
%     E        energy sum(F.^2) at P
%     stepNorm 2-norm of the step that led to P (NaN at iteration 0)
%     gradNorm 2-norm of the energy gradient 2*J'*F at P
%     extras   struct with one field per extra history column, or [] for NaN
%     display  "off" | "iter" | "final"
%     callback function handle callback(P, h) or []
%   The stored time is the wall-clock time since clk minus the time spent in
%   earlier callbacks, so visualisation does not distort timing comparisons.
    k = numel(h.iteration);
    h.iteration(end+1, 1) = k;
    h.resnorm(end+1, 1) = E;
    h.time(end+1, 1) = toc(clk) - h.pausedTime;
    h.stepNorm(end+1, 1) = stepNorm;
    h.gradNorm(end+1, 1) = gradNorm;
    for f = h.extraNames
        if isempty(extras)
            h.(f)(end+1, 1) = NaN;
        else
            h.(f)(end+1, 1) = extras.(f);
        end
    end
    if display == "iter"
        if k == 0
            fprintf('%-18s %5s %14s %12s %12s', h.solver, 'iter', 'resnorm', '|step|', '|grad|');
            fprintf(' %12s', h.extraNames);
            fprintf('\n');
        end
        fprintf('%-18s %5d %14.6g %12.4g %12.4g', '', k, E, stepNorm, gradNorm);
        for f = h.extraNames
            fprintf(' %12.4g', h.(f)(end));
        end
        fprintf('\n');
    end
    if ~isempty(callback)
        tCb = tic;
        callback(P, h);
        h.pausedTime = h.pausedTime + toc(tCb);
    end
end
