function h = newHistory(solverName, extraNames)
%NEWHISTORY  Empty history struct shared by all edreg.solvers.
%   h = newHistory(solverName, extraNames)
%
%   Per-iteration fields (column vectors, entry 1 is the starting point,
%   iteration 0, entry k+1 the state after k steps):
%     iteration, resnorm, time, stepNorm, gradNorm
%   plus one column per name in extraNames (method-specific, NaN at iteration 0).
%   The scalar bookkeeping fields funcCount / jacCount count calls of fun with
%   one / two outputs; pausedTime accumulates time spent in Callback and is
%   removed again by finish.
    arguments
        solverName (1,1) string
        extraNames (1,:) string = string.empty
    end
    h = struct('solver', solverName, 'iteration', zeros(0, 1), 'resnorm', zeros(0, 1), ...
        'time', zeros(0, 1), 'stepNorm', zeros(0, 1), 'gradNorm', zeros(0, 1));
    for f = extraNames
        h.(f) = zeros(0, 1);
    end
    h.extraNames = extraNames;
    h.funcCount = 0;
    h.jacCount = 0;
    h.pausedTime = 0;
    h.exitReason = "";
end
