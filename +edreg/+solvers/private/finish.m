function h = finish(h, reason, display)
%FINISH  Close the history: store the exit reason and the iteration count.
%   h = finish(h, reason, display)
    h.exitReason = string(reason);
    h.iterations = numel(h.iteration) - 1;
    h = rmfield(h, {'pausedTime', 'extraNames'});
    if display ~= "off"
        fprintf('%s: %s after %d iterations, resnorm %.6g, %.2f s\n', h.solver, ...
            h.exitReason, h.iterations, h.resnorm(end), h.time(end));
    end
end
