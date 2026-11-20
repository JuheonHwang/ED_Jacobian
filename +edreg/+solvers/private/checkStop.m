function reason = checkStop(E, Eprev, stepInf, gradInf, tolF, tolX, tolG)
%CHECKSTOP  Legacy-style termination tests; a tolerance of 0 disables a test.
%   reason = checkStop(E, Eprev, stepInf, gradInf, tolF, tolX, tolG)
%
%   E, Eprev : energies sum(F.^2) at the current and previous iterate
%   stepInf  : inf-norm of the last step
%   gradInf  : inf-norm of the gradient 2*J'*F at the current iterate
%   Tests (as in the legacy Gauss-Newton scripts, with E in place of the
%   residual vector they accidentally used):
%     FunctionTolerance : |E - Eprev| < tolF * (1 + E)
%     StepTolerance     : stepInf     < tolX * (1 + stepInf)
%     GradientTolerance : gradInf     < tolG * (1 + E)
%   Returns "" when no test fires, otherwise the name of the first that does.
    reason = "";
    if abs(E - Eprev) < tolF * (1 + E)
        reason = "FunctionTolerance";
    elseif stepInf < tolX * (1 + stepInf)
        reason = "StepTolerance";
    elseif gradInf < tolG * (1 + E)
        reason = "GradientTolerance";
    end
end
