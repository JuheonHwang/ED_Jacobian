function [P, out] = register(model, target, opts)
%REGISTER  Non-rigid ED registration of the source model onto a target mesh.
%   [P, out] = edreg.register(model, target)
%   [P, out] = edreg.register(model, target, UseJacobian=true, MaxIterations=30)
%
%   Minimises the stacked residual of edreg.residual with lsqnonlin
%   (trust-region-reflective).  Correspondences are recomputed at every
%   function evaluation, as in the legacy code.
%
%   Inputs
%     model, target : edreg.buildModel result and a mesh struct with V (and F)
%   Options
%     Weights        struct with rot, smooth, lap (default edreg.defaultWeights)
%     UseJacobian    true: pass the analytic Jacobian (SpecifyObjectiveGradient);
%                    false: lsqnonlin finite differences (slow: 6+12N
%                    residual evaluations per iteration)
%     MaxIterations  iteration cap (default 30)
%     InitialParam   starting parameter vector (default identity)
%     Display        lsqnonlin Display option (default 'off')
%     ShowIterations draw source/deformed/target after every iteration (default false)
%   Outputs
%     P   : optimised parameter vector
%     out : .resnormHistory  resnorm after iteration 0..end (column vector)
%           .timeHistory     seconds since the start of lsqnonlin at each recorded iteration
%           .elapsed         wall-clock seconds
%           .iterations, .funcCount, .exitflag, .resnorm, .Vd (final deformed vertices)
    arguments
        model struct
        target struct
        opts.Weights = []
        opts.UseJacobian (1,1) logical = true
        opts.MaxIterations (1,1) double {mustBeInteger, mustBePositive} = 30
        opts.InitialParam (:,1) double = []
        opts.Display (1,1) string = "off"
        opts.ShowIterations (1,1) logical = false
    end
    w = edreg.defaultWeights(opts.Weights);
    P0 = opts.InitialParam;
    if isempty(P0)
        P0 = edreg.identityParam(model.nNodes);
    end

    history = zeros(0, 1);
    timeHist = zeros(0, 1);
    tStart = tic;
    fig = [];
    options = optimoptions('lsqnonlin', Display=opts.Display, ...
        SpecifyObjectiveGradient=opts.UseJacobian, FunctionTolerance=1e-10, ...
        MaxIterations=opts.MaxIterations, OutputFcn=@recordIteration);
    objective = @(x) edreg.residual(x, model, target, w);

    [P, resnorm, ~, exitflag, output] = lsqnonlin(objective, P0, [], [], options);
    elapsed = toc(tStart);
    if ~isempty(fig) && isvalid(fig)
        close(fig);
    end

    out.resnormHistory = history;
    out.timeHistory = timeHist;
    out.elapsed = elapsed;
    out.iterations = output.iterations;
    out.funcCount = output.funcCount;
    out.exitflag = exitflag;
    out.resnorm = resnorm;
    out.Vd = edreg.deform(P, model);

    function stop = recordIteration(x, optimValues, state)
        stop = false;
        if state == "iter"
            history(end+1, 1) = optimValues.resnorm; %#ok<AGROW>
            timeHist(end+1, 1) = toc(tStart); %#ok<AGROW>
            if opts.ShowIterations
                Vi = edreg.deform(x, model);
                fig = edreg.showMeshes({struct('V', model.V, 'F', model.F), ...
                    struct('V', Vi, 'F', model.F), target}, Figure=fig, Visible="on");
                drawnow;
            end
        end
    end
end
