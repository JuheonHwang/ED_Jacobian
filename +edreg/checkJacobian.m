function report = checkJacobian(P, model, target, w, opts)
%CHECKJACOBIAN  Compare the analytic Jacobian with central finite differences.
%   report = edreg.checkJacobian(P, model, target, w)
%   report = edreg.checkJacobian(P, model, target, w, Step=1e-4, Columns=[1:6 100:110])
%
%   Correspondences are computed once at P and then held FIXED for both
%   Jacobians (otherwise the nearest-neighbour switch makes the energy
%   non-differentiable and finite differences meaningless).
%
%   Inputs
%     P, model, target, w : as in edreg.residual (w = [] for defaults)
%   Options
%     Step     : finite-difference step (default 1e-4)
%     Columns  : parameter columns to test (default all); blocks without any
%                selected column are skipped
%     RelFloor : floor of the relative-error denominator as a fraction of the
%                largest |entry| of the whole term (default 1e-4). Blocks that
%                are exactly zero (e.g. Lap w.r.t. T) are then judged against
%                the finite-difference noise level instead of dividing by 0.
%     Display  : print the table (default true)
%   Output
%     report : table with one row per (term, block) and variables
%              Term, Block, MaxAbsErr, MaxRelErr, MaxAbsJ
%              MaxRelErr = MaxAbsErr / max(MaxAbsJ, RelFloor * termScale), with
%              MaxAbsJ the largest |entry| of the analytic block and termScale
%              the largest |entry| over all selected columns of that term.
    arguments
        P (:,1) double
        model struct
        target struct
        w = []
        opts.Step (1,1) double {mustBePositive} = 1e-4
        opts.Columns (1,:) double {mustBeInteger, mustBePositive} = 1:numel(P)
        opts.RelFloor (1,1) double {mustBeNonnegative} = 1e-4
        opts.Display (1,1) logical = true
    end
    w = edreg.defaultWeights(w);
    [~, J, info] = edreg.residual(P, model, target, w);
    corr = info.corr;
    fun = @(x) edreg.residual(x, model, target, w, corr);
    Jn = edreg.numericJacobian(fun, P, Step=opts.Step, Columns=opts.Columns);

    terms = string(fieldnames(info.rows))';
    blocks = string(fieldnames(info.cols))';
    Term = strings(0, 1); Block = strings(0, 1);
    MaxAbsErr = zeros(0, 1); MaxRelErr = zeros(0, 1); MaxAbsJ = zeros(0, 1);
    for tm = terms
        r = info.rows.(tm);
        termScale = max(abs(full(J(r, opts.Columns))), [], 'all');
        for b = blocks
            sel = ismember(opts.Columns, info.cols.(b));
            if ~any(sel)
                continue
            end
            Ja = full(J(r, opts.Columns(sel)));
            Jb = Jn(r, sel);
            err = max(abs(Ja - Jb), [], 'all');
            scale = max(abs(Ja), [], 'all');
            Term(end+1, 1) = tm; Block(end+1, 1) = b; %#ok<AGROW>
            MaxAbsErr(end+1, 1) = err; %#ok<AGROW>
            MaxAbsJ(end+1, 1) = scale; %#ok<AGROW>
            MaxRelErr(end+1, 1) = err / max(scale, opts.RelFloor * termScale); %#ok<AGROW>
        end
    end
    report = table(Term, Block, MaxAbsErr, MaxRelErr, MaxAbsJ);
    if opts.Display
        fprintf('Jacobian check (step %.1e, %d columns, fixed correspondences)\n', opts.Step, numel(opts.Columns));
        disp(report);
        fprintf('worst MaxRelErr = %.2e\n', max(report.MaxRelErr));
    end
end
