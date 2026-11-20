function Jn = numericJacobian(fun, P, opts)
%NUMERICJACOBIAN  Central-difference Jacobian of a vector function.
%   Jn = edreg.numericJacobian(fun, P)
%   Jn = edreg.numericJacobian(fun, P, Step=1e-6, Columns=1:20)
%
%   Inputs
%     fun : function handle, F = fun(P) returning a column vector
%     P   : parameter column vector
%   Options
%     Step    : absolute step h (default 1e-6)
%     Columns : indices of the columns to compute (default all)
%   Output
%     Jn : numel(F)-by-numel(Columns) full matrix,
%          Jn(:,j) = (fun(P + h e_c) - fun(P - h e_c)) / (2h) for c = Columns(j)
    arguments
        fun (1,1) function_handle
        P (:,1) double
        opts.Step (1,1) double {mustBePositive} = 1e-6
        opts.Columns (1,:) double {mustBeInteger, mustBePositive} = 1:numel(P)
    end
    assert(all(opts.Columns <= numel(P)), 'edreg:numericJacobian:columns', 'Columns out of range.');
    h = opts.Step;
    F0 = fun(P);
    Jn = zeros(numel(F0), numel(opts.Columns));
    for j = 1:numel(opts.Columns)
        c = opts.Columns(j);
        Pp = P; Pp(c) = Pp(c) + h;
        Pm = P; Pm(c) = Pm(c) - h;
        Jn(:, j) = (fun(Pp) - fun(Pm)) / (2*h);
    end
end
