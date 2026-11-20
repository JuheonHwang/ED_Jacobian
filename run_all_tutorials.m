function run_all_tutorials()
%RUN_ALL_TUTORIALS  Run tutorials t01-t06 in order, each in a clean workspace.
%   run_all_tutorials
%
%   Every tutorial is executed inside its own function call (so that the
%   tutorials' "clear" does not touch this function), the wall time of each is
%   printed, and the first failure stops the run with an error naming the
%   tutorial.  Figures are written to results/figures (headless).
    repoRoot = fileparts(mfilename('fullpath'));
    addpath(repoRoot);
    names = ["t01_ed_model" "t02_energy" "t03_jacobian" "t04_jacobian_check" "t05_solvers" "t06_performance"];
    times = zeros(size(names));
    for i = 1:numel(names)
        fprintf('\n===== %s =====\n', names(i));
        t = tic;
        try
            runOne(fullfile(repoRoot, 'tutorials', names(i) + ".m"));
        catch err
            error('run_all_tutorials:failed', 'Tutorial %s failed after %.1f s: %s', ...
                names(i), toc(t), err.message);
        end
        times(i) = toc(t);
        close all force;
    end
    fprintf('\n%-22s %10s\n', 'tutorial', 'time [s]');
    for i = 1:numel(names)
        fprintf('%-22s %10.1f\n', names(i), times(i));
    end
    fprintf('%-22s %10.1f\n', 'total', sum(times));
end

function runOne(file)
% Separate function = clean workspace for the script (which starts with clear).
    run(file);
end
