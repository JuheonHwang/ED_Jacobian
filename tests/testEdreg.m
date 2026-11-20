function tests = testEdreg
%TESTEDREG  Function-based unit tests of the edreg package and its solvers.
%   Run from the repository root with  runtests('tests')
%   All tests use a coarse ED model (SampleRate 0.004, about 20 nodes) of the
%   f10k body-scan pair and a handful of iterations, so the suite runs in
%   well under a minute.
    tests = functiontests(localfunctions);
end

% ======================================================================
%  Fixture
% ======================================================================
function setupOnce(testCase)
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repoRoot);
    testCase.addTeardown(@() rmpath(repoRoot));
    src = edreg.loadMesh(fullfile(repoRoot, 'data', 'human_020_f10k.ply'));
    tgt = edreg.loadMesh(fullfile(repoRoot, 'data', 'human_022_f10k.ply'));
    d.src = src;
    d.tgt = tgt;
    d.model = buildConnected(src, 0.004, struct());
    d.w = edreg.defaultWeights();
    d.P0 = edreg.identityParam(d.model.nNodes);
    d.fun = @(P) edreg.residual(P, d.model, tgt, d.w);
    testCase.TestData = d;
end

function model = buildConnected(src, rate, extra)
% ED model with a fixed seed; extra holds additional buildModel options.
    extra.SampleRate = rate;
    extra.Seed = 1;
    args = namedargs2cell(extra);
    model = edreg.buildModel(src.V, src.F, args{:});
end

% ======================================================================
%  Core package
% ======================================================================
function testRodriguesJacobian(testCase)
    rng(3);
    rs = [zeros(3, 1), 1e-4 * randn(3, 1), 0.7 * randn(3, 1), 2.5 * randn(3, 1)];
    h = 1e-6;
    for k = 1:size(rs, 2)
        r = rs(:, k);
        [R, dR] = edreg.rodrigues(r);
        verifyEqual(testCase, R' * R, eye(3), 'AbsTol', 1e-12);
        dFD = zeros(9, 3);
        for i = 1:3
            e = zeros(3, 1); e(i) = h;
            Rp = edreg.rodrigues(r + e);
            Rm = edreg.rodrigues(r - e);
            dFD(:, i) = (Rp(:) - Rm(:)) / (2*h);
        end
        verifyEqual(testCase, dR, dFD, 'AbsTol', 1e-7, ...
            sprintf('d vec(R)/dr mismatch for |r| = %.2g', norm(r)));
    end
    [R0, dR0] = edreg.rodrigues(zeros(3, 1));          % r = 0 must not give 0/0
    verifyEqual(testCase, R0, eye(3), 'AbsTol', 1e-15);
    verifyTrue(testCase, all(isfinite(dR0), 'all'));
end

function testEncodeDecodeRoundTrip(testCase)
    rng(4);
    N = 7;
    r = [0.3; -0.5; 0.2];
    R = edreg.rodrigues(r);
    T = randn(3, 1);
    A = randn(3, 3, N);
    t = randn(3, N);
    P = edreg.encodeParam(R, T, A, t);
    verifySize(testCase, P, [6 + 12*N, 1]);
    [R2, T2, A2, t2] = edreg.decodeParam(P, N);
    verifyEqual(testCase, R2, R, 'AbsTol', 1e-12);
    verifyEqual(testCase, T2, T);
    verifyEqual(testCase, A2, A);
    verifyEqual(testCase, t2, t);
    verifyEqual(testCase, edreg.encodeParam(R2, T2, A2, t2), P, 'AbsTol', 1e-12);
end

function testIdentityParam(testCase)
    N = 5;
    P = edreg.identityParam(N);
    [R, T, A, t] = edreg.decodeParam(P, N);
    verifyEqual(testCase, R, eye(3));
    verifyEqual(testCase, T, zeros(3, 1));
    verifyEqual(testCase, A, repmat(eye(3), [1 1 N]));
    verifyEqual(testCase, t, zeros(3, N));
end

function testCheckJacobianDefaultNeighbours(testCase)
    d = testCase.TestData;
    P = randomParam(d.model.nNodes);
    report = edreg.checkJacobian(P, d.model, d.tgt, d.w, Columns=sampleColumns(d.model.nNodes), Display=false);
    verifyLessThan(testCase, max(report.MaxRelErr), 1e-4);
end

function testCheckJacobianNonDefaultNeighbours(testCase)
    d = testCase.TestData;
    model = buildConnected(d.src, 0.004, struct('NumVertexNeighbors', 3, 'NumNodeNeighbors', 4));
    verifyEqual(testCase, size(model.idxNeighbor, 2), 3);
    verifyEqual(testCase, size(model.idxNodeNeighbor, 2), 4);
    P = randomParam(model.nNodes);
    report = edreg.checkJacobian(P, model, d.tgt, d.w, Columns=sampleColumns(model.nNodes), Display=false);
    verifyLessThan(testCase, max(report.MaxRelErr), 1e-4);
end

function testFitTermVanishesForIdenticalMeshes(testCase)
    d = testCase.TestData;
    [F, ~, info] = edreg.residual(d.P0, d.model, struct('V', d.src.V), d.w);
    verifyLessThan(testCase, norm(F(info.rows.fit)), 1e-10);
    % the rotation and smoothness terms vanish at the identity as well
    verifyLessThan(testCase, norm(F(info.rows.rot1)), 1e-12);
    verifyLessThan(testCase, norm(F(info.rows.rot2)), 1e-12);
    verifyLessThan(testCase, norm(F(info.rows.smooth)), 1e-12);
end

function testBuildModelReplacesFragmentNodes(testCase)
    d = testCase.TestData;
    model = edreg.buildModel(d.src.V, d.src.F, SampleRate=0.02, Seed=1);
    verifyGreaterThanOrEqual(testCase, model.nReplacedNodes, 1);
    P = randomParam(model.nNodes);
    report = edreg.checkJacobian(P, model, d.tgt, d.w, Columns=sampleColumns(model.nNodes), Display=false);
    verifyLessThan(testCase, max(report.MaxRelErr), 1e-4);
end

% ======================================================================
%  Solvers
% ======================================================================
function testGradientDescentReducesEnergy(testCase)
    d = testCase.TestData;
    [~, h] = edreg.solvers.gradientDescent(d.fun, d.P0, StepSize=2e-4, MaxIterations=10);
    verifyHistory(testCase, h, 10);
    verifyLessThanOrEqual(testCase, diff(h.resnorm), 1e-9 * h.resnorm(1));   % monotone at this step size
    verifyLessThan(testCase, h.resnorm(end), h.resnorm(1));
end

function testGaussNewtonReducesEnergy(testCase)
    d = testCase.TestData;
    [~, h] = edreg.solvers.gaussNewton(d.fun, d.P0, MaxIterations=6);
    verifyHistory(testCase, h, 6);
    verifyLessThan(testCase, h.resnorm(end), h.resnorm(1));      % undamped: not necessarily monotone
end

function testGaussNewtonLsqVariantRuns(testCase)
    d = testCase.TestData;
    [~, h] = edreg.solvers.gaussNewton(d.fun, d.P0, LinearSolver="lsq", MaxIterations=1);
    verifyEqual(testCase, h.iterations, 1);
    verifyLessThan(testCase, h.resnorm(2), h.resnorm(1));
end

function testLevenbergMarquardtIsMonotone(testCase)
    d = testCase.TestData;
    [~, h] = edreg.solvers.levenbergMarquardt(d.fun, d.P0, MaxIterations=8);
    verifyHistory(testCase, h, 8);
    verifyLessThan(testCase, diff(h.resnorm), 0);                % accept/reject guarantees descent
    verifyGreaterThanOrEqual(testCase, h.lambda(2:end), 0);
    verifyGreaterThanOrEqual(testCase, h.nTrials(2:end), 1);
end

function testLevenbergMarquardtLegacyVariantReducesEnergy(testCase)
    d = testCase.TestData;
    [~, h] = edreg.solvers.levenbergMarquardt(d.fun, d.P0, Variant="hjh1", MaxIterations=6);
    verifyHistory(testCase, h, 6);
    verifyLessThan(testCase, h.resnorm(end), h.resnorm(1));      % hjh1 may accept uphill steps
end

function testLevenbergMarquardtCGReachesDirectEnergy(testCase)
    d = testCase.TestData;
    [~, hd] = edreg.solvers.levenbergMarquardt(d.fun, d.P0, MaxIterations=4);
    [~, hc] = edreg.solvers.levenbergMarquardtCG(d.fun, d.P0, MaxIterations=4);
    verifyLessThan(testCase, diff(hc.resnorm), 0);
    verifyGreaterThan(testCase, hc.cgIterations(2), 0);
    verifyEqual(testCase, hc.resnorm(end), hd.resnorm(end), 'RelTol', 1e-3);
    [~, hp] = edreg.solvers.levenbergMarquardtCG(d.fun, d.P0, Preconditioner="none", MaxIterations=2);
    verifyLessThan(testCase, hp.resnorm(end), hp.resnorm(1));
end

function testCallbackAndTolerance(testCase)
    d = testCase.TestData;
    calls = containers.Map({'n'}, {0});                      % handle object: callbacks can update it
    [~, h] = edreg.solvers.levenbergMarquardt(d.fun, d.P0, MaxIterations=3, ...
        Callback=@(~, ~) countCall(calls));
    verifyEqual(testCase, calls('n'), numel(h.iteration));
    [~, h] = edreg.solvers.levenbergMarquardt(d.fun, d.P0, MaxIterations=50, FunctionTolerance=0.5);
    verifyEqual(testCase, h.exitReason, "FunctionTolerance");
    verifyLessThan(testCase, h.iterations, 50);
end

function testCGStepEqualsDirectStep(testCase)
    % One LM iteration with a fixed lambda: the CG-solved damped step must equal
    % the Cholesky-solved one and the explicit solution of the damped system.
    d = testCase.TestData;
    [F, J] = d.fun(d.P0);
    JtJ = J' * J;
    lambda = 1e-2;
    Adamp = JtJ + lambda * spdiags(diag(JtJ), 0, size(JtJ, 1), size(JtJ, 1));
    stepRef = -(Adamp \ (J' * F));
    [~, hd] = edreg.solvers.levenbergMarquardt(d.fun, d.P0, LambdaInit=lambda, MaxIterations=1);
    [~, hc] = edreg.solvers.levenbergMarquardtCG(d.fun, d.P0, LambdaInit=lambda, MaxIterations=1);
    verifyEqual(testCase, hd.stepNorm(2), norm(stepRef), 'RelTol', 1e-6);
    verifyEqual(testCase, hc.stepNorm(2), norm(stepRef), 'RelTol', 1e-4);
end

% ======================================================================
%  Helpers
% ======================================================================
function countCall(calls)
    calls('n') = calls('n') + 1;
end

function P = randomParam(N)
    rng(2);
    P = edreg.identityParam(N) + 0.05 * randn(6 + 12*N, 1);
    P(1:3) = 0.2 * randn(3, 1);
end

function cols = sampleColumns(N)
% R, T, all parameters of the first nodes' A and t blocks, and some random ones.
    rng(7);
    cols = unique([1:6, 6 + (1:27), 6 + 9*N + (1:9), 6 + randperm(12*N, 40)]);
end

function verifyHistory(testCase, h, maxIter)
    n = numel(h.iteration);
    verifyEqual(testCase, h.iteration, (0:n-1)');
    verifyLessThanOrEqual(testCase, n - 1, maxIter);
    for f = ["resnorm" "time" "stepNorm" "gradNorm"]
        verifySize(testCase, h.(f), [n 1]);
    end
    verifyTrue(testCase, all(diff(h.time) >= 0));
    verifyTrue(testCase, isnan(h.stepNorm(1)));
    verifyTrue(testCase, all(isfinite(h.resnorm)));
    verifyNotEmpty(testCase, h.exitReason);
    verifyEqual(testCase, h.iterations, n - 1);
end
