function [R, dR] = rodrigues(r)
%RODRIGUES  Rotation vector to rotation matrix, with its Jacobian.
%   [R, dR] = edreg.rodrigues(r)
%
%   Input
%     r  : 3-vector (axis * angle)
%   Output
%     R  : 3-by-3 rotation matrix, R = expm(skew(r))
%     dR : 9-by-3 matrix, dR(:, i) = d vec(R) / d r(i) (column-major vec)
%
%   With theta = |r|, K = skew(r), R = I + a(theta) K + b(theta) K^2 where
%   a = sin(theta)/theta and b = (1-cos(theta))/theta^2.  Differentiating,
%       dR/dr_i = a K_i + b (K_i K + K K_i) + r_i (alpha K + beta K^2),
%   with K_i = skew(e_i), alpha = a'(theta)/theta and beta = b'(theta)/theta.
%   For theta < 1e-2 the coefficients are evaluated by their Taylor series
%   (no 0/0), so r = 0 gives R = I and dR/dr_i = skew(e_i) exactly.
    arguments
        r (3,1) double
    end
    th2 = r' * r;
    th = sqrt(th2);
    if th < 1e-2
        a     = 1 - th2/6 + th2^2/120 - th2^3/5040;
        b     = 1/2 - th2/24 + th2^2/720 - th2^3/40320;
        alpha = -1/3 + th2/30 - th2^2/840;
        beta  = -1/12 + th2/180 - th2^2/6720;
    else
        a     = sin(th) / th;
        b     = (1 - cos(th)) / th2;
        alpha = (th*cos(th) - sin(th)) / th^3;
        beta  = (th*sin(th) - 2*(1 - cos(th))) / th^4;
    end
    K = skew(r);
    K2 = K * K;
    R = eye(3) + a*K + b*K2;
    if nargout > 1
        dR = zeros(9, 3);
        for i = 1:3
            Ki = skew(double((1:3)' == i));
            dRi = a*Ki + b*(Ki*K + K*Ki) + r(i)*(alpha*K + beta*K2);
            dR(:, i) = dRi(:);
        end
    end
end

function K = skew(v)
    K = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
end
