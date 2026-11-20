function r = rotationToVector(R)
%ROTATIONTOVECTOR  Rotation matrix to rotation vector (inverse of edreg.rodrigues).
%   r = edreg.rotationToVector(R)
%
%   Input  : R, 3-by-3 rotation matrix
%   Output : r, 3-by-1 vector (axis * angle) with |r| in [0, pi]
%
%   Away from 0 and pi the skew-symmetric part of R gives v = sin(theta)*axis.
%   Near pi that part vanishes, so the axis is taken from the symmetric part
%   (R + R')/2 - cos(theta) I = (1 - cos(theta)) axis*axis' instead.
    arguments
        R (3,3) double
    end
    v = [R(3,2) - R(2,3); R(1,3) - R(3,1); R(2,1) - R(1,2)] / 2;
    s = norm(v);
    c = min(max((trace(R) - 1) / 2, -1), 1);
    theta = atan2(s, c);
    if theta < 1e-8
        r = v;
    elseif pi - theta > 1e-3
        r = v * (theta / s);
    else
        B = (R + R') / 2 - c * eye(3);
        [~, k] = max(diag(B));
        axisv = B(:, k) / norm(B(:, k));
        if axisv' * v < 0
            axisv = -axisv;
        end
        r = axisv * theta;
    end
end
