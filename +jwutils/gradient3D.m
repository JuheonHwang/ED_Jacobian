function [gradX, gradY, gradZ] = gradient3D(f, w)
    if nargin < 2
        gradX = jwutils.gradient3DX(f);
        gradY = jwutils.gradient3DY(f);
        gradZ = jwutils.gradient3DZ(f);
    else
        [H, W, D] = size(f);
        gradX = w.*jwutils.gradient3DX(f, 0);
        gradX(:, 1:(W-1), :) = gradX(:, 1:(W-1), :) + gradX(:, 2:W, :);
        gradY = jwutils.gradient3DY(f, 0);
        gradY(1:(H-1), :, :) = gradY(1:(H-1), :, :) + gradY(2:H, :, :);
        gradZ = jwutils.gradient3DZ(f, 0);
        gradZ(:, :, 1:(D-1)) = gradZ(:, :, 1:(D-1)) + gradZ(:, :, 2:D);
        
        gradX(:, 2:(W-1), :) = gradX(:, 2:(W-1), :) ./ 2;
        gradY(2:(H-1), :, :) = gradY(2:(H-1), :, :) ./ 2;
        gradZ(:, :, 2:(D-1)) = gradZ(:, :, 2:(D-1)) ./ 2;
    end
end