function gradY = gradient3DY(f, flag)
    H = size(f, 1);
    if nargin < 2
        gradY = (f([2:H H], :, :) - f([1 1:(H-1)], :, :)) * 0.5;
    else
        gradY = (f(1:H, :, :) - f([1 1:(H-1)], :, :));
    end
end