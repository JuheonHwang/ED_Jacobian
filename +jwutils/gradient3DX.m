function gradX = gradient3DX(f, flag)
    W = size(f, 2);
    if nargin < 2
        gradX = (f(:, [2:W W], :) - f(:, [1 1:(W-1)], :)) * 0.5;
    else
        gradX = (f(:, 1:W, :) - f(:, [1 1:(W-1)], :));
    end
end