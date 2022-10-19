function gradZ = gradient3DZ(f, flag)
    D = size(f, 3);
    if nargin < 2
        gradZ = (f(:, :, [2:D D]) - f( :, :, [1 1:(D-1)])) * 0.5;
    else
        gradZ = (f(:, :, 1:D) - f( :, :, [1 1:(D-1)]));
    end
end