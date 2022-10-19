function [gradX, gradY, gradZ] = gradient3DN(f)
    [H, W, D] = size(f);
    pos = f > 0;
    
    if isa(f, 'gpuArray')
        gradXp = gpuArray(ones(size(f)));
        gradYp = gpuArray(ones(size(f)));
        gradZp = gpuArray(ones(size(f)));
    else
        gradXp = ones(size(f));
        gradYp = ones(size(f));
        gradZp = ones(size(f));
    end
    
    gradXn = jwutils.gradient3DX(f, 0);
    gradXp(:, 1:(W-1), :) = gradXn(:, 2:W, :);
    gradposXInc = gradXp <= 0;
    gradposXDec = gradXn >= 0;
    gradnegXInc = gradXn <= 0;
    gradnegXDec = gradXp >= 0;
    gradX = pos.*(gradposXInc.*gradXp + gradposXDec.*gradXn) ./ (gradposXInc+gradposXDec)+...
        ~pos.*(gradnegXInc.*gradXn + gradnegXDec.*gradXp) ./ (gradnegXInc+gradnegXDec);   

    
    gradYn = jwutils.gradient3DY(f, 0);
    gradYp(1:(H-1), :, :) = gradYn(2:H, :, :);
    gradposYInc = gradYp <= 0;
    gradposYDec = gradYn >= 0;
    gradnegYInc = gradYn <= 0;
    gradnegYDec = gradYp >= 0;
    gradY = pos.*(gradposYInc.*gradYp + gradposYDec.*gradYn) ./ (gradposYInc+gradposYDec) +...
        ~pos.*(gradnegYInc.*gradYn + gradnegYDec.*gradYp) ./ (gradnegYInc+gradnegYDec);
    
    
    gradZn = jwutils.gradient3DZ(f, 0);
    gradZp(:, :, 1:(D-1)) = gradZn(:, :, 2:D);
    gradposZInc = gradZp <= 0;
    gradposZDec = gradZn >= 0;
    gradnegZInc = gradZn <= 0;
    gradnegZDec = gradZp >= 0;
    gradZ = pos.*(gradposZInc.*gradZp + gradposZDec.*gradZn) ./ (gradposZInc+gradposZDec) +...
        ~pos.*(gradnegZInc.*gradZn + gradnegZDec.*gradZp) ./ (gradnegZInc+gradnegZDec);


end