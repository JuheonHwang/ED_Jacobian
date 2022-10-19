function [lap, divcomp] = lap(f, divdim, w)
    if nargin < 3
        [fx, fy, fz]= jwutils.gradient3D(f);  
    else
        [fx, fy, fz]= jwutils.gradient3D(f, w);  
        
    end
    
    lap = jwutils.div(fx, fy, fz);
    
    if nargin > 1
        switch divdim
            case 1
                divcomp = fx;
            case 2
                divcomp = fy;
            case 3
                divcomp = fz;
        end
    end
end