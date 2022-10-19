function [bin, dims] = readBin8(filename, type)
    fid = fopen(filename, 'rb');
    dims = fread(fid, 3, 'int');
    
    if nargin > 2
        bin = fread(fid,dims(1)*dims(2)*dims(3), type);
    else
        bin = fread(fid,dims(1)*dims(2)*dims(3), 'int8');
    end
    
    bin = reshape(bin,[dims(1),dims(2),dims(3)]);
    fclose(fid);
end
