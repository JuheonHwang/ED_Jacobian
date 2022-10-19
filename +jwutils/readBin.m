function [bin, dims] = readBin(filename, type)
    fid = fopen(filename, 'rb');
    n_dim = fread(fid, 1, 'int');
    dims = fread(fid, n_dim, 'int');
    dims = dims(:)';
    
    if nargin > 2
        bin = fread(fid,prod(dims), type);
    else
        bin = fread(fid,prod(dims), 'uint8');
    end
    
    bin = reshape(bin, dims);
    fclose(fid);
end
