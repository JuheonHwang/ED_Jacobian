function [bin, dims, orgins, size] = readSil(filename)
    fid = fopen(filename, 'rb');
    dims = fread(fid, 3, 'int');
    orgins = fread(fid, 3, 'single');
    size = fread(fid, 1, 'single');
    bin = fread(fid,dims(1)*dims(2)*dims(3), 'uint8');
    bin = reshape(bin,[dims(1),dims(2),dims(3)]);
    fclose(fid);
end
