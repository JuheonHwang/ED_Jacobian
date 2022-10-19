function writeBin(vol, filename, dims, type)
    fid = fopen([filename '.bin'], 'wb');
    fwrite(fid, length(dims), 'int');
    fwrite(fid, int32(dims), 'int');
    if nargin > 3
        fwrite(fid, vol, type);
    else
        fwrite(fid, vol, 'uint8');
    end
    fclose(fid);
end