function writeTSDF(tsdf, filename, dims, orgins, size, margin)
    fid = fopen([filename '.bin'], 'wb');
    tsdfHeader = zeros(8, 1, 'single');
    tsdfHeader(1:3) = dims;
    tsdfHeader(4:6) = orgins;
    tsdfHeader(7) = size;
    tsdfHeader(8) = margin;
    fwrite(fid, single(tsdfHeader), 'single');
    fwrite(fid, single(tsdf), 'single');
    fclose(fid);
end