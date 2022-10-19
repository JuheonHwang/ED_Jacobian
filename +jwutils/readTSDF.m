function [tsdf, dims, orgins, size, margin] = readTSDF(filename)
    fid = fopen(filename, 'rb');
    tsdfHeader = fread(fid,8, 'single');
    dims = tsdfHeader(1:3);
    orgins = tsdfHeader(4:6);
    size = tsdfHeader(7);
    margin = tsdfHeader(8);
    tsdf = fread(fid,dims(1)*dims(2)*dims(3),'single');
    tsdf = reshape(tsdf,[dims(1),dims(2),dims(3)]);
    fclose(fid);
end