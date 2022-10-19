function M = readMatrix(filename, n)
    FP = fopen(filename, 'rt');
    M = fscanf(FP, '%f');
    if nargin < 2
        n = sqrt(numel(M));
    end
    M = reshape(M, [n n])';
    fclose(FP);
end