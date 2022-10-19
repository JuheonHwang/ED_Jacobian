function M = readVector(filename)
    FP = fopen(filename, 'rt');
    M = fscanf(FP, '%f');
    fclose(FP);
end
