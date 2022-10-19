function Y = rgb2y(I1, I2, I3)
    Y = 0.299 * I1 + 0.587 * I2 + 0.114 * I3;
end