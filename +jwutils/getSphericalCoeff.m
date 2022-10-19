function L_A_Ex = getSphericalCoeff()
    L_A = [3.141593, 2.094395, 0.785398, 0.0, -0.130900, 0.0, 0.049087];
    L_A_Ex = [];
    for i = 1:5
        if i == 4
            continue;
        end
        for j = (-i+1):(i-1)
            L_A_Ex = [L_A_Ex, L_A(i)]; %#ok<AGROW>
        end
    end
end