function S = calSphericalShade(Vsp, L_A_Ex, Y)
    S = max(sum(Vsp .* L_A_Ex .* Y, 2), 0.0);
end
