function S = getSphericalShade(Nx, Ny, Nz, L_A_Ex, Vsp)
    Y = makeSphericalBasis(Nx, Ny, Nz);
    S = calSphericalShade(Vsp, L_A_Ex, Y);
end