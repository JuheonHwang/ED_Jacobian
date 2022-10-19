function f = div(nx, ny, nz)

    nxx = jwutils.gradient3DX(nx);
    nyy = jwutils.gradient3DY(ny);
    nzz = jwutils.gradient3DZ(nz);
    
    f=nxx+nyy+nzz;
end