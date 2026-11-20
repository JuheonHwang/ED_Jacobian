function mesh = loadMesh(plyPath)
%LOADMESH  Read a triangle mesh from a PLY file.
%   mesh = edreg.loadMesh(plyPath)
%
%   Input
%     plyPath : path to a PLY file with vertices and triangular faces.
%   Output
%     mesh.V  : n-by-3 vertex positions (double)
%     mesh.F  : m-by-3 one-based triangle vertex indices (double)
    arguments
        plyPath (1,1) string
    end
    assert(isfile(plyPath), 'edreg:loadMesh:notFound', 'File not found: %s', plyPath);
    [F, V] = jwutils.plyread(char(plyPath), 'tri');
    mesh.V = double(V);
    mesh.F = double(F);
end
