function dispMesh (shp, tl, color, ~)
% Note, shp has nx3 elements and tl has mx3 elements
% Note, If saveMesh activated, mesh is saved as obj file format

    if length(color) == 4
        alpha = color(4);
        color = color(1:3);
    elseif length(color) == 3
        color = color(1:3);
        alpha = 1;
    else
        alpha = 1;
    %     FV.facevertexcdata = color;
    end

    FV.vertices = shp;
    FV.faces = tl;
    
    if length(color) <=4
        patch(FV, 'facecolor', color, 'edgecolor', 'none', 'vertexnormalsmode', 'auto', 'FaceAlpha', alpha);
    else
        patch(FV,  'facecolor', 'interp', 'FaceVertexCData', color, 'edgecolor', 'none', 'vertexnormalsmode', 'auto', 'FaceAlpha', alpha);
    end
    
    if nargin < 4
        camlight(180, 0);
        camlight(0, 0);
        %camlight('headlight');
        lighting phong;
        material dull;
    end
    axis vis3d
    axis equal;
end