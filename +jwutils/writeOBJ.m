function writeObj(points, faces, filename, color, voxelGridOrigin, voxelSize)
    if size(points, 2) == 3
        points =points';
    end
    if size(faces, 2) == 3
        faces = faces';
    end
    
    if nargin < 4
        color = repmat([175;198;233]/255.,1,size(points,2));
    else
        if size(color, 2) == 3
            color = color';
        end
    end

    % Set mesh color (light blue)
    
    if nargin > 4
        % Transform mesh from voxel coordinates to camera coordinates
        meshPoints(1,:) = voxelGridOrigin(1) + points(1,:)*voxelSize; % x y axes are swapped from isosurface
        meshPoints(2,:) = voxelGridOrigin(2) + points(2,:)*voxelSize;
        meshPoints(3,:) = voxelGridOrigin(3) + points(3,:)*voxelSize;
    else
        meshPoints = points;
    end

    % Write header for mesh file
    if ~(length(filename) > 3 && strcmpi(filename((end-3):end), '.obj'))
        filename = [filename '.obj'];
    end
    
    fid = fopen(filename,'wt');
    
    % Write vertices
    fprintf(fid, "v %f %f %f %f %f %f\n", [meshPoints; color]);
    fprintf(fid, "f %u %u %u\n", faces);
    fclose(fid);

end