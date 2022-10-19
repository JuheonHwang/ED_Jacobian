function [meshPoints, faces, points] = makeTSDFSurface(tsdf, voxelGridOrigin, voxelSize)
    % Convert from TSDF to mesh  
    fv = isosurface(permute(tsdf, [2 1 3]), 0);
    points = fv.vertices';
    faces = fv.faces';

    % Transform mesh from voxel coordinates to camera coordinates
    meshPoints(1,:) = voxelGridOrigin(1) + points(1,:)*voxelSize; % x y axes are swapped from isosurface
    meshPoints(2,:) = voxelGridOrigin(2) + points(2,:)*voxelSize;
    meshPoints(3,:) = voxelGridOrigin(3) + points(3,:)*voxelSize;
end