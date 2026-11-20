function d = surfaceDistance(V, targetV)
%SURFACEDISTANCE  Distance from every source vertex to the nearest target point.
%   d = edreg.surfaceDistance(V, targetV)
%
%   Inputs
%     V       : n-by-3 source (e.g. deformed) vertices
%     targetV : nT-by-3 target points
%   Output
%     d       : n-by-1 Euclidean distances (same units as the input)
%
%   Uses the same nearest-neighbour backend as edreg.findCorrespondence, with
%   the query/search roles swapped (source vertices query the target cloud).
    arguments
        V (:,3) double
        targetV (:,3) double
    end
    idx = multiQueryKNNSearchImpl(pointCloud(targetV), V, 1);
    d = sqrt(sum((V - targetV(double(idx(:)), :)).^2, 2));
end
