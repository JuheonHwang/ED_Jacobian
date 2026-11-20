function idx = findCorrespondence(Vd, targetV)
%FINDCORRESPONDENCE  Nearest deformed source vertex for every target point.
%   idx = edreg.findCorrespondence(Vd, targetV)
%
%   Inputs
%     Vd      : n-by-3 deformed source vertices
%     targetV : nT-by-3 target points
%   Output
%     idx     : nT-by-1 indices into Vd (legacy direction: target -> source)
%
%   Uses multiQueryKNNSearchImpl (Computer Vision Toolbox) because knnsearch
%   on plain double arrays needs the Statistics Toolbox.
    arguments
        Vd (:,3) double
        targetV (:,3) double
    end
    idx = multiQueryKNNSearchImpl(pointCloud(Vd), targetV, 1);
    idx = double(idx(:));
end
