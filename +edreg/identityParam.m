function P = identityParam(nNodes)
%IDENTITYPARAM  Parameter vector of the identity deformation (r=0, T=0, A=I, t=0).
%   P = edreg.identityParam(nNodes)
    arguments
        nNodes (1,1) double {mustBeInteger, mustBePositive}
    end
    P = edreg.encodeParam(eye(3), zeros(3, 1), repmat(eye(3), [1 1 nNodes]), zeros(3, nNodes));
end
