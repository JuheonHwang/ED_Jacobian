function w = defaultWeights(w)
%DEFAULTWEIGHTS  Energy weights (legacy defaults), optionally completing a partial struct.
%   w = edreg.defaultWeights()   returns struct with fields rot=1.0, smooth=0.1, lap=0.01.
%   w = edreg.defaultWeights(w)  fills the missing fields of w with the defaults.
    arguments
        w = []
    end
    if isempty(w)
        w = struct();
    end
    d = struct('rot', 1.0, 'smooth', 0.1, 'lap', 0.01);
    for f = string(fieldnames(d))'
        if ~isfield(w, f)
            w.(f) = d.(f);
        end
    end
end
