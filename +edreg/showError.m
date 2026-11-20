function [fig, ax] = showError(V, F, dist, opts)
%SHOWERROR  Draw a mesh coloured by a per-vertex scalar (e.g. distance to target).
%   [fig, ax] = edreg.showError(V, F, dist)
%   [fig, ax] = edreg.showError(V, F, dist, CLim=[0 20], Title="After", Axes=ax)
%
%   Inputs
%     V, F : n-by-3 vertices and m-by-3 triangles
%     dist : n-by-1 scalar per vertex, shown as face-interpolated colour
%   Options
%     CLim      [lo hi] colour limits (default [0, 95th percentile of dist]);
%               pass the same value to several calls for comparable figures
%     Colormap  colormap name or k-by-3 array (default "turbo")
%     Title     title text
%     Rotation  [azimuth elevation] camorbit angles as in edreg.showMeshes
%     Visible   "on" | "off" (default "off"); ignored when Axes is given
%     SavePath  PNG file to export (default none)
%     Axes      existing axes to draw into (e.g. a tile of a tiledlayout);
%               then no figure is created and nothing is exported
%     Label     colourbar label (default "distance")
%     Colorbar  true (default) adds a colourbar below the mesh; false omits it
%               (use one shared colourbar for several panels)
%     Limits    3-by-2 [xmin xmax; ymin ymax; zmin zmax] axis limits; give the
%               same value to several panels so the meshes are drawn at the
%               same scale (default: fit to this mesh)
%   Outputs
%     fig, ax : figure and axes handles
    arguments
        V (:,3) double
        F (:,3) double
        dist (:,1) double
        opts.CLim (1,2) double = [NaN NaN]
        opts.Colormap = "turbo"
        opts.Title string = ""
        opts.Rotation (1,2) double = [-90 0]
        opts.Visible (1,1) string {mustBeMember(opts.Visible, ["on" "off"])} = "off"
        opts.SavePath string = ""
        opts.Axes = []
        opts.Label string = "distance"
        opts.Colorbar (1,1) logical = true
        opts.Limits (3,2) double = nan(3, 2)
    end
    assert(numel(dist) == size(V, 1), 'edreg:showError:size', 'dist needs one value per vertex.');
    lims = opts.CLim;
    if any(isnan(lims))
        lims = [0, max(prctile95(dist), eps)];
    end
    if isempty(opts.Axes)
        fig = edreg.newFigure(Visible=opts.Visible);
        ax = axes('Parent', fig);
    else
        ax = opts.Axes;
        fig = ancestor(ax, 'figure');
    end
    set(fig, 'CurrentAxes', ax);
    hold(ax, 'on');
    jwutils.dispMesh(V, F, dist);              % per-vertex scalar -> interpolated colour
    colormap(ax, opts.Colormap);
    clim(ax, lims);
    if opts.Colorbar
        cb = colorbar(ax, 'southoutside');
        cb.Label.String = opts.Label;
    end
    if ~any(isnan(opts.Limits), 'all')
        xlim(ax, opts.Limits(1, :)); ylim(ax, opts.Limits(2, :)); zlim(ax, opts.Limits(3, :));
    end
    camorbit(ax, opts.Rotation(1), 0, 'data', [0 0 1]);
    camorbit(ax, opts.Rotation(2), 0, 'data', [1 0 0]);
    axis(ax, 'off');
    if strlength(opts.Title) > 0
        title(ax, opts.Title, 'Visible', 'on', 'Color', 'k');
    end
    if strlength(opts.SavePath) > 0 && isempty(opts.Axes)
        folder = fileparts(opts.SavePath);
        if strlength(folder) > 0 && ~isfolder(folder)
            mkdir(folder);
        end
        exportgraphics(fig, opts.SavePath, 'Resolution', 150);
    end
end

function p = prctile95(x)
% 95th percentile without the Statistics Toolbox.
    x = sort(x(:));
    p = x(max(1, ceil(0.95 * numel(x))));
end
