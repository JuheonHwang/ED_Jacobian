function [fig, ax] = showMeshes(meshes, opts)
%SHOWMESHES  Overlay several triangle meshes (source / deformed / target) in one figure.
%   [fig, ax] = edreg.showMeshes({src, deformed, target})
%   [fig, ax] = edreg.showMeshes(meshes, Visible="off", SavePath="a.png", Title="...")
%
%   Input
%     meshes : cell array of structs with fields V (n-by-3) and F (m-by-3);
%              drawn with jwutils.dispMesh
%   Options
%     Colors     k-by-4 RGBA rows, one per mesh (default: blue, red, green, grey)
%     Rotation   [azimuth elevation] passed to camorbit as in the legacy code
%                (default [-90 0]: front view of the body scans)
%     Title      figure title text
%     Visible    "on" | "off" (default "on"); use "off" for headless runs
%     SavePath   if given, the figure is exported to this PNG file
%     Figure     existing figure to redraw into (cleared first)
%   Outputs
%     fig, ax : figure and axes handles (add overlays such as scatter3 on ax)
    arguments
        meshes cell
        opts.Colors (:,4) double = [0.30 0.35 0.85 0.35; 0.85 0.25 0.25 0.55; 0.25 0.65 0.25 0.35; 0.5 0.5 0.5 0.5]
        opts.Rotation (1,2) double = [-90 0]
        opts.Title string = ""
        opts.Visible (1,1) string {mustBeMember(opts.Visible, ["on" "off"])} = "on"
        opts.SavePath string = ""
        opts.Figure = []
    end
    if isempty(opts.Figure) || ~isvalid(opts.Figure)
        fig = edreg.newFigure(Visible=opts.Visible);
    else
        fig = opts.Figure;
        clf(fig);
    end
    ax = axes('Parent', fig);
    hold(ax, 'on');
    for i = 1:numel(meshes)
        c = opts.Colors(mod(i - 1, size(opts.Colors, 1)) + 1, :);
        if i == 1
            jwutils.dispMesh(meshes{i}.V, meshes{i}.F, c);       % also adds the lights
        else
            jwutils.dispMesh(meshes{i}.V, meshes{i}.F, c, true); % extra arg: skip lights
        end
    end
    camorbit(ax, opts.Rotation(1), 0, 'data', [0 0 1]);
    camorbit(ax, opts.Rotation(2), 0, 'data', [1 0 0]);
    axis(ax, 'off');
    if strlength(opts.Title) > 0
        title(ax, opts.Title, 'Visible', 'on', 'Color', 'k');
    end
    if strlength(opts.SavePath) > 0
        folder = fileparts(opts.SavePath);
        if strlength(folder) > 0 && ~isfolder(folder)
            mkdir(folder);
        end
        exportgraphics(fig, opts.SavePath, 'Resolution', 150);
    end
end
