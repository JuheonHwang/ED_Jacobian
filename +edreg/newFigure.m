function fig = newFigure(opts)
%NEWFIGURE  Create a figure with the light theme and a white background.
%   fig = edreg.newFigure()
%   fig = edreg.newFigure(Visible="off", Position=[100 100 900 700])
%
%   MATLAB R2026b follows the system dark theme, which gives black axes in
%   headless runs; forcing the light theme keeps saved PNGs legible.
%
%   Options
%     Visible  : "on" | "off" (default "off", suitable for batch runs)
%     Position : [left bottom width height] in pixels (default [100 100 700 800])
%   Output
%     fig : figure handle (axes created afterwards inherit the light theme)
    arguments
        opts.Visible (1,1) string {mustBeMember(opts.Visible, ["on" "off"])} = "off"
        opts.Position (1,4) double = [100 100 700 800]
    end
    fig = figure('Visible', opts.Visible, 'Position', opts.Position, 'Color', 'w');
    try
        theme(fig, "light");
    catch
        % releases without figure themes are always light
    end
    fig.Color = 'w';
end
