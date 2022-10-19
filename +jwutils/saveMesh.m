function saveMesh(V, F, rot1, rot2, filepath)
    h = figure('visible', 'off');
    jwutils.dispMesh(V, F, [0.8 0.8 0.8 1.0]);
    camorbit(rot1, 0, 'data', [0 0 1]);
    camorbit(rot2, 0, 'data ', [1 0 0]);
    axis off;
    set(gcf, 'Position', get(0, 'ScreenSize'));
    imgframe = getframe(gcf);
    imwrite(imgframe.cdata, filepath);
    close(h);
end