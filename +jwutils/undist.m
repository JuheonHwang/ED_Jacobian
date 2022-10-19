
function [Xproj_x, Xproj_y] = undist(Xproj_undist, dist)
    if length(dist) == 8
        dist = dist([1 2 5 6 7 8 3 4]);
        rs2 = Xproj_undist(:, 1).^2 + Xproj_undist(:, 2).^2;
        rs4 = rs2.*rs2;
        rs6 = rs4.*rs2;

        dist_mult = (1 + dist(1)*rs2 + dist(2)*rs4 + dist(3)*rs6) ./ (1 + dist(4)*rs2 + dist(5)*rs4 + dist(6)*rs6);

        Xproj_x = Xproj_undist(:, 1).*dist_mult + 2*dist(7)*Xproj_undist(:, 1).*Xproj_undist(:, 2) + dist(8).*(rs2 + 2*Xproj_undist(:, 1).^2);
        Xproj_y = Xproj_undist(:, 2).*dist_mult + dist(7)*(rs2 + 2*Xproj_undist(:, 2).^2) + 2*dist(8).*Xproj_undist(:, 1).*Xproj_undist(:, 2);
    elseif length(dist) == 5
        dist = dist([1 2 5 3 4]);
        rs2 = Xproj_undist(:, 1).^2 + Xproj_undist(:, 2).^2;
        rs4 = rs2.*rs2;
        rs6 = rs4.*rs2;

        dist_mult = (1 + dist(1)*rs2 + dist(2)*rs4 + dist(3)*rs6);

        Xproj_x = Xproj_undist(:, 1).*dist_mult + 2*dist(4)*Xproj_undist(:, 1).*Xproj_undist(:, 2) + dist(5).*(rs2 + 2*Xproj_undist(:, 1).^2);
        Xproj_y = Xproj_undist(:, 2).*dist_mult + dist(4)*(rs2 + 2*Xproj_undist(:, 2).^2) + 2*dist(5).*Xproj_undist(:, 1).*Xproj_undist(:, 2);
    else
        Xproj_x = Xproj_undist(:, 1);
        Xproj_y = Xproj_undist(:, 2);
    end
end