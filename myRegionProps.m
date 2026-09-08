% myRegionProps Measures the largest component using pixel moments and a convex hull.
function props = myRegionProps(mask)
    names = {'Area','Perimeter','CentroidX','CentroidY','MajorAxisLength','MinorAxisLength','Eccentricity','Solidity','Extent'};
    values = zeros(1, numel(names));
    [labels, sizes] = myConnectedComponents(mask);
    if isempty(sizes)
        props = cell2struct(num2cell(values), names, 2); return;
    end
    [~, largest] = max(sizes);
    component = labels == largest;
    [rows, cols] = find(component);
    area = numel(rows);
    centroidX = mean(cols); centroidY = mean(rows);
    boundary = component & conv2(double(component), ones(3), 'same') < 9;
    perimeter = sum(boundary(:));
    centered = [cols - centroidX, rows - centroidY];
    covariance = (centered' * centered) / area;
    axesValues = sort(eig(covariance), 'descend');
    major = 4 * sqrt(max(axesValues(1), 0));
    minor = 4 * sqrt(max(axesValues(2), 0));
    if major > 0, eccentricity = sqrt(max(0, 1 - (minor / major)^2)); else, eccentricity = 0; end
    minRow = min(rows); maxRow = max(rows); minCol = min(cols); maxCol = max(cols);
    extent = area / ((maxRow - minRow + 1) * (maxCol - minCol + 1));
    if area >= 3
        hull = convhull(cols, rows);
        hullArea = polyarea(cols(hull), rows(hull));
        solidity = min(1, area / max(hullArea, 1));
    else
        solidity = 1;
    end
    values = [area, perimeter, centroidX, centroidY, major, minor, eccentricity, solidity, extent];
    props = cell2struct(num2cell(values), names, 2);
end
