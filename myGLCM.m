% myGLCM Builds symmetric normalized 64-level GLCMs and four Haralick statistics.
function features = myGLCM(grayImg)
    maxValue = max(grayImg(:));
    if maxValue == 0
        features = [0, 0, 1, 1];
        return;
    end
    levels = 64;
    quantized = min(levels - 1, floor(double(grayImg) / double(maxValue) * (levels - 1)));
    offsets = [0 1; 1 1; 1 0; 1 -1];
    glcm = zeros(levels, levels);
    [rows, cols] = size(quantized);
    for offset = 1:size(offsets,1)
        dr = offsets(offset,1); dc = offsets(offset,2);
        for row = 1:rows
            for col = 1:cols
                nr = row + dr; nc = col + dc;
                if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols
                    a = quantized(row,col) + 1; b = quantized(nr,nc) + 1;
                    glcm(a,b) = glcm(a,b) + 1; glcm(b,a) = glcm(b,a) + 1;
                end
            end
        end
    end
    glcm = glcm / sum(glcm(:));
    [i, j] = ndgrid(1:levels, 1:levels);
    contrast = sum(sum((i - j).^2 .* glcm));
    muI = sum(sum(i .* glcm)); muJ = sum(sum(j .* glcm));
    sigmaI = sqrt(sum(sum((i - muI).^2 .* glcm)));
    sigmaJ = sqrt(sum(sum((j - muJ).^2 .* glcm)));
    if sigmaI == 0 || sigmaJ == 0
        correlation = 0;
    else
        correlation = sum(sum((i - muI) .* (j - muJ) .* glcm)) / (sigmaI * sigmaJ);
    end
    energy = sum(sum(glcm.^2));
    homogeneity = sum(sum(glcm ./ (1 + abs(i - j))));
    features = [contrast, correlation, energy, homogeneity];
end
