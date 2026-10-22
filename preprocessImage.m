% preprocessImage Apply the toolbox-free preprocessing pipeline to one RGB image.
function result = preprocessImage(img, targetSize)
    if nargin < 2 || isempty(targetSize)
        targetSize = [256, 256];
    end
    if ~isnumeric(img) || ndims(img) ~= 3 || size(img,3) ~= 3
        error('preprocessImage:InvalidImage', 'Input must be a numeric RGB image.');
    end

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'toolboxfree'));

    result.original = myToUint8(img);
    result.classLabel = '';
    result.resized = myImresize(result.original, targetSize);
    result.hsv = myRgb2Hsv(result.resized);
    result.filtered = myGaussianFilter(result.resized, 1.5);

    filteredHsv = myRgb2Hsv(result.filtered);
    filteredHsv(:,:,3) = myContrastStretch(filteredHsv(:,:,3));
    result.enhanced = myHsv2Rgb(filteredHsv);

    enhancedHsv = myRgb2Hsv(result.enhanced);
    rawMask = enhancedHsv(:,:,1) >= 0.05 & enhancedHsv(:,:,1) <= 0.45 & ...
              enhancedHsv(:,:,2) >= 0.15 & enhancedHsv(:,:,3) >= 0.10;
    areaMask = myAreaOpen(rawMask, 500);
    result.cleanMask = myMorphClose(myFillHoles(areaMask), 3);
    result.segMask = rawMask;
    result.maskedOutput = myToUint8(result.enhanced) .* uint8(repmat(result.cleanMask, [1, 1, 3]));
end