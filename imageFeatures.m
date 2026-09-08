function features = imageFeatures(img, mask)
if ~any(mask(:))
    features = zeros(1, 50);
    return;
end
hsv = rgb2hsv(img);
hHist = normalizedHistogram(hsv(:,:,1), mask, 16);
sHist = normalizedHistogram(hsv(:,:,2), mask, 8);
vHist = normalizedHistogram(hsv(:,:,3), mask, 4);

moments = zeros(1, 9);
for channel = 1:3
    values = double(img(:,:,channel));
    values = values(mask);
    first = (channel - 1) * 3;
    moments(first + 1:first + 3) = [mean(values), std(values), mySkewness(values)];
end
gray = 0.2989 * double(img(:,:,1)) + 0.5870 * double(img(:,:,2)) + 0.1140 * double(img(:,:,3));
texture = myGLCM(gray);
props = myRegionProps(mask);
shape = [props.Area, props.Perimeter, props.CentroidX, props.CentroidY, ...
    props.MajorAxisLength, props.MinorAxisLength, props.Eccentricity, props.Solidity, props.Extent];
features = [hHist, sHist, vHist, moments, texture, shape];
end

function histogram = normalizedHistogram(channel, mask, binCount)
edges = linspace(0, 1, binCount + 1);
histogram = histcounts(channel(mask), edges);
histogram = histogram / sum(histogram);
end