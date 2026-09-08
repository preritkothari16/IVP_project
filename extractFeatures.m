% extractFeatures Build color, texture, and shape features from processed images.
function featureTable = extractFeatures(preprocessedRoot)
    if nargin < 1 || isempty(preprocessedRoot)
        preprocessedRoot = uigetdir('.', 'Select Preprocessed Folder');
        if isequal(preprocessedRoot, 0)
            featureTable = table();
            fprintf('Cancelled.\n');
            return;
        end
    end
    if ~exist(preprocessedRoot, 'dir')
        error('extractFeatures:MissingFolder', 'Folder not found: %s', preprocessedRoot);
    end

    imds = imageDatastore(preprocessedRoot, 'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', 'FileExtensions', '.png');
    datastoreCount = numel(imds.Files);
    selected = false(datastoreCount, 1);
    for i = 1:datastoreCount
        selected(i) = ~isempty(regexp(imds.Files{i}, '_preprocessed\.png$', 'once'));
    end
    files = imds.Files(selected);
    labels = imds.Labels(selected);
    imageCount = numel(files);
    if imageCount == 0
        error('extractFeatures:NoImages', 'No *_preprocessed.png images found in %s.', preprocessedRoot);
    end

    featureCount = 50;
    featureMatrix = zeros(imageCount, featureCount);
    labelList = cell(imageCount, 1);
    fprintf('Extracting features from %d images...\n', imageCount);
    for i = 1:imageCount
        if mod(i, 25) == 1 || i == imageCount
            fprintf('  [%d/%d] %s\n', i, imageCount, files{i});
        end
        img = myToUint8(imread(files{i}));
        maskPath = regexprep(files{i}, '_preprocessed\.png$', '_mask.png');
        if exist(maskPath, 'file')
            mask = imread(maskPath) > 0;
            if ndims(mask) == 3, mask = mask(:,:,1); end
        else
            mask = any(img > 0, 3);
        end
        featureMatrix(i,:) = imageFeatures(img, mask);
        labelList{i} = char(labels(i));
    end

    names = cell(1, featureCount);
    for i = 1:featureCount, names{i} = sprintf('Feature%03d', i); end
    featureTable = array2table(featureMatrix, 'VariableNames', names);
    featureTable.Label = categorical(labelList);
    save(fullfile(preprocessedRoot, 'extractedFeatures.mat'), 'featureTable');
    writetable(featureTable, fullfile(preprocessedRoot, 'extractedFeatures.csv'));
    fprintf('Saved %d features per image to %s\n', featureCount, preprocessedRoot);
end

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
