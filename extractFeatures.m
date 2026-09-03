% extractFeatures.m
% Purpose: Stub / outline for feature extraction from preprocessed strawberry images.
%          This is NOT a complete implementation — it shows the structure and
%          key functions you would use for color + texture features.
% Inputs:  preprocessedRoot - Folder with preprocessed images (default: 'preprocessed')
% Outputs: featureTable - MATLAB table with one row per image, columns = features

function featureTable = extractFeatures(preprocessedRoot)
    % ============================================================
    % SETUP
    % ============================================================
    if nargin < 1 || isempty(preprocessedRoot)
        preprocessedRoot = 'preprocessed';
    end
    
    if ~exist(preprocessedRoot, 'dir')
        error('Preprocessed folder not found: %s. Run preprocessDataset.m first.', preprocessedRoot);
    end
    
    % Load preprocessed images with labels
    imds = imageDatastore(preprocessedRoot, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', ...
        'FileExtensions', '.png');
    
    numImages = imds.Count;
    fprintf('Extracting features from %d preprocessed images...\n', numImages);
    
    % Preallocate feature matrix (will grow dynamically if needed)
    % For now, we just outline the feature computation per image.
    
    % ============================================================
    % FEATURE EXTRACTION PER IMAGE (outline)
    % ============================================================
    % For each image, you would compute:
    
    featuresList = {};
    labelsList = {};
    
    for i = 1:numImages
        % Progress
        if mod(i, 50) == 1
            fprintf('  [%d/%d]\n', i, numImages);
        end
        
        % Read preprocessed image (already masked, background removed)
        img = readimage(imds, i);
        label = imds.Labels(i);
        
        % --- COLOR FEATURES ---
        % 1. HSV Color Histogram (most discriminative for disease)
        %    - Quantize H into 16 bins, S into 8 bins, V into 4 bins = 512 features
        %    - Normalize to sum to 1 (probability distribution)
        % hsvImg = rgb2hsv(img);
        % hHist = imhist(hsvImg(:,:,1), 16);  % Hue histogram
        % sHist = imhist(hsvImg(:,:,2), 8);   % Saturation histogram
        % vHist = imhist(hsvImg(:,:,3), 4);   % Value histogram
        % colorHist = [hHist, sHist, vHist] / sum([hHist, sHist, vHist]);
        
        % 2. RGB Color Moments (mean, std, skewness per channel = 9 features)
        % rgbMean = mean(img, [1,2]);
        % rgbStd  = std(double(img), 0, [1,2]);
        % rgbSkew = ... (compute skewness)
        % colorMoments = [rgbMean, rgbStd, rgbSkew];
        
        % --- TEXTURE FEATURES (GLCM - Gray Level Co-occurrence Matrix) ---
        % Convert to grayscale (use Value channel or luminance)
        % grayImg = rgb2gray(img);  % or use hsvImg(:,:,3) * 255
        % grayImg = im2uint8(grayImg);
        
        % GLCM parameters
        % offsets = [0 1; 1 1; 1 0; 1 -1];  % 4 directions (0, 45, 90, 135 deg)
        % glcm = graycomatrix(grayImg, 'Offset', offsets, 'NumLevels', 64, 'Symmetric', true);
        % % glcm is 64x64x1x4 - average over directions
        % glcmAvg = squeeze(mean(glcm, 4));
        
        % Extract Haralick texture properties
        % stats = graycoprops(glcmAvg, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
        % textureFeats = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
        
        % --- SHAPE / MORPHOLOGICAL FEATURES (if mask available) ---
        % If you saved the cleanMask from preprocessing, you could compute:
        % - Area, Perimeter, Circularity, Aspect Ratio, Solidity, Extent
        % - Major/Minor axis lengths, Eccentricity
        % regionProps = regionprops(mask, 'Area', 'Perimeter', 'Centroid', ...
        %                           'MajorAxisLength', 'MinorAxisLength', ...
        %                           'Eccentricity', 'Solidity', 'Extent');
        
        % --- COMBINE ALL FEATURES INTO ONE VECTOR ---
        % allFeatures = [colorHist, colorMoments, textureFeats, shapeFeats];
        % featuresList{end+1} = allFeatures;
        % labelsList{end+1} = label;
    end
    
    % ============================================================
    % POST-PROCESSING
    % ============================================================
    % Convert to table for easy export to classification
    % featureTable = array2table(cell2mat(featuresList'));
    % featureTable.Label = categorical(labelsList');
    
    % Save for later use
    % save('extractedFeatures.mat', 'featureTable');
    % writetable(featureTable, 'extractedFeatures.csv');
    
    fprintf('\nFeature extraction outline complete.\n');
    fprintf('Implement the commented sections above for real features.\n');
    fprintf('Then uncomment the table creation and save lines.\n');
    
    % Return empty table as placeholder
    featureTable = table();
end

% ============================================================
% QUICK REFERENCE: KEY MATLAB FUNCTIONS FOR FEATURES
% ============================================================
% Color:
%   rgb2hsv, rgb2gray, imhist, mean, std, skewness
%
% Texture (GLCM):
%   graycomatrix - Compute GLCM
%       'Offset'       - Pixel pair offsets [dx dy]
%       'NumLevels'    - Gray levels (e.g., 64)
%       'Symmetric'    - true for symmetric matrix
%   graycoprops - Extract properties from GLCM
%       'Contrast'     - Local variations
%       'Correlation'  - Linear dependency
%       'Energy'       - Uniformity (ASM)
%       'Homogeneity'  - Closeness of distribution
%       'Entropy'      - Randomness (requires custom)
%
% Shape:
%   regionprops - Measure region properties
%       'Area', 'Perimeter', 'Centroid', 'BoundingBox'
%       'MajorAxisLength', 'MinorAxisLength', 'Eccentricity'
%       'Solidity', 'Extent', 'Orientation'
%
% Dimensionality Reduction (later):
%   pca - Principal Component Analysis
%   fitcecoc - Multiclass SVM/ensemble classifier
%   fitctree - Decision tree
%   fitcknn  - k-Nearest Neighbors

% ============================================================
% EXAMPLE: MINIMAL WORKING COLOR HISTOGRAM (uncomment to use)
% ============================================================
% function histFeats = computeColorHistogram(img)
%     % img: RGB uint8 image (already preprocessed, masked)
%     hsvImg = rgb2hsv(img);
%     % Only compute on non-zero (masked) pixels
%     mask = any(img > 0, 3);
%     h = hsvImg(:,:,1);
%     s = hsvImg(:,:,2);
%     v = hsvImg(:,:,3);
%     hHist = imhist(h(mask), 16);
%     sHist = imhist(s(mask), 8);
%     vHist = imhist(v(mask), 4);
%     histFeats = [hHist, sHist, vHist] / sum([hHist, sHist, vHist]);
% end

% ============================================================
% EXAMPLE: MINIMAL WORKING GLCM TEXTURE (uncomment to use)
% ============================================================
% function texFeats = computeGLCMFeatures(grayImg)
%     % grayImg: uint8 grayscale (0-255)
%     offsets = [0 1; 1 1; 1 0; 1 -1];
%     glcm = graycomatrix(grayImg, 'Offset', offsets, 'NumLevels', 64, 'Symmetric', true);
%     glcmAvg = squeeze(mean(glcm, 4));
%     stats = graycoprops(glcmAvg, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
%     texFeats = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
% end

% --- Allow running as script ---
if ~isfunction('extractFeatures')
    extractFeatures('preprocessed');
end