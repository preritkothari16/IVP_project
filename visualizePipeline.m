% visualizePipeline.m
% Purpose: Visualize each stage of the preprocessing pipeline for a single image.
% Usage:
%   visualizePipeline                -> opens file picker dialog
%   visualizePipeline(imgPath)       -> processes given image
%   visualizePipeline(imgPath, lbl)  -> processes with custom label

function visualizePipeline(imgPath, classLabel)
    if nargin < 1
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Image Files (*.jpg, *.png, *.bmp, *.tif)'}, 'Select an Image');
        if isequal(file, 0)
            fprintf('Cancelled.\n');
            return;
        end
        imgPath = fullfile(path, file);
    end

    if ischar(imgPath) || isstring(imgPath)
        if ~exist(imgPath, 'file')
            error('Image file not found: %s', imgPath);
        end
        img = imread(imgPath);
        displayName = char(imgPath);
        if nargin < 2
            parts = strsplit(fileparts(imgPath), filesep);
            classLabel = parts{end};
        end
    else
        img = imgPath;
        displayName = 'in-memory image';
        if nargin < 2
            classLabel = 'Sample';
        end
    end

    fprintf('Processing: %s\n', displayName);
    result = preprocessImage(img);

    figure('Name', 'Preprocessing Pipeline', 'Position', [100, 100, 1400, 900]);
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(sprintf('Pipeline: %s', classLabel), 'FontSize', 16, 'FontWeight', 'bold');

    nexttile; imshow(result.original);
    title('1. Original'); axis off;

    nexttile; imshow(result.resized);
    title('2. Resized (256x256)'); axis off;

    nexttile; imshow(result.filtered);
    title('3. Gaussian Filtered'); axis off;

    nexttile; imshow(result.enhanced);
    title('4. Contrast Enhanced'); axis off;

    nexttile; imshow(result.cleanMask);
    title('5. Segmentation Mask'); axis off; colormap(gca, 'gray');

    nexttile; imshow(result.maskedOutput);
    title('6. Final Output'); axis off;

    fprintf('Done.\n');
end
