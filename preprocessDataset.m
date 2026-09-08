% preprocessDataset.m
% Purpose: Batch-process all images through the preprocessing pipeline.
% Usage:
%   preprocessDataset                         -> opens folder pickers
%   preprocessDataset(inputDir)               -> uses given input, picks output
%   preprocessDataset(inputDir, outputDir, targetSize) -> uses supplied settings

function preprocessDataset(datasetRoot, outputRoot, targetSize)
    if nargin < 1
        datasetRoot = uigetdir('.', 'Select Dataset Folder');
        if isequal(datasetRoot, 0)
            fprintf('Cancelled.\n');
            return;
        end
    end
    if nargin < 2
        outputRoot = uigetdir('.', 'Select Output Folder');
        if isequal(outputRoot, 0)
            fprintf('Cancelled.\n');
            return;
        end
    end
    if nargin < 3 || isempty(targetSize)
        targetSize = [256, 256];
    end

    if ~exist(datasetRoot, 'dir')
        error('Dataset folder not found: %s', datasetRoot);
    end
    if ~exist(outputRoot, 'dir')
        mkdir(outputRoot);
    end

    fprintf('=== Batch Preprocessing ===\n');
    fprintf('Input:  %s\n', datasetRoot);
    fprintf('Output: %s\n\n', outputRoot);

    imds = imageDatastore(datasetRoot, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', ...
        'FileExtensions', {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'});

    fprintf('Classes:\n');
    uniqueClasses0 = unique(imds.Labels);
    for c = 1:length(uniqueClasses0)
        cnt = sum(imds.Labels == uniqueClasses0(c));
        fprintf('  %-35s : %d images\n', char(uniqueClasses0(c)), cnt);
    end

    totalImages = numel(imds.Files);
    fprintf('Total images: %d\n\n', totalImages);

    allFiles = imds.Files;
    allLabels = imds.Labels;
    uniqueClasses = unique(allLabels);

    for c = 1:length(uniqueClasses)
        className = char(uniqueClasses(c));
        classOutDir = fullfile(outputRoot, className);
        if ~exist(classOutDir, 'dir')
            mkdir(classOutDir);
        end
    end

    processedCount = 0;
    errorCount = 0;

    for i = 1:totalImages
        imgPath = allFiles{i};
        classLabel = char(allLabels(i));

        if mod(i, 20) == 1 || i == totalImages
            fprintf('[%d/%d] %s\n', i, totalImages, imgPath);
        end

        try
            img = imread(imgPath);
            if size(img, 3) == 1
                img = cat(3, img, img, img);
            elseif size(img, 3) == 4
                img = img(:,:,1:3);
            end

            result = preprocessImage(img, targetSize);
            [~, baseName] = fileparts(imgPath);
            outPath = fullfile(outputRoot, classLabel, [baseName '_preprocessed.png']);
            imwrite(result.maskedOutput, outPath);
            imwrite(uint8(result.cleanMask) * 255, ...
                fullfile(outputRoot, classLabel, [baseName '_mask.png']));
            processedCount = processedCount + 1;

        catch ME
            fprintf('  ERROR: %s - %s\n', imgPath, ME.message);
            errorCount = errorCount + 1;
        end
    end

    fprintf('\n=== Complete ===\n');
    fprintf('Processed: %d | Errors: %d\n', processedCount, errorCount);
    fprintf('Output: %s\n', outputRoot);
end
