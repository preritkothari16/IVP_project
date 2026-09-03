% preprocessDataset.m
% Purpose: Batch-process all images in the dataset through the preprocessing pipeline.
% Inputs:  datasetRoot  - Path to input dataset (default: 'union_dataset')
%          outputRoot   - Path to save preprocessed images (default: 'preprocessed')
%          targetSize   - Resize dimensions [H W] (default: [256 256])
% Outputs: Saves preprocessed images to outputRoot/<class_name>/ preserving labels.
%          Prints progress every 20 images. Skips bad images with try/catch.

function preprocessDataset(datasetRoot, outputRoot, targetSize)
    % --- Defaults ---
    if nargin < 1 || isempty(datasetRoot)
        datasetRoot = 'union_dataset';
    end
    if nargin < 2 || isempty(outputRoot)
        outputRoot = 'preprocessed';
    end
    if nargin < 3 || isempty(targetSize)
        targetSize = [256, 256];
    end
    
    fprintf('=== Batch Preprocessing Started ===\n');
    fprintf('Input dataset:  %s\n', datasetRoot);
    fprintf('Output folder:  %s\n', outputRoot);
    fprintf('Target size:    %dx%d\n\n', targetSize(1), targetSize(2));
    
    % Verify input exists
    if ~exist(datasetRoot, 'dir')
        error('Dataset folder not found: %s', datasetRoot);
    end
    
    % Create output root if needed
    if ~exist(outputRoot, 'dir')
        mkdir(outputRoot);
        fprintf('Created output folder: %s\n', outputRoot);
    end
    
    % Use imageDatastore for efficient loading with labels
    imds = imageDatastore(datasetRoot, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', ...
        'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'});
    
    % Print class distribution
    fprintf('Class distribution:\n');
    labelCounts = countEachLabel(imds);
    disp(labelCounts);
    fprintf('\n');
    
    totalImages = imds.Count;
    fprintf('Total images to process: %d\n\n', totalImages);
    
    % Get all file paths and labels upfront for progress tracking
    allFiles = imds.Files;
    allLabels = imds.Labels;
    uniqueClasses = unique(allLabels);
    
    % Create output subfolders for each class
    for c = 1:length(uniqueClasses)
        className = uniqueClasses{c};
        classOutDir = fullfile(outputRoot, className);
        if ~exist(classOutDir, 'dir')
            mkdir(classOutDir);
        end
    end
    
    % --- Processing loop ---
    processedCount = 0;
    skippedCount = 0;
    errorCount = 0;
    
    for i = 1:totalImages
        imgPath = allFiles{i};
        classLabel = allLabels(i);
        
        % Progress print every 20 images
        if mod(i, 20) == 1 || i == totalImages
            fprintf('[%d/%d] Processing: %s (%s)\n', i, totalImages, classLabel, imgPath);
        end
        
        % --- Try/Catch block: one bad image won't crash the batch ---
        try
            % Read image
            img = imread(imgPath);
            
            % Ensure RGB
            if size(img, 3) == 1
                img = cat(3, img, img, img);  % Grayscale to RGB
            elseif size(img, 3) == 4
                img = img(:,:,1:3);  % Drop alpha channel
            end
            
            % Run preprocessing pipeline
            result = preprocessImage(img, targetSize);
            
            % Construct output filename (preserve original name)
            [~, baseName, ext] = fileparts(imgPath);
            outFileName = [baseName, '_preprocessed', '.png'];  % Save as PNG (lossless)
            outPath = fullfile(outputRoot, classLabel, outFileName);
            
            % Save preprocessed image
            imwrite(result.maskedOutput, outPath);
            
            processedCount = processedCount + 1;
            
        catch ME
            % Log error but continue with next image
            fprintf('  !! ERROR processing %s: %s\n', imgPath, ME.message);
            errorCount = errorCount + 1;
            
            % Optional: save error log
            % errorLog = [errorLog; {imgPath, ME.message}];
        end
    end
    
    % --- Summary ---
    fprintf('\n=== Batch Preprocessing Complete ===\n');
    fprintf('Successfully processed: %d\n', processedCount);
    fprintf('Skipped (errors):       %d\n', errorCount);
    fprintf('Output location:        %s\n', outputRoot);
    
    % Verify output
    if exist(outputRoot, 'dir')
        outFolders = dir(outputRoot);
        outFolders = outFolders([outFolders.isdir]);
        outFolders = outFolders(~ismember({outFolders.name}, {'.', '..'}));
        fprintf('Output classes created: %d\n', length(outFolders));
        for k = 1:length(outFolders)
            classPath = fullfile(outputRoot, outFolders(k).name);
            outFiles = dir(fullfile(classPath, '*.png'));
            fprintf('  %s: %d images\n', outFolders(k).name, length(outFiles));
        end
    end
end

% --- Allow running as script with defaults ---
if ~isfunction('preprocessDataset')
    preprocessDataset('union_dataset', 'preprocessed', [256, 256]);
end