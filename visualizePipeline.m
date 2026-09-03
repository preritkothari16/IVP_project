% visualizePipeline.m
% Purpose: Visualize each stage of the preprocessing pipeline for a single image.
% Inputs:  imgPath - Path to an image file (or pass image array directly)
%          classLabel - Optional string for title
% Outputs: Displays a 2x3 tiled figure showing the pipeline stages.

function visualizePipeline(imgPath, classLabel)
    % Handle input: file path or image array
    if ischar(imgPath) || isstring(imgPath)
        if ~exist(imgPath, 'file')
            error('Image file not found: %s', imgPath);
        end
        img = imread(imgPath);
        if nargin < 2
            % Try to extract class name from parent folder
            classLabel = fileparts(fileparts(imgPath));
            classLabel = classLabel(end);
        end
    else
        img = imgPath;
        if nargin < 2
            classLabel = 'Sample';
        end
    end
    
    % Run preprocessing pipeline
    fprintf('Processing image for visualization...\n');
    result = preprocessImage(img);
    
    % Create figure with 2x3 layout
    figure('Name', 'Preprocessing Pipeline Visualization', 'Position', [100, 100, 1400, 900]);
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Title for the whole figure
    titleStr = sprintf('Preprocessing Pipeline: %s', classLabel);
    title(titleStr, 'FontSize', 18, 'FontWeight', 'bold');
    
    % --- Tile 1: Original ---
    nexttile;
    imshow(result.original);
    title('1. Original Image', 'FontSize', 12, 'FontWeight', 'bold');
    axis off;
    
    % --- Tile 2: Resized ---
    nexttile;
    imshow(result.resized);
    title('2. Resized (256x256)', 'FontSize', 12, 'FontWeight', 'bold');
    axis off;
    
    % --- Tile 3: Noise Filtered ---
    nexttile;
    imshow(result.filtered);
    title('3. Gaussian Filtered (\sigma=1.5)', 'FontSize', 12, 'FontWeight', 'bold');
    axis off;
    
    % --- Tile 4: Contrast Enhanced ---
    nexttile;
    imshow(result.enhanced);
    title('4. Contrast Enhanced (imadjust on V)', 'FontSize', 12, 'FontWeight', 'bold');
    axis off;
    
    % --- Tile 5: Segmentation Mask ---
    nexttile;
    imshow(result.cleanMask);
    title('5. Cleaned Segmentation Mask', 'FontSize', 12, 'FontWeight', 'bold');
    axis off;
    colormap(gca, 'gray');
    
    % --- Tile 6: Final Masked Output ---
    nexttile;
    imshow(result.maskedOutput);
    title('6. Final Output (Background Removed)', 'FontSize', 12, 'FontWeight', 'bold');
    axis off;
    
    % Add a colorbar for the mask if desired
    % (Not needed for binary mask)
    
    fprintf('Visualization complete. Close figure to continue.\n');
end

% --- Demo mode: run on a sample image if called as script ---
if ~isfunction('visualizePipeline')
    % Find first image in dataset
    datasetRoot = 'union_dataset';
    subfolders = dir(datasetRoot);
    classFolders = subfolders([subfolders.isdir]);
    classFolders = classFolders(~ismember({classFolders.name}, {'.', '..'}));
    
    if length(classFolders) > 0
        firstClass = classFolders(1).name;
        imgFiles = dir(fullfile(datasetRoot, firstClass, '*.jpg'));
        if length(imgFiles) > 0
            samplePath = fullfile(datasetRoot, firstClass, imgFiles(1).name);
            visualizePipeline(samplePath, firstClass);
        end
    end
end