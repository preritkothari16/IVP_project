% inspect_dataset.m
% Purpose: Inspect the strawberry disease dataset folder structure.
% Inputs:  datasetRoot - path to the dataset folder (default: 'union_dataset')
% Outputs: Prints class names and image counts, displays one sample per class.

function inspect_dataset(datasetRoot)
    if nargin < 1
        datasetRoot = 'union_dataset';
    end
    
    fprintf('=== Dataset Inspection ===\n');
    fprintf('Dataset root: %s\n\n', datasetRoot);
    
    % Get all subfolders (each represents a class)
    if ~exist(datasetRoot, 'dir')
        error('Dataset folder not found: %s', datasetRoot);
    end
    
    subfolders = dir(datasetRoot);
    classFolders = subfolders([subfolders.isdir]);
    % Remove '.' and '..'
    classFolders = classFolders(~ismember({classFolders.name}, {'.', '..'}));
    
    numClasses = length(classFolders);
    fprintf('Found %d classes:\n\n', numClasses);
    
    % Store info for display
    classNames = {};
    imageCounts = zeros(numClasses, 1);
    sampleImages = cell(numClasses, 1);
    
    for i = 1:numClasses
        className = classFolders(i).name;
        classPath = fullfile(datasetRoot, className);
        
        % Count image files
        imgFiles = dir(fullfile(classPath, '*.jpg'));
        imgFiles = [imgFiles; dir(fullfile(classPath, '*.jpeg'))];
        imgFiles = [imgFiles; dir(fullfile(classPath, '*.png'))];
        imgFiles = [imgFiles; dir(fullfile(classPath, '*.bmp'))];
        imgFiles = [imgFiles; dir(fullfile(classPath, '*.tif'))];
        imgFiles = [imgFiles; dir(fullfile(classPath, '*.tiff'))];
        
        count = length(imgFiles);
        classNames{i} = className;
        imageCounts(i) = count;
        
        fprintf('  [%2d] %-35s : %4d images\n', i, className, count);
        
        % Load first image for display
        if count > 0
            firstImgPath = fullfile(classPath, imgFiles(1).name);
            sampleImages{i} = imread(firstImgPath);
        end
    end
    
    fprintf('\nTotal images: %d\n', sum(imageCounts));
    
    % Display sample images in a tiled layout
    if numClasses > 0
        figure('Name', 'Dataset Samples - One per Class', 'Position', [100, 100, 1200, 800]);
        tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
        title('Sample Images from Each Class', 'FontSize', 16, 'FontWeight', 'bold');
        
        for i = 1:numClasses
            nexttile;
            if ~isempty(sampleImages{i})
                imshow(sampleImages{i});
                title(classNames{i}, 'FontSize', 9, 'Interpreter', 'none');
                axis off;
            else
                text(0.5, 0.5, 'No image', 'HorizontalAlignment', 'center');
                title(classNames{i}, 'FontSize', 9);
            end
        end
    end
    
    % Return data for potential further use
    if nargout > 0
        varargout{1} = classNames;
        if nargout > 1
            varargout{2} = imageCounts;
            if nargout > 2
                varargout{3} = sampleImages;
            end
        end
    end
end

% Allow running as script
if ~isfunction('inspect_dataset')
    inspect_dataset('union_dataset');
end