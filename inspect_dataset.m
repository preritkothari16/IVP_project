% inspect_dataset.m
% Purpose: Inspect the dataset folder structure.
% Usage:
%   inspect_dataset              -> opens folder picker
%   inspect_dataset(datasetRoot) -> inspects given folder

function inspect_dataset(datasetRoot)
    if nargin < 1
        datasetRoot = uigetdir('.', 'Select Dataset Folder');
        if isequal(datasetRoot, 0)
            fprintf('Cancelled.\n');
            return;
        end
    end

    if ~exist(datasetRoot, 'dir')
        error('Dataset folder not found: %s', datasetRoot);
    end

    fprintf('=== Dataset Inspection ===\n');
    fprintf('Path: %s\n\n', datasetRoot);

    subfolders = dir(datasetRoot);
    classFolders = subfolders([subfolders.isdir]);
    classFolders = classFolders(~ismember({classFolders.name}, {'.', '..'}));

    numClasses = length(classFolders);
    fprintf('Found %d classes:\n\n', numClasses);

    classNames = {};
    imageCounts = zeros(numClasses, 1);
    sampleImages = cell(numClasses, 1);

    for i = 1:numClasses
        className = classFolders(i).name;
        classPath = fullfile(datasetRoot, className);

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

        if count > 0
            sampleImages{i} = imread(fullfile(classPath, imgFiles(1).name));
        end
    end

    fprintf('\nTotal: %d images\n', sum(imageCounts));

    if numClasses > 0
        figure('Name', 'Dataset Samples', 'Position', [100, 100, 1200, 800]);
        tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
        title('Sample Per Class', 'FontSize', 14, 'FontWeight', 'bold');

        for i = 1:numClasses
            nexttile;
            if ~isempty(sampleImages{i})
                imshow(sampleImages{i});
                title(classNames{i}, 'FontSize', 8, 'Interpreter', 'none');
                axis off;
            else
                text(0.5, 0.5, 'No image', 'HorizontalAlignment', 'center');
                title(classNames{i}, 'FontSize', 8);
            end
        end
    end
end
