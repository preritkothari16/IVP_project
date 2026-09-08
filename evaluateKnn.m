% evaluateKnn Performs a stratified train/test split and k-NN using base MATLAB.
function results = evaluateKnn(featureTable, trainFraction, k)
    if nargin < 1 || isempty(featureTable)
        loaded = load('extractedFeatures.mat', 'featureTable');
        featureTable = loaded.featureTable;
    end
    if nargin < 2 || isempty(trainFraction), trainFraction = 0.75; end
    if nargin < 3 || isempty(k), k = 3; end
    if trainFraction <= 0 || trainFraction >= 1 || k < 1
        error('evaluateKnn:InvalidInput', 'trainFraction must be in (0,1) and k must be positive.');
    end

    features = featureTable{:, 1:end-1};
    labels = cellstr(featureTable.Label);
    classes = unique(labels);
    trainIndices = false(numel(labels), 1);
    rng(1);
    for classIndex = 1:numel(classes)
        members = find(strcmp(labels, classes{classIndex}));
        if numel(members) < 2, trainIndices(members) = true; continue; end
        count = max(1, min(numel(members) - 1, floor(trainFraction * numel(members))));
        order = randperm(numel(members));
        trainIndices(members(order(1:count))) = true;
    end
    testIndices = ~trainIndices;
    if ~any(testIndices)
        error('evaluateKnn:NoTestData', 'At least two samples per class are required for evaluation.');
    end

    trainX = features(trainIndices,:);
    testX = features(testIndices,:);
    trainY = labels(trainIndices);
    testY = labels(testIndices);
    averages = mean(trainX, 1);
    scales = std(trainX, 0, 1);
    scales(scales == 0) = 1;
    trainX = (trainX - averages) ./ scales;
    testX = (testX - averages) ./ scales;

    predictions = cell(numel(testY), 1);
    for sample = 1:numel(testY)
        distances = sum((trainX - testX(sample,:)).^2, 2);
        [~, order] = sort(distances, 'ascend');
        neighbors = trainY(order(1:min(k, numel(order))));
        votes = zeros(numel(classes), 1);
        for neighbor = 1:numel(neighbors)
            votes(strcmp(classes, neighbors{neighbor})) = votes(strcmp(classes, neighbors{neighbor})) + 1;
        end
        [~, winner] = max(votes);
        predictions{sample} = classes{winner};
    end

    confusion = zeros(numel(classes), numel(classes));
    for sample = 1:numel(testY)
        actual = find(strcmp(classes, testY{sample}));
        predicted = find(strcmp(classes, predictions{sample}));
        confusion(actual, predicted) = confusion(actual, predicted) + 1;
    end
    accuracy = sum(strcmp(testY, predictions)) / numel(testY);
    results = struct('Classes', {classes}, 'ConfusionMatrix', confusion, ...
        'Accuracy', accuracy, 'Predictions', {predictions}, 'Actual', {testY}, ...
        'TrainCount', sum(trainIndices), 'TestCount', sum(testIndices));
    save('classificationResults.mat', 'results');
    fprintf('k-NN accuracy: %.2f%% (%d train, %d test)\n', 100 * accuracy, ...
        results.TrainCount, results.TestCount);
    disp(array2table(confusion, 'VariableNames', classes, 'RowNames', classes));
end
