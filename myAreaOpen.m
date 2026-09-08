% myAreaOpen Removes 8-connected foreground components smaller than minPixels.
function cleaned = myAreaOpen(mask, minPixels)
    [labels, sizes] = myConnectedComponents(mask);
    cleaned = false(size(mask));
    for label = 1:numel(sizes)
        if sizes(label) >= minPixels
            cleaned(labels == label) = true;
        end
    end
end
