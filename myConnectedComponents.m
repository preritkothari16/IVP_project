% myConnectedComponents Label 8-connected true regions with iterative flood fill.
function [labels, sizes] = myConnectedComponents(mask)
    [rows, cols] = size(mask);
    labels = zeros(rows, cols);
    sizes = zeros(0, 1);
    stack = zeros(numel(mask), 2);
    label = 0;
    for row = 1:rows
        for col = 1:cols
            if ~mask(row, col) || labels(row, col) ~= 0
                continue;
            end
            label = label + 1;
            top = 1;
            stack(1,:) = [row, col];
            labels(row, col) = label;
            count = 0;
            while top > 0
                current = stack(top,:);
                top = top - 1;
                count = count + 1;
                for dr = -1:1
                    for dc = -1:1
                        nr = current(1) + dr;
                        nc = current(2) + dc;
                        if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols && ...
                                mask(nr,nc) && labels(nr,nc) == 0
                            labels(nr,nc) = label;
                            top = top + 1;
                            stack(top,:) = [nr, nc];
                        end
                    end
                end
            end
            sizes(label,1) = count;
        end
    end
end
