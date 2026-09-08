% myFillHoles Flood-fills background reachable from the border; the rest are holes.
function filled = myFillHoles(mask)
    [rows, cols] = size(mask);
    outside = false(rows, cols);
    queue = zeros(numel(mask), 2);
    tail = 0;
    for row = 1:rows
        for col = [1, cols]
            if ~mask(row,col) && ~outside(row,col)
                tail = tail + 1; queue(tail,:) = [row, col]; outside(row,col) = true;
            end
        end
    end
    for col = 1:cols
        for row = [1, rows]
            if ~mask(row,col) && ~outside(row,col)
                tail = tail + 1; queue(tail,:) = [row, col]; outside(row,col) = true;
            end
        end
    end
    head = 1;
    while head <= tail
        current = queue(head,:); head = head + 1;
        for dr = [-1, 1]
            nr = current(1) + dr; nc = current(2);
            if nr >= 1 && nr <= rows && ~mask(nr,nc) && ~outside(nr,nc)
                tail = tail + 1; queue(tail,:) = [nr, nc]; outside(nr,nc) = true;
            end
            nr = current(1); nc = current(2) + dr;
            if nc >= 1 && nc <= cols && ~mask(nr,nc) && ~outside(nr,nc)
                tail = tail + 1; queue(tail,:) = [nr, nc]; outside(nr,nc) = true;
            end
        end
    end
    filled = mask | ~outside;
end
