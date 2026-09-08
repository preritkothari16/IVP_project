% myMorphClose Performs binary dilation then erosion with a disk-shaped kernel.
function closed = myMorphClose(mask, radius)
    [x, y] = meshgrid(-radius:radius, -radius:radius);
    se = (x.^2 + y.^2) <= radius^2;
    dilated = conv2(double(mask), double(se), 'same') > 0;
    closed = conv2(double(dilated), double(se), 'same') == sum(se(:));
end
