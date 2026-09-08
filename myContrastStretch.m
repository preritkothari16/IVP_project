% myContrastStretch Linearly maps an array's minimum and maximum to [0, 1].
function out = myContrastStretch(in)
    low = min(in(:));
    high = max(in(:));
    if high > low
        out = (in - low) / (high - low);
    else
        out = in;
    end
end
