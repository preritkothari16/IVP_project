% mySkewness Computes the standardized third central moment without Statistics Toolbox.
function value = mySkewness(values)
    values = double(values(:));
    deviation = values - mean(values);
    sigma = sqrt(mean(deviation.^2));
    if sigma == 0
        value = 0;
    else
        value = mean(deviation.^3) / sigma^3;
    end
end
