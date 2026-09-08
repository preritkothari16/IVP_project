% myGaussianFilter Smooth each RGB channel using a sampled Gaussian and conv2.
function filtered = myGaussianFilter(img, sigma)
    radius = ceil(3 * sigma);
    x = -radius:radius;
    kernel1d = exp(-(x.^2) / (2 * sigma^2));
    kernel1d = kernel1d / sum(kernel1d);
    kernel = kernel1d' * kernel1d;
    filtered = zeros(size(img), 'uint8');
    for channel = 1:size(img, 3)
        filtered(:,:,channel) = uint8(min(255, max(0, ...
            round(conv2(double(img(:,:,channel)), kernel, 'same')))));
    end
end
