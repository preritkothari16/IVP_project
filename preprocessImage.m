% preprocessImage.m
% Purpose: Apply a complete preprocessing pipeline to a single strawberry leaf/fruit image.
% Input:   img - Input RGB image (MxNx3 uint8 or double)
% Output:  result - Struct containing all intermediate and final results:
%           .original        - Original input image
%           .resized         - Resized to 256x256
%           .hsv             - HSV version (for visualization)
%           .filtered        - Noise-reduced image
%           .enhanced        - Contrast-enhanced image
%           .segMask         - Binary segmentation mask (leaf/fruit vs background)
%           .cleanMask       - Cleaned mask (small noise removed, holes filled)
%           .maskedOutput    - Final image with background removed
%           .classLabel      - Optional: class label if provided

function result = preprocessImage(img, targetSize)
    % --- Input validation and defaults ---
    if nargin < 2
        targetSize = [256, 256];  % Default fixed size for all images
    end
    
    if ~isnumeric(img)
        error('Input must be a numeric image array.');
    end
    
    % Ensure RGB uint8
    if size(img, 3) ~= 3
        error('Expected RGB image (3 channels), got %d channels.', size(img, 3));
    end
    
    if ~isa(img, 'uint8')
        img = im2uint8(img);
    end
    
    % Initialize result struct
    result.original = img;
    result.classLabel = '';
    
    % ============================================================
    % STEP 1: RESIZE TO FIXED SIZE
    % WHY: Neural networks and traditional classifiers require
    % fixed-size inputs. 256x256 preserves detail while being
    % computationally manageable.
    % ============================================================
    result.resized = imresize(img, targetSize);
    
    % ============================================================
    % STEP 2: CONVERT TO HSV COLOR SPACE
    % WHY: Disease spots (brown, yellow, black lesions) often
    % have distinct Hue values that separate better from healthy
    % green tissue in HSV than in RGB. We keep RGB for final output
    % but use HSV for segmentation.
    % ============================================================
    result.hsv = rgb2hsv(result.resized);
    
    % ============================================================
    % STEP 3: NOISE REDUCTION
    % WHY: Camera sensor noise and compression artifacts can
    % create false edges and interfere with thresholding.
    % Gaussian smoothing preserves edges better than averaging.
    % sigma=1.5 is a good balance for 256x256 images.
    % ============================================================
    result.filtered = imgaussfilt(result.resized, 1.5);
    
    % ============================================================
    % STEP 4: CONTRAST ENHANCEMENT
    % WHY: Disease symptoms may be subtle (early yellowing,
    % faint spots). Enhancing contrast on the luminance channel
    % makes these more detectable without shifting colors.
    % We work in HSV: enhance the Value (V) channel.
    % ============================================================
    hsvFiltered = rgb2hsv(result.filtered);
    vChannel = hsvFiltered(:,:,3);           % Extract Value channel
    vEnhanced = imadjust(vChannel);          % Stretch contrast to full range
    % Alternative: vEnhanced = histeq(vChannel);  % Histogram equalization
    hsvFiltered(:,:,3) = vEnhanced;          % Put back
    result.enhanced = hsv2rgb(hsvFiltered);  % Convert back to RGB
    
    % ============================================================
    % STEP 5: SEGMENTATION - ISOLATE LEAF/FRUIT FROM BACKGROUND
    % WHY: Background (soil, pots, hands, shadows) adds noise to
    % features. We threshold in HSV space because:
    %   - Healthy strawberry leaves: Hue ~ 0.25-0.45 (green)
    %   - Diseased areas: Hue shifts toward yellow/brown (0.1-0.2)
    %   - Background: often very different hue/saturation
    % We create a mask for "plant-like" pixels (green + yellow/brown).
    % ============================================================
    hsvEnhanced = rgb2hsv(result.enhanced);
    hueChannel = hsvEnhanced(:,:,1);
    satChannel = hsvEnhanced(:,:,2);
    valChannel = hsvEnhanced(:,:,3);
    
    % Define HSV ranges for strawberry plant tissue
    % Hue: green (0.25-0.45) + yellow/brown diseased (0.05-0.2)
    % Saturation: > 0.15 (exclude gray/white background)
    % Value: > 0.1 (exclude very dark shadows)
    hueMask = (hueChannel >= 0.05 & hueChannel <= 0.45);
    satMask = (satChannel >= 0.15);
    valMask = (valChannel >= 0.1);
    
    % Combine: pixel is plant if it passes all three
    rawMask = hueMask & satMask & valMask;
    
    % ============================================================
    % STEP 6: MASK CLEANUP
    % WHY: Raw thresholding produces speckles (noise) and holes
    % (specular highlights, veins). Clean up with morphology.
    % ============================================================
    % Remove small connected components (< 500 pixels = noise)
    cleanedMask = bwareaopen(rawMask, 500);
    
    % Fill holes inside the leaf/fruit region
    cleanedMask = imfill(cleanedMask, 'holes');
    
    % Optional: morphological closing to smooth boundary
    se = strel('disk', 3);
    cleanedMask = imclose(cleanedMask, se);
    
    result.segMask = rawMask;
    result.cleanMask = cleanedMask;
    
    % ============================================================
    % STEP 7: APPLY MASK - EXTRACT PLANT REGION
    % WHY: Zero out background so downstream features (color,
    % texture) only describe the plant tissue.
    % ============================================================
    % Replicate mask to 3 channels for RGB multiplication
    mask3 = repmat(cleanedMask, [1, 1, 3]);
    result.maskedOutput = result.enhanced .* uint8(mask3);
    
    % Set background to white (optional, for visualization)
    % result.maskedOutput(repmat(~cleanedMask, [1,1,3])) = 255;
end

% --- Local helper functions (could be separate files if preferred) ---

% No additional local functions needed; all steps are inline above
% with clear section comments. This keeps the pipeline readable
% as a single top-to-bottom flow.