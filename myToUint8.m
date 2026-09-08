% myToUint8 Convert numeric image data to uint8 without Image Processing Toolbox.
function out = myToUint8(img)
    if isa(img, 'uint8')
        out = img;
    elseif isinteger(img)
        out = uint8(round(double(img) / double(intmax(class(img))) * 255));
    else
        values = double(img);
        if isempty(values)
            out = uint8(values);
        elseif min(values(:)) >= 0 && max(values(:)) <= 1
            out = uint8(round(values * 255));
        else
            out = uint8(min(255, max(0, round(values))));
        end
    end
end
