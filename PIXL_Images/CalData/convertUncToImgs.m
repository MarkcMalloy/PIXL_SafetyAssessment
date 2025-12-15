
% TODO: Process all unc files in pwd and generate img using the following
% commands:
files = dir('*.unc'); % Get all .unc files in the current directory
for k = 1:length(files)
    fileName = files(k).name; % Get the file name
    [header, im, metaData, metaDataBin] = readAscImage(fileName, true); % Read the image
    I8 = uint8(255 * mat2gray(im))
    imwrite(I8,'NoObstacle_100mm_'+string(k)+'.png')
    % Process the image (e.g., save or display)
end