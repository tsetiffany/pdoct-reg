 
filepath = "I:\OCTA_Mohammad_0311\LFOV_test\ProcdData\OCTA_Mohammad_0311";
vol_files = dir(fullfile(filepath, '*.mat'));

for idx = 1:length(vol_files)
    
    % Get the full path to the current .mat file
    current_file = fullfile(filepath, vol_files(idx).name);
    
    % Load the data from the file
    file_data = load(current_file);
    var_name = fieldnames(file_data); % Get the variable name(s)
    vol = file_data.(var_name{1});
    
    OCT_mcorr = vol;
    numFrames = size(OCT_mcorr,3);
    startFrame = round(numFrames/2);
    axialShift = zeros([1 numFrames]);

    for k = startFrame-1:-1:1
         %%% Every 'for' loop, reference frame will be the next frame %%%
        [output] = dftregistration(fft2(imadjust(mat2gray(OCT_mcorr(:, :, k+1)))),...
            fft2(imadjust(mat2gray(OCT_mcorr(:, :, k)))));
        axialShift(k) = output(3);

        OCT_mcorr(:, :, k)  = circshift(OCT_mcorr(:,:,k),  [round(output(3)) 0]);
    end
    
    for k = startFrame+1:numFrames
         %%% Every 'for' loop, reference frame will be the previous frame %%%
        [output] = dftregistration(fft2(imadjust(mat2gray(OCT_mcorr(:, :, k-1)))),...
            fft2(imadjust(mat2gray(OCT_mcorr(:, :, k)))));
        axialShift(k) = output(3);
        OCT_mcorr(:, :, k)  = circshift(OCT_mcorr(:, :, k),  [round(output(3)) 0]);
    end

    figure;
    plot(axialShift)
    title('Axial Motion Estimation')
    xlim([1 500]); ylim([-15 15]);
    xlabel('Frame Index')
    ylabel('Axial Shifts (Pixel)')

    [~, name, ~] = fileparts(vol_files(idx).name); % Extract filename without extension
    new_filename = fullfile(filepath, name + "_mcorr.mat");
    
    % Save the processed data to the new file
    save(new_filename, 'OCT_mcorr', '-v7.3');
    disp(['global motion correction ', num2str(idx), '/', num2str(length(vol_files))]);

end


    