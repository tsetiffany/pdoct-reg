function [averaged_oct, averaged_dopu] = averageVolumes(filepath)
    
    % filepath = "I:\PS438_OS\Reg\outputs\registered_mat_files"; % Update this to your directory path
    
    % Get the list of .mat files in the directory
    files = dir(fullfile(filepath, '*.mat'));
    
    % Identify the fixed file (filename starts with 'fixed')
    fixed_file = files(contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'octv', 'IgnoreCase', true)).name;
    fixed_dopu_file = files(contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'dopu', 'IgnoreCase', true)).name;
    
    % Load the fixed volume
    fixed_data = load(fullfile(filepath, fixed_file));
    fixed = fixed_data.fixed;
    
    fixed_dopu_data = load(fullfile(filepath, fixed_dopu_file));
    fixed_dopu = fixed_dopu_data.fixed_dopu;
    
    
    % Filter the list of files to exclude the fixed file
    reg_oct_files = files(~contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'octv_mcorr_axmat', 'IgnoreCase', true));
    reg_dopu_files = files(~contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'dopu_mcorr_axmat', 'IgnoreCase', true));
    
    % Initialize the super sum volume with the fixed volume
    super_sum_oct = fixed;
    
    super_sum_dopu = fixed_dopu;
    
    % Loop through each axially matched file and add it to the super sum
    for i = 1:length(reg_oct_files)
        % Load the current axmat file
        reg_oct_data = load(fullfile(filepath, reg_oct_files(i).name));
        reg_dopu_data = load(fullfile(filepath, reg_dopu_files(i).name));
    
        axmat_oct = reg_oct_data.axmat_oct;
        axmat_dopu = reg_dopu_data.axmat_dopu;
        % Add the current axmat volume to the super sum
        super_sum_oct = sum(cat(4, super_sum_oct, axmat_oct), 4,'omitnan'); % Sum while ignoring NaN
        super_sum_dopu = sum(cat(4, super_sum_dopu, axmat_dopu), 4,'omitnan');
    end
    
    averaged_oct = super_sum_oct./(1+length(reg_oct_files));
    averaged_dopu = super_sum_dopu./(1+length(reg_oct_files));
    
    enface = imadjust(mat2gray(squeeze(mean(averaged_oct(:,:,:)))));
    figure;imshow(enface(25:end-20,25:end-20))
    imwrite(flipud(imrotate(enface(25:end-20,25:end-20),90)), fullfile(filepath,'\registered_enface.tif'))
    
    % for i=1:size(averaged_vol,3)
    %     averaged_vol_test(:,i,:) = imhistmatch(imadjust(mat2gray(squeeze(averaged_vol(:,i,:)))),imadjust(mat2gray(squeeze(averaged_vol(:,150,:)))));
    %     disp(i)
    % end
    
    disp('Saving .tif stacks...');
    figure;
    for i=1:size(averaged_oct,3)
        imshow(imadjust(mat2gray((abs(averaged_oct(:,:,i))))))
        pause(0.01)
        imwrite(uint8(255* imadjust(mat2gray(abs(averaged_oct(:,:,i))))),fullfile(filepath, 'FINAL_OCT_reg_avg_volume.tif'),'WriteMode','append');
        imwrite(uint8(255* imadjust(mat2gray(abs(averaged_dopu(:,:,i))))),fullfile(filepath, 'FINAL_DOPU_reg_avg_volume.tif'),'WriteMode','append');
        imwrite(uint8(255* imadjust(mat2gray(abs(fixed(:,:,i))))),fullfile(filepath, 'fixed_oct_volume.tif'),'WriteMode','append');
        imwrite(uint8(255* imadjust(mat2gray(abs(fixed_dopu(:,:,i))))),fullfile(filepath, 'fixed_dopu_volume.tif'),'WriteMode','append');
    end
    
    disp('Saving averaged_oct and averaged_dopu volumes...');
    % Save the resulting super sum volume
    output_filename = fullfile(filepath, 'FINAL_OCT_reg_avg_volume.mat');
    save(output_filename, 'averaged_oct', '-v7.3');
    output_dopu_filename = fullfile(filepath, 'FINAL_DOPU_reg_avg_volume.mat');
    save(output_dopu_filename, 'averaged_dopu', '-v7.3');
    
    
    disp('Super sum volume created and saved.');
end
