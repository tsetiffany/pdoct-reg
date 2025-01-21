% Define the filepath where the .mat files are located
filepath = "H:\LFOV\Reg\outputs\registered_mat_files"; % Update this to your directory path

% Get the list of .mat files in the directory
files = dir(fullfile(filepath, '*.mat'));

% Identify the fixed file (filename starts with 'fixed')
fixed_file = files(contains({files.name}, 'fixed', 'IgnoreCase', true)).name;

% Load the fixed volume
fixed_data = load(fullfile(filepath, fixed_file));
fixed = fixed_data.fixed;
fixed(fixed == 0) = NaN;

% Filter the list of files to exclude the fixed file
reg_files = files(contains({files.name}, 'axmat', 'IgnoreCase', true));

% Initialize the super sum volume with the fixed volume
super_sum = fixed;

% Loop through each axially matched file and add it to the super sum
for i = 1:length(reg_files)
    % Load the current axmat file
    reg_data = load(fullfile(filepath, reg_files(i).name));
    axmat = reg_data.axmat;
    axmat(axmat == 0) = NaN;
    
    % Add the current axmat volume to the super sum
    super_sum = sum(cat(4, super_sum, axmat), 4,'omitnan'); % Sum while ignoring NaN
end

averaged_vol = super_sum./6;
enface = imadjust(mat2gray(squeeze(mean(abs(averaged_vol(:,:,:))))));
figure;imshow(enface(25:end-20,25:end-20))
imwrite(flipud(imrotate(enface(25:end-20,25:end-20),90)), "H:\LFOV\Reg\outputs\registered_mat_files\registered_enface.tif")

figure;
for i=1:600
    imshow(imadjust(mat2gray((abs(super_sum(:,:,i))))))
    imwrite(uint8(255* imadjust(mat2gray(abs(super_sum(:,:,i))))),fullfile(filepath, 'FINAL_OCT_reg_avg_volume.tif'),'WriteMode','append');
    imwrite(uint8(255* imadjust(mat2gray(abs(fixed(:,:,i))))),fullfile(filepath, 'fixed_volume.tif'),'WriteMode','append');
end

% Save the resulting super sum volume
output_filename = fullfile(filepath, 'FINAL_OCT_reg_avg_volume.mat');
save(output_filename, 'super_sum', '-v7.3');

disp('Super sum volume created and saved.');