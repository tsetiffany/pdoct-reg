% Define the filepath where the .mat files are located
filepath = "I:\26March25_Registration_CC\output\registered_mat_files"; % Update this to your directory path

% Get the list of .mat files in the directory
files = dir(fullfile(filepath, '*.mat'));

% Identify the fixed file (filename starts with 'fixed')
fixed_file = files(contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'mrmov', 'IgnoreCase', true)).name;

% Load the fixed volume
fixed_data = load(fullfile(filepath, fixed_file));
fixed = fixed_data.fixed_mrmov;
fixed(fixed == 0) = NaN;

% Filter the list of files to exclude the fixed file
reg_files = files(contains({files.name}, 'axmat_mrmov', 'IgnoreCase', true));

% Initialize the super sum volume with the fixed volume
all_volumes = zeros([size(fixed), length(reg_files) + 1]);
all_volumes(:,:,:,1) = fixed;

% Loop through each axially matched file and add it to the super sum
for i = 1:length(reg_files)
    % Load the current axmat file
    reg_data = load(fullfile(filepath, reg_files(i).name));
    mrmov = reg_data.mrmov;
    mrmov(mrmov == 1) = NaN;
    
    all_volumes(:,:,:,i+1) = mrmov;
    % Add the current axmat volume to the super sum
%     super_sum = sum(cat(4, super_sum, axmat), 4,'omitnan'); % Sum while ignoring NaN
end

sum_vol = sum(all_volumes, 4, 'omitnan');
count_vol = sum(~isnan(all_volumes), 4);
averaged_vol = sum_vol./count_vol;

enface = imadjust(mat2gray(squeeze(mean(averaged_vol(:,:,:)))));
figure;imshow(enface(25:end-20,25:end-20))
imwrite(flipud(imrotate(enface(25:end-20,25:end-20),90)), fullfile(filepath,'\registered_enface.tif'))

for i=1:size(averaged_vol,3)
    averaged_vol_test(:,i,:) = imhistmatch(imadjust(mat2gray(squeeze(averaged_vol(:,i,:)))),imadjust(mat2gray(squeeze(averaged_vol(:,150,:)))));
    disp(i)
end

figure;
for i=1:size(averaged_vol,3)
    imshow(imadjust(mat2gray((abs(averaged_vol(:,:,i))))))
    pause(0.01)
%     imwrite(uint8(255* imadjust(mat2gray(abs(super_sum(:,:,i))))),fullfile(filepath, 'FINAL_OCT_reg_avg_volume.tif'),'WriteMode','append');
%     imwrite(uint8(255* imadjust(mat2gray(abs(fixed(:,:,i))))),fullfile(filepath, 'fixed_volume.tif'),'WriteMode','append');
end

% Save the resulting super sum volume
output_filename = fullfile(filepath, 'FINAL_OCT_reg_avg_volume.mat');
save(output_filename, 'super_sum', '-v7.3');

disp('Super sum volume created and saved.');