% This code removes motion artifacts in OCTA volumes based on white strips.
% This step should be run after registration and prior to averaging.

% Define the filepath where the .mat files are located
filepath = "I:\26March25_Registration_CC\output\registered_mat_files"; % Update this to your directory path

% Get the list of .mat files in the directory
files = dir(fullfile(filepath, '*.mat'));

% Identify the fixed file (filename starts with 'fixed')
fixed_file = files(contains({files.name}, 'fixed', 'IgnoreCase', true)).name;

% Load the fixed volume
fixed_data = load(fullfile(filepath, fixed_file));
fixed = fixed_data.fixed;
fixed_mrmov = imrotate3(fixed,90,[0 1 0]);

fixed_mrmov_enface = imadjust(mat2gray(squeeze(mean(fixed_mrmov(:,:,:)))));
col_means = mean(fixed_mrmov_enface,1,'omitnan');
cols_to_nan = col_means >= 0.9; % OCTA motion manifests as white lines, set threshold to determine motion
fixed_mrmov_enface(:,cols_to_nan) = 0;
fixed_mrmov(:,:,cols_to_nan) = 0;

fixed_mrmov(fixed_mrmov == 0) = NaN;

% Save the mrmov file
[~, fixed_name, ~] = fileparts(fixed_file);
output_file = fullfile(filepath, fixed_name + "_mrmov.mat");
save(output_file, 'fixed_mrmov', '-v7.3');

disp("Saved fixed motion removed volume to " + output_file);

% Filter the list of files to exclude the fixed file
axmat_files = files(~contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'axmat', 'IgnoreCase', true));

for i=1:length(axmat_files)
    tic
    axmat_file = axmat_files(1).name;
    
    % Load the reg volume
    axmat_data = load(fullfile(filepath, axmat_file));
    axmat = axmat_data.axmat;
    
    mrmov = imrotate3(axmat,90,[0 1 0]);

    mrmov_enface = imadjust(mat2gray(squeeze(mean(mrmov(:,:,:)))));
    mrmov_enface(mrmov_enface == 0) = NaN;
    col_means = mean(mrmov_enface,1,'omitnan');
    cols_to_nan = col_means >= 0.9; % OCTA motion manifests as white lines, set threshold to determine motion
%     mrmov_enface(:,cols_to_nan) = 0;
    mrmov(:,:,cols_to_nan) = 0;

    mrmov(mrmov == 0) = NaN;

    % Save the mrmov file
    [~, axmat_name, ~] = fileparts(axmat_file);
    output_file = fullfile(filepath, axmat_name + "_mrmov.mat");
    save(output_file, 'mrmov', '-v7.3');

    disp("Saved motion removed volume to " + output_file);
    
end





