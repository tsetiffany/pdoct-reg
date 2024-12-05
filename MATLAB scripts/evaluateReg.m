
filepath = "H:\OCT_MJ_MFOV\Reg\outputs_withGMC\registered_mat_files";
files = dir(fullfile(filepath, '*.mat'));

% Identify the fixed file (filename starts with 'fixed')
fixed_file = files(contains({files.name}, 'fixed', 'IgnoreCase', true)).name;

% Load the fixed volume
fixed_data = load(fullfile(filepath, fixed_file));
fixed = fixed_data.fixed;
fixed(fixed == 0) = NaN;

% Filter the list of files to exclude the fixed file
reg_files = files(contains({files.name}, 'axmat', 'IgnoreCase', true));
avged_file = files(contains({files.name}, 'avg', 'IgnoreCase', true)).name;
avged_data = load(fullfile(filepath, avged_file));
avged = avged_data.super_sum;

% Compute CNR of final averaged image and the reference @ centre B-scan

[SNR_fixed, CNR_fixed]  = computeSNRandCNR(244,664,227,526,100,30,imadjust(mat2gray(fixed(:,:,end/2))));
[SNR_reg, CNR_reg]  = computeSNRandCNR(244,664,227,526,100,30,imadjust(mat2gray(abs(avged(:,:,end/2)))));

% Compute SSIM between each registered image with the reference
for i=1:length(reg_files)
    tic
    reg_file = reg_files(i).name;
    
    % Load the reg volume
    reg_data = load(fullfile(filepath, reg_file));
    reg = reg_data.axmat;

    SSIM(i) = multissim3(mat2gray(fixed),mat2gray(abs(reg)));

end

mean_SSIM = mean(SSIM);
std_SSIM = std(SSIM);

% Display
fprintf('CNR_fixed = %.4f dB\n', CNR_fixed);
fprintf('CNR_reg = %.4f dB\n', CNR_reg);
fprintf('SNR_fixed = %.4f dB\n', SNR_fixed);
fprintf('SNR_reg = %.4f dB\n', SNR_reg);
fprintf('CNR improvement of %.4f dB\n', CNR_reg-CNR_fixed);
fprintf('SNR improvement of %.4f dB\n', SNR_reg-SNR_fixed);
% fprintf('Mean 3D MS-SSIM: %.4f ± %.4f\n', mean_SSIM, std_SSIM);

function [SNR, CNR] = computeSNRandCNR(sigX,sigY,bgX,bgY,w,h,image)
    % Define coordinates for signal and background regions
    signal_coords = [sigX sigY w h]; % [x, y, width, height] for signal ROI
    background_coords = [bgX bgY w h]; % [x, y, width, height] for background ROI

    % Extract the signal and background regions
    signal_region = image(signal_coords(2):(signal_coords(2)+signal_coords(4)-1), ...
        signal_coords(1):(signal_coords(1)+signal_coords(3)-1));
    background_region = image(background_coords(2):(background_coords(2)+background_coords(4)-1), ...
        background_coords(1):(background_coords(1)+background_coords(3)-1));

    % Compute mean intensity for signal and background
    mu_signal = mean(signal_region(:));
    mu_background = mean(background_region(:));

    % Compute standard deviation for background
    sigma_background = std(background_region(:));

    % Compute standard deviation for signal (for SNR)
    sigma_signal = std(signal_region(:));

    % Compute CNR
    CNR = 20*log10(abs(mu_signal - mu_background) / sigma_background);

    % Compute SNR
    SNR = 20 * log10(mu_signal / sigma_signal);

end
