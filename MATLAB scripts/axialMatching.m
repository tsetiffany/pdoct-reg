
% Define the filepath where the .mat files are located
filepath = "H:\OCT_MJ_MFOV\Reg\outputs\registered mat files"; % Update this to your directory path

% Get the list of .mat files in the directory
files = dir(fullfile(filepath, '*.mat'));

% Identify the fixed file (filename starts with 'fixed')
fixed_file = files(contains({files.name}, 'fixed', 'IgnoreCase', true)).name;

% Load the fixed volume
fixed_data = load(fullfile(filepath, fixed_file));
fixed = fixed_data.fixed;

% Filter the list of files to exclude the fixed file
reg_files = files(~contains({files.name}, 'fixed', 'IgnoreCase', true));

for i=1:length(reg_files)
    tic
    reg_file = reg_files(i).name;
    
    % Load the reg volume
    reg_data = load(fullfile(filepath, reg_file));
    reg = reg_data.reg;

    usfac = 1; % upsampling factor
    axmat = reg;
    
    %     figure;
    %     subplot(2,2,1); imagesc(mean(abs(vol_mcorr),3)); title('Unaligned averaged Bscan'); xlabel("A-scans"); ylabel("Depth"); set(gca, FontSize=15)
    
    % Iterate axial correction until max error is below 0.25 pixels
    ii = 0;
    yshift_global = 1;
    while max(abs(yshift_global)) > 0.25 % Iterates until maximum axial shift is < 0.25 px
        ii = ii + 1;
        [axmat, yshift_global, xshift_global] = mcorrLocal_axial(fixed,reg);
        if ii > 10
            break
        end
    end
       disp("Axial registration completed in " + num2str(ii) + " iterations and " + num2str(toc) + " s")

       % Save the final axmat file
       [~, reg_name, ~] = fileparts(reg_file);
       output_file = fullfile(filepath, reg_name + "_axmat.mat");
       save(output_file, 'axmat', '-v7.3');

       disp("Saved registered volume to " + output_file);
      
end

% for ii = 1:size(reg_axmat,3)
%     img1 = imadjust(mat2gray(squeeze(fixed(:,2:end-2,ii))));
%     img2 = imadjust(mat2gray(squeeze(reg_axmat(:,2:end-2,ii))));
%     subplot(1,2,1)
%     imagesc(imadjust(mat2gray(img1))); colormap("gray")
%     subplot(1,2,2)
%     imagesc(imadjust(mat2gray(img2))); colormap("gray")
%     pause(0.03)
% end
% 
% %     subtracting images to check registration; if well registered, there
% %     should be little to no residue signal after subtraction
% for ii = 1:size(reg_axmat,3)
%     img1 = ((20.*log10(squeeze(fixed(10:end-10,2:end-2,ii)))));
%     img2 = ((20.*log10(squeeze(reg(10:end-10,2:end-2,ii)))));
%     img3 = ((20.*log10(squeeze(reg_axmat(10:end-10,2:end-2,ii)))));
%     subplot(1,2,1)
%     imagesc(((img2-img1))); colormap("gray")
%     subplot(1,2,2)
%     imagesc(((img3-img1))); colormap("gray")
%     pause(0.03)
% end