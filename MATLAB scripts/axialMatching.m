function output_filepath = axialMatching(filepath,numBatch)
    
%     [parent,~,~] = fileparts(filepath);
%     fixed_files = dir(fullfile(parent, '*.mat'));

    output_filepath = fullfile(filepath,"registered_mat_files");
    mkdir(output_filepath);
    
    % Get the list of .mat files in the directory
    files = dir(fullfile(filepath, '*.mat'));
    
    % Identify the fixed file (filename starts with 'fixed')
    fixed_file = files(contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'octv_mcorr', 'IgnoreCase', true)).name;
    
    % Load the fixed volume
    fixed_data = load(fullfile(filepath, fixed_file));
    fixed = fixed_data.fixed;
    
    % Filter the list of files to exclude the fixed file
    reg_oct_files = files(~contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'reg_vol', 'IgnoreCase', true) & contains({files.name}, 'octv_mcorr', 'IgnoreCase', true));
    reg_dopu_files = files(~contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'reg', 'IgnoreCase', true) & contains({files.name}, 'dopu_mcorr', 'IgnoreCase', true));

    for i=1:length(reg_oct_files)
        tic
        reg_oct_file = reg_oct_files(i).name;
        reg_dopu_file = reg_dopu_files(i).name;
        
        % Load the reg volume
        reg_oct_data = load(fullfile(filepath, reg_oct_file));
        reg_oct = reg_oct_data.reg;
    
        reg_dopu_data = load(fullfile(filepath, reg_dopu_file));
        reg_dopu = reg_dopu_data.reg_dopu;
    
        axmat = reg_oct;
        axmat_dopu = reg_dopu;
       
        [axmat_oct, axmat_dopu, yshift_global] = mcorrLocal_axial(fixed,reg_oct,reg_dopu,numBatch);
    
        disp("Axial registration completed in " + num2str(toc) + " s")
    
        % Save the final axmat file
        [~, reg_name, ~] = fileparts(reg_oct_file);
        output_file = fullfile(output_filepath, reg_name + "_axmat.mat");
        save(output_file, 'axmat_oct', '-v7.3');
    
        [~, reg_dopu_name, ~] = fileparts(reg_dopu_file);
        output_file_dopu = fullfile(output_filepath, reg_dopu_name + "_axmat.mat");
        save(output_file_dopu, 'axmat_dopu', '-v7.3');
    
        disp("Saved registered volume to " + output_file);
        
         for k=1:size(reg_oct,3)
             imshowpair(imadjust(mat2gray(fixed(:,:,k))),imadjust(mat2gray(abs(axmat_oct(:,:,k)))))
%              imshow(imadjust(mat2gray(axmat_oct(:,:,k))))
         end
    
        close all;

    end

    disp("Axial matching complete for all volumes");
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