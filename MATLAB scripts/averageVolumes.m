function [averaged_oct, averaged_dopu] = averageVolumes(filepath)
    
    [parent,~,~] = fileparts(filepath);
    % Get the list of .mat files in the directory
    files = dir(fullfile(filepath, '*.mat'));
    fixed_files = dir(fullfile(parent, '*.mat'));
    
    % Identify the fixed file (filename starts with 'fixed')
    fixed_file = fixed_files(contains({fixed_files.name}, 'fixed', 'IgnoreCase', true) & contains({fixed_files.name}, 'octv', 'IgnoreCase', true)).name;
    fixed_dopu_file = fixed_files(contains({fixed_files.name}, 'fixed', 'IgnoreCase', true) & contains({fixed_files.name}, 'dopu', 'IgnoreCase', true));
    
    use_dopu = ~isempty(fixed_dopu_file);

    if ~use_dopu
        disp("No DOPU fixed file detected - skipping DOPU averaging.");
        averaged_dopu = [];
    end

    % Load the fixed volume
    fixed_data = load(fullfile(parent, fixed_file));
    fixed = fixed_data.fixed;
    reg_oct_files = files(~contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'octv_mcorr_axmat', 'IgnoreCase', true));
    super_sum_oct = fixed;

    if use_dopu
        fixed_dopu_data = load(fullfile(parent, fixed_dopu_file.name));
        fixed_dopu = fixed_dopu_data.fixed_dopu;
        reg_dopu_files = files(~contains({files.name}, 'fixed', 'IgnoreCase', true) & contains({files.name}, 'dopu_mcorr_axmat', 'IgnoreCase', true));
        super_sum_dopu = fixed_dopu;
    end  
      
    disp('Generating super sum...');
    % Loop through each axially matched file and add it to the super sum
    for i = 1:length(reg_oct_files)
        
        reg_oct_data = load(fullfile(filepath, reg_oct_files(i).name));   
        axmat_oct = reg_oct_data.axmat_oct;
        super_sum_oct = sum(cat(4, super_sum_oct, axmat_oct), 4,'omitnan'); % Sum while ignoring NaN

        if use_dopu
            reg_dopu_data = load(fullfile(filepath, reg_dopu_files(i).name));
            axmat_dopu = reg_dopu_data.axmat_dopu;
            super_sum_dopu = sum(cat(4, super_sum_dopu, axmat_dopu), 4,'omitnan');
        end
    end
    
    averaged_oct = super_sum_oct./(1+length(reg_oct_files));

    if use_dopu
        averaged_dopu = super_sum_dopu./(1+length(reg_oct_files));
    end
   
    figure;imshow(flipud(imrotate(imadjust(mat2gray(squeeze(mean(averaged_oct(:,26:end-25,26:end-25))))),90))) % maybe remove flipud for longitudinal!!!!!!!!!
    imwrite(flipud(imrotate(imadjust(mat2gray(squeeze(mean(averaged_oct(:,26:end-25,26:end-25))))),90)), fullfile(filepath,'\registered_enface.tif'))

    disp('Saving .tif stacks...');
    
    for i=26:size(averaged_oct,3)-25
%         imshow(imadjust(mat2gray((abs(averaged_oct(:,:,i))))))
        imwrite(uint8(255* imadjust(mat2gray(abs(averaged_oct(:,26:end-25,i))))),fullfile(filepath, 'FINAL_OCT_reg_avg_volume.tif'),'WriteMode','append');
        imwrite(uint8(255* imadjust(mat2gray(abs(fixed(:,26:end-25,i))))),fullfile(filepath, 'fixed_oct_volume.tif'),'WriteMode','append');
        if use_dopu
            imwrite(uint8(255* imadjust(mat2gray(abs(averaged_dopu(:,26:end-25,i))))),fullfile(filepath, 'FINAL_DOPU_reg_avg_volume.tif'),'WriteMode','append');
            imwrite(uint8(255* imadjust(mat2gray(abs(fixed_dopu(:,26:end-25,i))))),fullfile(filepath, 'fixed_dopu_volume.tif'),'WriteMode','append');
        end
    end
    
    disp('Saving averaged_oct and averaged_dopu volumes...');
    averaged_oct = averaged_oct(:,26:end-25,26:end-25);
    output_filename = fullfile(filepath, 'FINAL_OCT_reg_avg_volume.mat');
    save(output_filename, 'averaged_oct', '-v7.3');
    
    if use_dopu
        averaged_dopu = averaged_dopu(:,26:end-25,26:end-25);
        output_dopu_filename = fullfile(filepath, 'FINAL_DOPU_reg_avg_volume.mat');
        save(output_dopu_filename, 'averaged_dopu', '-v7.3');
    end
    
    disp('Organizing directory...');
    src_fixed = fullfile(parent,fixed_file);
    dst_fixed = fullfile(filepath,fixed_file);
    movefile(src_fixed,dst_fixed);

    if use_dopu
        src_dopu = fullfile(parent, fixed_dopu_file);
        dst_dopu = fullfile(filepath,fixed_dopu_file);
        movefile(src_dopu,dst_dopu);
    end

    disp('Super sum volume created and saved.');
end
