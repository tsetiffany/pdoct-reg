%% Post-processing for 3D OCT & DOPU Registration
% Author: Tiffany Tse
% Last edit: June 6, 2025
% This code takes the output files from Python code and performs the
% following post-processing steps: axial matching, multi-volume averaging,
% visualization (DOPU composite, en-face generation, get RPE segmentation,
% etc.)

%% COMPLETE REGISTRATION

filepath = 'H:\PS391\OD\Reg\outputs'; % Update this to Python output directory
numBatch = 60;

% Axial matching
output_filepath = axialMatching(filepath,numBatch);

% Averaging
[averaged_oct, averaged_dopu] = averageVolumes(output_filepath);
[depth, numAscans, numBscans] = size(averaged_dopu); 

% GENERATE DOPU COMPOSITE

cmap_dopu_r = load('cmap_dopu_r.mat').cmap_dopu_r;    %load the DOPU colormap
% DOPU_Bscans = genDOPUComposite(averaged_dopu(:,26:end-25,:),averaged_oct(:,26:end-25,:),cmap_dopu_r);
DOPU_Bscans = genDOPUComposite(averaged_dopu(:,:,:),averaged_oct(:,:,:),cmap_dopu_r);

disp('Saving DOPU composite...')
for i=1:size(averaged_dopu,3)
%     imshow(flipud(DOPU_Bscans(:,:,:,i))); colormap(cmap_dopu_r)
    imwrite(uint8(255*flipud(DOPU_Bscans(:,:,:,i))),cmap_dopu_r,fullfile(output_filepath,'FINAL_DOPU_COMP_reg_avg_volume.tif'),'WriteMode','append');
end

% save(fullfile(output_filepath,'FINAL_DOPU_COMP_reg_avg_volume'), 'DOPU_Bscans', '-v7.3');

%% DOPU ENFACE IMAGE GENERATION

DOPU_filt=averaged_dopu;
% DOPU_filt(DOPU_filt>0.95) = 1; % threshold the DOPU

f=1-(squeeze(min(DOPU_filt,[],1)));
imgF=imrotate(fliplr(f),90); 
imgF=mat2gray(imgF);
imInd = gray2ind(imgF,256);
img_RGB=ind2rgb(imInd,hot(256));

figure;imshow(img_RGB(:,:,:));colormap(hot); 
exportgraphics(gcf,fullfile(output_filepath,'\fixed_registered_dopu.tif'));

%% Save registered OCT en-face

figure;imshow(imadjust(mat2gray(imrotate(fliplr(squeeze(mean(averaged_oct(:,:,:)))),90))));
exportgraphics(gcf,fullfile(output_filepath,'\fixed_registered_oct.tif'));

%% OPTIONAL: RPE SEGMENTATION
% the image looks the same with or without segmentation line

for i=1:numBscans
    
    currentSlice = averaged_dopu(:,:,i);
    
    thresholdValue = max(currentSlice(:)) - 0.01;
    binarySlice = currentSlice < thresholdValue;    
    binarySlice = bwareaopen(binarySlice, 25);  % Remove small objects
    
    for j = 1:numAscans
        
        idx = find(binarySlice(:,j));  % get all nonzero indices along the y-axis
            
            if ~isempty(idx)
                segRPE(j,i) = idx(1);      % first nonzero
            else
                segRPE(j,i) = 1;         % NaN if no segmentation found
            end
    end
end

% Project en-face
segRPEvol = zeros(depth, numAscans, numBscans);

segRPE(segRPE<1) = 1; % fix interpolation outside of [1 depth]
segRPE(segRPE>depth) = depth;

for i = 1:numBscans 
    for j = 1:numAscans
        rowRPE = round(segRPE(j,i));  % Get the row index of the seg line
        if rowRPE >= 1 && rowRPE <= depth
            segRPEvol(rowRPE,j,i) = 1;  % Set that point to 1 in the 3D mask
        end
    end
end

for i=50:550
    overlay = imfuse(averaged_dopu(:,25:end-25,i),segRPEvol(:,25:end-25,i));
    imshow(overlay)
    title(i)
    pause(0.01)
end

% Enface visualization
for i=1:600 
    for j=1:numBscans
        enface(i,j) = squeeze(min(averaged_dopu(round(segRPE(j,i)):depth,j,i),[],1));
    end
end          

%% SAVING FOR VISUALIZATION (PPT VIDEO ONLY)
cmap_dopu_r = load('cmap_dopu_r.mat').cmap_dopu_r;    %load the DOPU colormap

fixed_DOPU_Bscans = genDOPUComposite(fixed_dopu(:,36:end-35,:),fixed(:,36:end-35,:),cmap_dopu_r);
reg_DOPU_Bscans = genDOPUComposite(axmat_dopu(:,14:520,:),axmat_oct(:,14:520,:),cmap_dopu_r);
output_filepath = 'H:\For Jiwon\OPTH_RD_2026\melanoma1\outputs\registered_mat_files';
for i=1:size(axmat_dopu,3) % use every other frame to make the size consistent with 2FA processed data / shorter video. crop out noise slices, typically 20 frames
% %  imshow(flipud(reg_DOPU_Bscans(:,:,:,399))); colormap(cmap_dopu_r)
%     imwrite(uint8(255*reg_DOPU_Bscans(:,:,:,i)),cmap_dopu_r,fullfile(output_filepath,'t2_FINAL_DOPU_COMP_reg_volume.tif'),'WriteMode','append');
    imwrite(uint8(255*fixed_DOPU_Bscans(:,:,:,i)),cmap_dopu_r,fullfile(output_filepath,'t3_DOPU_COMP_reg_volume.tif'),'WriteMode','append');
% %     imwrite(uint8(255*flipud(DOPU_Bscans(:,:,:,i))),cmap_dopu_r,fullfile(output_filepath,'FINAL_CROPPED_DOPU_COMP_reg_avg_volume.tif'),'WriteMode','append');
% %     imwrite(uint8(255* imadjust(mat2gray(averaged_oct(:,:,i)))),fullfile(output_filepath,'FINAL_CROPPED_OCT_reg_avg_volume.tif'),'WriteMode','append');
%     imwrite(uint8(255* imadjust(mat2gray(flipud(axmat_oct(:,14:520,i))))),fullfile(output_filepath,'t2_FINAL_OCT_reg_volume.tif'),'WriteMode','append');
    imwrite(uint8(255* imadjust(mat2gray(flipud(fixed(:,14:520,i))))),fullfile(output_filepath,'t3_FINAL_OCT_reg_volume.tif'),'WriteMode','append');
end






