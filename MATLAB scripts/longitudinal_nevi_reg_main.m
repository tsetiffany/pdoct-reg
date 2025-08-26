% Longitudinal data reg
folder = 'H:\Longitudinal_Reg\06_JL';
name = '198355';

figure;imshow(imadjust(mat2gray(squeeze(mean(OCT(:,:,:))))))
figure;imshow(imadjust(mat2gray(squeeze(mean(OCT_t2(:,:,:))))))

%if flipped ud
OCT_t1 = OCT(:,end:-1:1,:);
DOPU_t1 = DOPU(:, end:-1:1, :);

%if flipped lr
OCT_t1 = OCT(:,:,end:-1:1);
DOPU_t1 = DOPU(:,:,end:-1:1);

figure;imshow(imadjust(mat2gray(squeeze(mean(OCT_t1(:,:,:))))))

enface_large = imadjust(mat2gray(squeeze(mean(OCT_t1(:,:,:)))));

h = imshow(enface_large, 'InitialMagnification', 'fit');

% Activate cropping rectangle
h_rect = imrect(gca, [100, 100, 600, 600]); % Initial position (x, y, width, height)

% Constrain aspect ratio to square
setFixedAspectRatioMode(h_rect, true);

% Wait for user to move/resize the rectangle and double-click or press Enter
position = wait(h_rect);  % Returns [x, y, width, height]

x_start = round(position(1));
y_start = round(position(2));
x_end = x_start + 599;
y_end = y_start + 599;

OCT = OCT_t1(:, y_start:y_end, x_start:x_end);
DOPU = DOPU_t1(:, y_start:y_end, x_start:x_end);

figure;imshow(imrotate(imadjust(mat2gray(squeeze(mean(OCT(:,:,:))))),90))

save(fullfile(folder,['fixed_',name,'_octv.mat']), 'OCT', '-v7.3');
save(fullfile(folder,['fixed_',name,'_dopu.mat']), 'DOPU', '-v7.3');

imwrite(imrotate(imadjust(mat2gray(squeeze(mean(OCT(:,:,:))))),90),fullfile(folder,['fixed_',name,'_enface_oct.tif']))

%% GO TO PYTHON REG
% do global mcorr and 3D registration

%% average en-face images
folder = 'H:\Longitudinal_Reg\04_DM';

enf_fixed = imadjust(mat2gray(squeeze(mean(fixed(:,:,:)))));
enf_reg = imadjust(mat2gray(squeeze(mean(reg(:,:,:)))));

avg_enf = (enf_fixed + enf_reg)./2;
figure;imshow(imadjust(mat2gray(avg_enf)))
imwrite(imrotate(fliplr(avg_enf),90),fullfile(folder,'outputs','averaged_enface.tif'));

enf_dopu_reg = reg_dopu;
f=1-(squeeze(min(enf_dopu_reg,[],1)));
imgF=imrotate(fliplr(f),90); 
imgF=mat2gray(imgF);
imInd = gray2ind(imgF,256);
img_RGB=ind2rgb(imInd,hot(256));

figure;imshow(img_RGB(21:580,21:580));colormap(hot); 
exportgraphics(gcf,fullfile(folder,'outputs','registered_dopu.tif'));

%% generate dopu composites and oct nifti for re-segmentation

cmap_dopu_r = load('cmap_dopu_r.mat').cmap_dopu_r;    %load the DOPU colormap
DOPU_Bscans_fixed = genDOPUComposite(fixed_dopu(:,26:end-25,:),fixed(:,26:end-25,:),cmap_dopu_r);
DOPU_Bscans_reg = genDOPUComposite(reg_dopu(:,26:end-25,:),reg(:,26:end-25,:),cmap_dopu_r);

disp('Saving DOPU composite...')
for i=1:size(fixed_dopu,3)
    imwrite(uint8(255*flipud(DOPU_Bscans_reg(:,:,:,i))),cmap_dopu_r,fullfile(folder,'outputs','dopu_comp_reg.tif'),'WriteMode','append');
    imwrite(uint8(255*flipud(DOPU_Bscans_fixed(:,:,:,i))),cmap_dopu_r,fullfile(folder,'outputs','dopu_comp_fixed.tif'),'WriteMode','append');
end

for i=1:size(reg,3)
    imadjusted_vol_reg(:,:,i) = imadjust(mat2gray(reg(:,:,i)));
    imadjusted_vol_fixed(:,:,i) = imadjust(mat2gray(fixed(:,:,i)));
end

niftiwrite(imadjusted_vol_reg,fullfile(folder,'outputs','reg_octv'),'Compressed',true)
niftiwrite(imadjusted_vol_fixed,fullfile(folder,'outputs','fixed_octv'),'Compressed',true)


niftiwrite(reg_dopu,fullfile(folder,'outputs','reg_dopu'),'Compressed',true)
niftiwrite(fixed_dopu,fullfile(folder,'outputs','fixed_dopu'),'Compressed',true)

