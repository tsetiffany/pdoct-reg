clc, clearvars, close all
tic

file_dir = "C:\Users\tiffa\Downloads\Untitled.nii.gz";
OCT = load('I:\LFOV_500x500\2025-03-17T16-12-31.573165\573165_avgOCT.mat');

var_name = fieldnames(OCT); % Get the variable name(s)
OCT3d = OCT.(var_name{1});

seg = niftiread(file_dir);
[depth,width,frames] = size(seg);

segILM = zeros(size(seg,2),size(seg,3));
segIPL = zeros(size(seg,2),size(seg,3));
segINL = zeros(size(seg,2),size(seg,3));

for i = 1:size(seg,2)
    for j = 1:size(seg,3)
        idx = find(seg(:,i,j));      
        if length(idx) == 3
            segILM(i,j) = idx(1);
            segIPL(i,j) = idx(2);
            segINL(i,j) = idx(3);
        else
            
        end
    end
end

svp_enface = zeros(size(OCT3d,2),size(OCT3d,3));
for i = 1:size(OCT3d,2)-1
    for j = 1:size(OCT3d,3)
        depth_start = segILM(i,j)+10;
        depth_end = segIPL(i,j);
        svp_enface(i,j) = squeeze(max(OCT3d(depth_start:depth_end,i,j),[],1));
    end
end
figure; imshow(imadjust(mat2gray(svp_enface)))

dvp_enface = zeros(size(OCT3d,2),size(OCT3d,3));
for i = 1:size(OCT3d,2)-1
    for j = 1:size(OCT3d,3)
        depth_start = segIPL(i,j);
        depth_end = segINL(i,j);
        dvp_enface(i,j) = squeeze(max(OCT3d(depth_start:depth_end,i,j),[],1));
    end
end
figure; imshow(imadjust(mat2gray(dvp_enface)))
figure; imagesc(imadjust(mat2gray(dvp_enface))),colormap(flipud(gray))

full_enface = zeros(size(OCT3d,2),size(OCT3d,3));
for i = 1:size(OCT3d,2)-1
    for j = 1:size(OCT3d,3)
        depth_start = segILM(i,j)+10;
        depth_end = segINL(i,j);
        full_enface(i,j) = squeeze(mean(OCT3d(depth_start:depth_end,i,j),1));
    end
end
figure; imshow(imadjust(mat2gray(full_enface)))
figure; imagesc(imadjust(mat2gray(imrotate(flipud(full_enface),-90)))),colormap(flipud(gray))
set(gcf,'Position',[0 0 size(OCT3d,2) size(OCT3d,3)]);
set(gca,'units','normalized','position',[0 0 1 1]);
axis off;
axis tight;
saveas(gcf,'H:\1. SFOV\Reg\invert_test\fixed_684773_octv_inv_mcorr.tif')
imwrite(imadjust(mat2gray(full_enface)),flipud(gray),'H:\1. SFOV\Reg\invert_test\fixed_684773_octv_inv_mcorr.tif')

for i=1:600
%     overlay = imfuse(imadjust(mat2gray(seg(:,:,i))),imadjust(mat2gray(OCT3d(:,:,i))));
    imshow(overlay)
    pause(0.01)
end