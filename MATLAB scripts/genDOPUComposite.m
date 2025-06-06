function DOPU_Bscans = genDOPUComposite(DOPU,OCT,cmap_dopu_r)
    % DOPU combined B-scans with log scale avgOCT %
    % prepare matlab matrix for the Bscans in rgb

    [numPoints,numAlines,numBscans] = size(DOPU);
    DOPU_Bscans = zeros(numPoints, numAlines, 3, numBscans); %matrix for the rgb images (index with the frame number LAST!)

    % % prepare Tiff object
    % outputFileName = fullfile(loadloc,[fn_num,'_DOPU_Bscans_',num2str(kernel(1)),'_',num2str(kernel(2)),'.tif']);

    % loop over frames
    for ii=1:size(DOPU,3)

        % preparing the greyscale avgOCT luminance mask
        if ii==1 || ii==size(DOPU,3)
            G = flipud(squeeze(OCT(:,:,ii)));
        else
            % average one frame prior and after
            G = flipud(mean(OCT(:,:,ii-1:ii+1),3));
        end
        %for when there are zeros in the avgOCT (set them to around 13-30)
        G(G==0)=30;
        imgL=10*log10(G);

        imgL=mat2gray(imgL);
        imgL_test=imadjust(imgL,[0.25,0.8],[]); % perform some imadjustment
        % ******************* COMMENT OUT BEFORE RUNNING!!! **************
    %     figure;subplot(1,2,1);imagesc(imgL);colormap(gray);
    %     subplot(1,2,2);imagesc(imgL_test);colormap(gray);

        % convert the DOPU image to RGB then HSV
        imgD=flipud(DOPU(:,:,ii));
        colorMapForSaving = imresize(cmap_dopu_r,[256, 3], 'nearest'); % interpolate to 256 colour values (one for eevry number possible in the image)
        imgD = uint8(255.0 *mat2gray(imgD,[0.3,1])); %without mat2gray the images lose resolution and look like gray2ind, even though mat2gray on ingD doesn't seem to change teh values
        imgD_rgb=ind2rgb(floor(imgD),colorMapForSaving);
        imgD_hsv=rgb2hsv(imgD_rgb); %h=1, s=2, v=3

        % replace the V channel with the log image
        imgD_hsv(:,:,3)=imgL_test;
        imgD_rgb_r=hsv2rgb(imgD_hsv); % final combined B-scan (is rgb so a 3D matrix)

    %     %%%%%%%%%% COMMENT OUT WHEN RUNNING!
    %     figure;
    %     ax1=subplot(2,2,1);imshow(imgD);colormap(ax1,cmap_dopu_r);
    %     subplot(2,2,2);imshow(imgD_rgb);
    %     ax3=subplot(2,2,3);imagesc(imgL_test);colormap(ax3,gray);
    %     subplot(2,2,4);imshow(imgD_rgb_r);


        % save into a matrix
        DOPU_Bscans(:,:,:,ii) = imgD_rgb_r;

        % save as a tiff stack
    %     imwrite(imgD_rgb_r, outputFileName, 'WriteMode', 'append',  'Compression','none');

    %     fprintf('DOPU postprocess : %d\n', ii);
    end


end