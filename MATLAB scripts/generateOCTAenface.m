% This code requires manual intervention for finding depth values
% corresponding to SVP and DVP. Corrects for artifacts using notch filter
% and CLAHE (adapthisteq).

imshow(imadjust(mat2gray(OCTA_tcorr_sub_1(:,:,1))))

top = 220;

for i=top:300
    imshow(imadjust(mat2gray(squeeze(OCTA_tcorr_sub_1(i,:,:)))))
    title(i)
    pause()
end

en1 = imadjust(mat2gray(squeeze(mean(OCTA_tcorr_sub_1(top:239,:,:)))));
imshow(en1)
en2 = imadjust(mat2gray(squeeze(mean(OCTA_tcorr_sub_1(234:245,:,:)))));
imshow(en2)
en3 = en1+en2;
imshow(en3)

bottom = 245;

figure; imshow(imadjust(mat2gray(squeeze(mean(OCTA_tcorr_sub_1(top:bottom,:,:))))))

% enf_ini = mat2gray(squeeze(max(abs(OCTA_tcorr_sub_1(top:bottom,:,:)),[],1)));
enf_ini = en3;
enf_hist2 = adapthisteq(enf_ini,"NumTiles",[25 25],"ClipLimit",0.002,"Distribution","uniform");
enf_enh2 = imcenhance(enf_hist2);
% enf_filt2 = imadjust(medfilt2(enf_enh2,[3 3]));
figure; imshow(enf_enh2);

save('I:\26March25_Registration_CC\fixed_730688_OCTA_enface_sum.mat','enf_enh2','-v7.3')
imwrite(enf_enh2,'I:\26March25_Registration_CC\fixed_730688_OCTA_enface_sum.tif')

function imout = imcenhance(img)

    % imout = adapthisteq(mat2gray(img));
    imout = (mat2gray(img));
%     s = size(img,1);
%     imout = imresize(imout, [2*s 2*s]);
    % imout = imresize(imout, 2); % This scales both dimensions proportionally
    imout = notchfilter(imout);
    % imout = imadjust(mat2gray(imout));
    % imshow(imout)

end

function [filt] = notchfilter(img)
    imgfft = fftshift(fft2(img));
    imgfft2 = imgfft;
    
    size(img,1);
    xg1=1:size(img,1);
    xg2=1:size(img,1);
    ag = 1;
    bg1 = size(img,1)/2;
    bg2=size(img,1)/2;
    
    cg1 = 1.4;
    cg2 = 1;
    fg = exp(-(xg1-bg1).^2./(2*cg1^2));
    fg1 = exp(-(xg2-bg2).^2./(2*cg2^2));
    fg = abs(fg-1);
    
    ffg = repmat(fg,[size(img,1) 1]);
    ffg1 = repmat(fg1',[1 size(img,1)]);
    
    ffg2 = ffg+ffg1;
    ffg2(ffg2>1) = 1;
    fffg = imgfft2.*ffg2';
    filt = abs(ifft2(fffg));

end