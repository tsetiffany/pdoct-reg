% Testing code for generating synthetic motion-free template using two
% en-face OCTA images with motion artifacts. This code combines the two
% images together.

reg1 = squeeze(reg_image(:,:,:));
figure; imshow(imadjust(mat2gray(reg1)))

reg2 = squeeze(reg_image(:,:,:));
figure; imshow(imadjust(mat2gray(reg2)));

new = zeros([size(reg1,1), size(reg1,2)]);
new(:,1:100) = reg2(:,1:100);
new(:,101:end) = reg1(:,101:end);

% % For the second part (from column 101 to the end), shift the pixels upwards by x
% vshift = -10;  % Specify the number of pixels to shift upwards8
% hshift = -3;
% shifted_part = circshift(reg1(:, 101:end), [vshift, hshift]);

% Copy the shifted part of reg1 into the new image
% new(:, 101:end) = shifted_part;

imshow(imadjust(mat2gray(new)))

new_uint8 = uint8(new * 255);  % Scale if the image is in the range [0, 1]

% Save the image as a TIFF file
imwrite(new_uint8, 'I:\26March25_Registration_CC\test\new_image.tif');

reg3 = double(squeeze(reg_image(:,:,:)));
imshow(imadjust(mat2gray(reg3)))

new2 = zeros([size(reg1,1), size(reg1,2)]);
new2(:,1:100) = reg2(:,1:100);
new2(:,101:end) = reg3(:,101:end);

imshow(imadjust(mat2gray(new2)))


% reg 2 has no motion on the left
reg4 = squeeze(reg_image(:,:,:));
new = zeros([size(reg1,1), size(reg1,2)]);
new(:,1:100) = reg2(:,1:100);
new(:,101:end) = reg4(:,101:end);

enface1 = zeros([size(fixed_730688_OCTA_enface_sum,1), size(fixed_730688_OCTA_enface_sum,2)]);
enface1(:,200:end) = fixed_730688_OCTA_enface_sum(:,200:end);

enface1 = zeros([size(fixed_730688_OCTA_enface_sum,1), size(fixed_730688_OCTA_enface_sum,2)]);
enface2(:,1:199) = x666513_OCTA_enface_sum(:,1:199);

figure; imshow(imadjust(mat2gray(fixed)))
figure; imshow(imadjust(mat2gray(moving)))
figure; imshow(imadjust(mat2gray(reg)))

fixed_new = zeros([size(fixed,1), size(fixed,2)]);
fixed_new(:,65:end) = fixed(:,65:end);
imshow(fixed_new)

moving_new = zeros([size(fixed,1), size(fixed,2)]);
moving_new(:,1:125) = moving(:,1:125);
imshow(moving_new)

fixed_uint8 = uint8(fixed_new * 255);
imwrite(fixed_uint8, 'I:\26March25_Registration_CC\April10\new_fixed.tif');

moving_uint8 = uint8(moving_new * 255);
imwrite(moving_uint8, 'I:\26March25_Registration_CC\April10\new_moving.tif');

synth_fixed = zeros([size(fixed,1), size(fixed,2)]);
synth_fixed(:,1:66) = new_reg(:,1:66)*3;
synth_fixed(:,67:125) = (new_reg(:,67:125)+ fixed(:,67:125)*2);
synth_fixed(:,126:end) = fixed(:,126:end)*3;
imshow(imadjust(mat2gray(synth_fixed)))

