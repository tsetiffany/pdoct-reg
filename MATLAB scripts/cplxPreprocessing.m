% This script is for combining cplxdata from P and S channels to be fed
% into the registration algorithm. Phase difference is calculated and
% compensated before summing. Real and imaginary parts are separated and
% saved.

% load cplx A and B

for i=1:size(cplxData_A,3)
    OCT_P  = cplxData_A(:,:,i);  
    OCT_S  = cplxData_B(:,:,i);
    
    % phase correction 
    phi = repmat(angle(sum(OCT_S.*conj(OCT_P),1)), [size(OCT_S,1) 1]);
    OCT_Cplx = (OCT_P) + (OCT_S.*exp(-1j.*phi)); 
    CompCplx(:,:,i) = OCT_Cplx;
end

save('C:\Users\tiffa\Documents\1. Projects\3D Reg\EyeLiner-main\OCT_segmentation\Raw Data 939084\CompCplx.mat','CompCplx','-v7.3')

% real = real(CompCplx);
% im = imag(CompCplx);
% 
% save('C:\Users\tiffa\Documents\1. Projects\3D Reg\EyeLiner-main\OCT_segmentation\Raw Data 372322\real.mat','real','-v7.3')
% save('C:\Users\tiffa\Documents\1. Projects\3D Reg\EyeLiner-main\OCT_segmentation\Raw Data 372322\im.mat','im','-v7.3')