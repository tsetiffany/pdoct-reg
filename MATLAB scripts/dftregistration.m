function [output] = dftregistration(buf1ft,buf2ft)

[nr,nc]=size(buf2ft);
Nr = ifftshift(-fix(nr/2):ceil(nr/2)-1);
Nc = ifftshift(-fix(nc/2):ceil(nc/2)-1);
CC = ifft2(buf1ft.*conj(buf2ft));
CCabs = abs(CC);
[row_shift, col_shift] = find(CCabs == max(CCabs(:)));
% CCmax = CC(row_shift,col_shift)*nr*nc;
%     Now change shifts so that they represent relative shifts and not indices
row_shift = Nr(row_shift);
col_shift = Nc(col_shift);


CCmax = max(CCabs(:));
error = 0;
diffphase = 0;

output=[error,CCmax,row_shift,col_shift];
end