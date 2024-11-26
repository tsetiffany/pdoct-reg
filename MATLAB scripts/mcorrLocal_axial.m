function [vol_mcorr, yshift, xshift] = mcorrLocal_axial(fixed, reg)

    numPoints   = size(fixed, 1);
    numLines    = size(fixed, 2);
    numFrames   = size(fixed, 3);   % For global motion correction
    
    kLinear         = linspace(-1,1,numPoints);
    kaxis           = repmat(kLinear',1,numLines);
    
    vol_mcorr = reg;
    
    for I = 1:numFrames
        % compare fixed and moving frame by frame
        [output] = dftregistration(fft2(abs(fixed(:,:,I))),...
            fft2(abs(reg(:,:,I))));
        yshift(I) = output(3);
        xshift(I) = output(4);

        % This line does the lateral shift
        % temp = ifft2(fft2(vol(:,:,I)).*cplx_shift);

        % Only do axial correction
%         vol_mcorr(:,:,I) = fft(ifft(reg(:,:,I)).*exp(2j.*(output(3)*(kaxis))));
        vol_mcorr(:,:,I) = circshift(fft(ifft(reg(:,:,I))), [output(3), 0]); 
    end
end