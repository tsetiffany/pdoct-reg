function [vol_mcorr, yshift, xshift] = mcorrLocal_axial(fixed, reg, numBatch)

    numPoints   = size(fixed, 1);
    numLines    = size(fixed, 2);
    numFrames   = size(fixed, 3);   % For global motion correction
     
    sizeBatch = ceil(numLines/numBatch); 
    vol_mcorr = reg;
    
    for I = 1:numFrames
        for j=1:numBatch
            
           % Determine start and end index of each batch
            startIdx = (j-1)*sizeBatch + 1;
            endIdx = startIdx + sizeBatch - 1;

            [output] = dftregistration(fft2(abs(fixed(:, startIdx:endIdx, I))),...
                fft2(abs(reg(:, startIdx:endIdx, I))));
            yshift(I) = output(3);
            xshift(I) = output(4);
            
            vol_mcorr(:,startIdx:endIdx,I) = circshift(fft(ifft(reg(:, startIdx:endIdx, I))), [output(3), 0]); 
        end
    end
end