% function [vol_mcorr, yshift, xshift] = mcorrLocal_axial(fixed, reg, numBatch)
% 
%     numPoints   = size(fixed, 1);
%     numLines    = size(fixed, 2);
%     numFrames   = size(fixed, 3);   % For global motion correction
%      
%     sizeBatch = ceil(numLines/numBatch); 
%     vol_mcorr = reg;
%     
%     for I = 1:numFrames
%         for j=1:numBatch
%             
%            % Determine start and end index of each batch
%             startIdx = (j-1)*sizeBatch + 1;
%             endIdx = startIdx + sizeBatch - 1;
% 
%             [output] = dftregistration(fft2(abs(fixed(:, startIdx:endIdx, I))),...
%                 fft2(abs(reg(:, startIdx:endIdx, I))));
%             yshift(j) = output(3);
%             xshift(j) = output(4);
%             
%             vol_mcorr(:,startIdx:endIdx,I) = circshift(fft(ifft(reg(:, startIdx:endIdx, I))), [output(3), 0]); 
%         end
%     end
% end

function [vol_mcorr, dopu_mcorr, yshift_global] = mcorrLocal_axial(fixed, reg, reg_dopu, numBatch)
    numLines = size(fixed, 2);
    numFrames = size(fixed, 3);
    sizeBatch = ceil(numLines / numBatch);
    vol_mcorr = reg;
    yshift_global = zeros(numFrames, numBatch);
    
    for I = 1:numFrames
        yshift = zeros(1, numBatch); % Initialize for the current frame
        
        for j = 1:numBatch
            startIdx = (j-1)*sizeBatch + 1;
            endIdx = startIdx + sizeBatch - 1;
            midIdx(j) = (startIdx + endIdx)/2;
            
            output = dftregistration(fft2(abs(fixed(:, startIdx:endIdx, I))), ...
                                      fft2(abs(reg(:, startIdx:endIdx, I))));
            yshift(j) = output(3);
        end

        yshift_smooth = round(smooth(yshift, 0.4));
        yshift_interp = round(interp1(midIdx,yshift_smooth,1:numLines,'linear','extrap'))';

%         row_shift_all(:,k) = row_shift_interp;
        yshift_global(:,I) = yshift_interp;
      
    end

    for k = 1:numFrames
        for j = 1:numLines
            vol_mcorr(:,j,k) = circshift(reg(:,j,k),[yshift_global(j,k),0]);
            dopu_mcorr(:,j,k) = circshift(reg_dopu(:,j,k),[yshift_global(j,k),0]);
        end
    end
end