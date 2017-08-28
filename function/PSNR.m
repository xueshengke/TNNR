function [ erec, psnr ] = PSNR( X_full, X_rec, missing )
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, April 2017.
% Contact information: see readme.txt.
%
% Hu et al. (2013) TNNR paper, IEEE Transactions on PAMI.
% First written by debingzhang, Zhejiang Universiy, November 2012.
%--------------------------------------------------------------------------
%     compute PSNR and reconstruction error for the recovered image and
%     original image
% 
%     Inputs:
%         X_full           --- original image
%         X_rec            --- recovered image
%         missing          --- index matrix of missing elements
% 
%     Outputs: 
%         erec             --- reconstruction error
%         psnr             --- PSNR (Peak Signal-to-Noise Ratio)
%--------------------------------------------------------------------------

X_rec = max(X_rec, 0);
X_rec = min(X_rec, 255);

erec = norm(X_full(:) - X_rec(:), 2);

[m, n, dim] = size(X_rec);
% MSE = 0;
% for i =1 : dim
%     MSE = MSE + norm((X_full(:,:,i)-X_rec(:,:,i)) .* missing, 'fro')^2;
% end
MSE = norm(vec((X_full-X_rec).*missing))^2 / nnz(missing);
% MSE = MSE / (3*nnz(missing));
psnr = 10 * log10(255^2 / MSE);

end