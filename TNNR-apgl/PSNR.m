function [ psnr ] = PSNR( Xfull, Xrecover, missing )
%PSNR Summary of this function goes here
%   Detailed explanation goes here

Xrecover = max(Xrecover, 0);
Xrecover = min(Xrecover, 255);
[m, n, dim] = size(Xrecover);
MSE = 0;
for i =1 : dim
    MSE = MSE + norm((Xfull(:,:,i)-Xrecover(:,:,i)) .* missing, 'fro')^2;
end
MSE = MSE / (3*nnz(missing));
psnr = 10 * log10(255^2 / MSE);

end