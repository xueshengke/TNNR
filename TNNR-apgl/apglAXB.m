function [ X_opt, iter ] = apglAXB( A, B, X, M, known, para)    
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, April 2017.
% Contact information: see readme.txt.
%
% Hu et al. (2013) TNNR paper, IEEE Transactions on PAMI.
% First written by debingzhang, Zhejiang Universiy, November 2012.
%--------------------------------------------------------------------------
%     iteratively solve the objective: min  g(x) + h(x)
%     g(x) = ||X||_*
%     h(x) = - trace(AXB') + \lambda * ||X_\Omega - M_\Omega||_F^2
%     Inputs:
%         A                 --- left singular vector
%         B                 --- right singular vector
%         X                 --- incomplete image
%         M                 --- original image
%         known             --- index matrix of known elements
%         para              --- struct of parameters
% 
%     Outputs: 
%         X_opt             --- optimized image
%         iter              --- number of iterations
%--------------------------------------------------------------------------

DISPLAY_EVERY = 10;

max_iter = para.apgl_iter;
tol = para.apgl_tol;
lambda = para.apgl_lambda;

AB = A' * B;
Y = X;
t = 1;
M_fro = norm(M, 'fro');
obj_val = zeros(max_iter, 1);

for k = 1 : max_iter
    % X update
    last_X = X;
    temp = Y + t * (AB - lambda * (Y - M) .* known);
    [U, sigma, V] = svd(temp);
    X = U * max(sigma - t, 0) * V';

    % t update
    last_t = t;
    t = (1 + sqrt(1 + 4 * last_t^2)) / 2;

    % Y update
    Y = X + (last_t-1) / t * (X - last_X);

    delta = norm(X - last_X, 'fro') / M_fro;
    if mod(k, DISPLAY_EVERY) == 0
        fprintf('    iter %d, ||X_k+1-X_k||_F/||M||_F=%.4f', k, delta);
    end
    if delta < tol
        fprintf('    converged at\n');
        fprintf('    iter %d, ||X_k+1-X_k||_F/||M||_F=%.4f\n', k, delta);
        break ;
    end

    obj_val(k) = nuclear_norm(X) - trace(A*X*B') ...
        + lambda / 2 * norm((X-M).*known, 'fro')^2;
    if mod(k, DISPLAY_EVERY) == 0
        fprintf(', obj value=%.4f\n', obj_val(k));
    end
    if k > 1 && abs(obj_val(k) - obj_val(k-1)) < tol
        fprintf('    converged at\n');
        fprintf('    iter %d, obj value=%.4f\n', k, obj_val(k));        
        break;
    end
end

X_opt = X;
iter = k;

end