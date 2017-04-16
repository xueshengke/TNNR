% written by debingzhang
% if you have any questions, please fell free to contact
% debingzhangchina@gmail.com

% version 1.0, 2012.11.16

function [ X_opt, iter ] = admmAXB( A, B, X, M, known, para)

% my_admm
% solve  minimize ||X||_*-trace(A*X*B')
DISPLAY_EVERY = 10;

max_iter = para.admm_iter;
tol = para.admm_tol;
rho = para.admm_rho;

W = X;
Y = X;
AB = A'*B;
missing = ones(size(known)) - known;
M_fro = norm(M, 'fro');
obj_val = zeros(max_iter, 1);

for k = 1 : max_iter
    % X update
    last_X = X;
    temp = W - Y / rho;
    [U, sigma, V] = svd(temp, 'econ');
    X = U * max(sigma - 1/rho, 0) * V';
    
    delta_X = norm(X - last_X, 'fro') / M_fro;
    if mod(k, DISPLAY_EVERY) == 0
        fprintf('    iter %d, ||X_k+1-X_k||_F=%.4f', k, delta_X);
    end
    if delta_X < tol
        fprintf('    converged at\n');
        fprintf('    iter %d, ||X_k+1-X_k||_F=%.4f\n', k, delta_X);
        break ;
    end
    
    % W update
    W = X + (AB + Y) / rho;
    W = W .* missing + M .* known;
    
    % Y update
    Y = Y + rho * (X - W);

    obj_val(k) = nuclear_norm(X) - trace(A*W*B') ...
        + rho / 2 * norm(X-W, 'fro')^2 + trace(Y'*(X-W));
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

function obj = nuclear_norm(X)
    if size(X, 1) > 2000
        sigma = svds(X, 100);
    else
        sigma = svd(X);
    end
    obj = sum(sigma);
end
