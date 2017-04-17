function [result, X_rec] = admm_pic(result_dir, image_name, X_full, mask, para)
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, April 2017.
% Contact information: see readme.txt.
%
% Hu et al. (2013) TNNR paper, IEEE Transactions on PAMI.
% First written by debingzhang, Zhejiang Universiy, November 2012.
%--------------------------------------------------------------------------
%     main part of TNNR algorithm via ADMM
% 
%     Inputs:
%         result_dir           --- result directory for saving figures
%         image_name           --- name of image file
%         X_full               --- original image
%         mask                 --- index matrix of known elements
%         para                 --- struct of parameters
% 
%     Outputs: 
%         result               --- result of algorithm
%         X_rec                --- recovered image under the best rank
%--------------------------------------------------------------------------

X_miss = X_full .* mask;	  % incomplete image with some pixels lost
[m, n, dim] = size(X_full);
known = mask(:, :, 1);        % index matrix of known elements
missing = ones(m,n) - known;  % index matrix of missing elements

min_R    = para.min_R;        % minimum rank of chosen image
max_R    = para.max_R;        % maximum rank of chosen image
max_iter = para.outer_iter;   % maximum number of outer iteration
tol      = para.outer_tol;    % tolerance of outer iteration

Erec = zeros(max_R, 1);  % reconstruction error, best value in each rank
Psnr = zeros(max_R, 1);  % PSNR, best value in each rank
time_cost = zeros(max_R, 1);        % consuming time, each rank
iter_cost = zeros(max_R, dim);      % number of iterations, each channel
total_iter = zeros(max_R, dim);     % number of total iterations
X_rec = zeros(m, n, dim, max_iter); % recovered image under the best rank

best_rank = 0;  % record the best value
best_psnr = 0;
best_erec = 0;

figure('NumberTitle', 'off', 'Name', 'TNNR-ADMM image');
subplot(1,3,1);
imshow(X_full ./ 255);   % show the original image
xlabel('original image');

subplot(1,3,2);
imshow(X_miss ./ 255);   % show the incomplete image
xlabel('incomplete image');

for R = min_R : max_R    % test if each rank is proper for completion
    t_rank = tic;
    X_iter = zeros(m, n, dim, max_iter);
    X_temp = zeros(m, n, dim);
    for c = 1 : dim    % process each channel separately
        fprintf('rank(r)=%d, channel(RGB)=%d\n', R, c);
        X = X_miss(:, :, c);
        M = X_full(:, :, c);
        M_fro = norm(M, 'fro');
        last_X = X;
        delta = inf;
        for i = 1 : max_iter
            fprintf('iter %d\n', i);
            [U, sigma, V] = svd(X);
            A = U(:, 1:R)'; B = V(:, 1:R)';
            [X, iter_count] = admmAXB(A, B, X, M, known, para);
            X_iter(:, :, c, i) = X;
            
            iter_cost(R, c) = iter_cost(R, c) + 1;
            total_iter(R, c) = total_iter(R, c) + iter_count;
            
            delta = norm(X - last_X, 'fro') / M_fro;
            fprintf('||X_k+1-X_k||_F/||M||_F %.4f\n', delta);
            if delta < tol
                fprintf('converged at iter: %d\n', i);
                break ;
            end
            last_X = X;
        end
        X_temp(:, :, c) = X;
    end
    time_cost(R) = toc(t_rank);
    X_temp = max(X_temp, 0);
    X_temp = min(X_temp, 255);
    [Erec(R), Psnr(R)] = PSNR(X_full, X_temp, missing);
    if best_psnr < Psnr(R)
        best_rank = R;
        best_psnr = Psnr(R);
        best_erec = Erec(R);
        X_rec = X_iter;
    end
end
%% compute the reconstruction error and PSNR in each iteration 
%  for the best rank
num_iter = min(iter_cost(best_rank, :));
psnr_iter = zeros(num_iter, 1);
erec_iter = zeros(num_iter, 1);
for t = 1 : num_iter
    X_temp = X_rec(:, :, :, t);
    [erec_iter(t), psnr_iter(t)] = PSNR(X_full, X_temp, missing);
end
X_best_rec = X_rec(:, :, :, num_iter);

%% display recovered image
subplot(1, 3, 3);
X_best_rec = max(X_best_rec, 0);
X_best_rec = min(X_best_rec, 255);
imshow(X_best_rec ./ 255);    % show the recovered image
xlabel('recovered image');

%% save eps figure in result directory
if para.save_eps
    fig_eps = figure;
    imshow(X_best_rec ./ 255, 'border', 'tight');
    split_name = regexp(image_name, '[.]', 'split');
    fig_name = sprintf('%s/%s_rank_%d_PSNR_%.2f_Erec_%.2f', ...
        result_dir, split_name{1}, best_rank, best_psnr, best_erec);
    saveas(gcf, [fig_name '.eps'], 'psc2');
    fprintf('eps figure saved in %s.eps\n', fig_name);
    close(fig_eps);
end

%% record performances for output
result.time = time_cost;
result.iterations = iter_cost;
result.total_iter = total_iter;
result.best_rank = best_rank;
result.best_psnr = best_psnr;
result.best_erec = best_erec;
result.Rank = (min_R : max_R)';
result.Psnr = Psnr(min_R:max_R);
result.Erec = Erec(min_R:max_R);
result.Psnr_iter = psnr_iter;
result.Erec_iter = erec_iter;

end