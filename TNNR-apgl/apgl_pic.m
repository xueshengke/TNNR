function [res, Xrecover] = apgl_pic(image_name, Xfull, mask, para)

Xmiss = Xfull .* mask;	% data matrix with some pixels lost
[m, n, dim] = size(Xfull);
known = mask(:, :, 1);        % index matrix of the known elements
missing = ones(m,n) - known;  % index matrix of the missing elements

min_R = para.min_R;
max_R = para.max_R;

Psnr = zeros(max_R, 1);
time_cost = zeros(max_R, 1);
iter_cost = zeros(max_R, 1);

best_psnr = 0;
best_rank = 0;

Xrecover = zeros(m, n, dim, max_R);

max_iter = para.outer_iter;
tol = para.outer_tol;

figure;
subplot(2,1,1);
imshow(Xmiss ./ 255);   % show the missing image
xlabel('image with missing pixels');

for R = min_R : max_R
    tic;
    for c = 1 : dim
        fprintf('rank(r)=%d, channel(RGB)=%d \n', R, c);
        X = Xmiss(:, :, c);
        M = X;
        X_rec = zeros(m, n, max_iter);
        for i = 1 : max_iter
            fprintf('iter %d\n', i);
            [U, sigma, V] = svd(X);
            A = U(:, 1:R)'; B = V(:, 1:R)'; 
            [X_rec(:, :, i), iter_count] = apglAXB(A, B, X, M, known, para);
            iter_cost(R) = iter_cost(R) + iter_count;
            
            if i > 1 && norm(X_rec(:,:,i) - X_rec(:,:,i-1), 'fro') / norm(M, 'fro') < tol
                X = X_rec(:, :, i);
                break ;
            end
            X = X_rec(:, :, i);
        end
        Xrecover(:, :, c, R) = X;
    end
    time_cost(R) = toc;
    temp_recover = max(Xrecover(:, :, :, R), 0);
    temp_recover = min(temp_recover, 255);
    Psnr(R) = PSNR(Xfull, temp_recover, missing);
    if best_psnr < Psnr(R)
        best_psnr = Psnr(R);
        best_rank = R;
    end 
end

subplot(2, 1, 2);   % show the recovered image
Xrecover = max(Xrecover(:, :, :, best_rank), 0);
Xrecover = min(Xrecover, 255);
imshow(Xrecover ./ 255);
xlabel('image recovered by TNNR-APGL');

if para.save_eps
    fig_eps = figure;
    imshow(Xrecover ./ 255, 'border', 'tight');
    fig_name = sprintf('TNNR-apgl/result/%s_rank_%d_PSNR_%.2f_APGL', ...
        image_name(1:end-4), best_rank, best_psnr);
    saveas(gcf, [fig_name '.eps'], 'psc2');
    fprintf('eps figure saved in %s.eps\n', fig_name);
    close(fig_eps);
end

res.time = time_cost;
res.iterations = iter_cost;
res.psnr = best_psnr;
res.rank = best_rank;
res.Psnr = Psnr(min_R:max_R);
res.Rank = (min_R : max_R)';
end
