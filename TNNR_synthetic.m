% Xue Shengke, Zhejiang University, April 2017. 
% Contact information: see readme.txt.
%
% Reference: 
% Hu, Y., Zhang, D., Ye, J., Li, X., & He, X. (2013). Fast and accurate matrix 
% completion via truncated nuclear norm regularization. IEEE Transactions on 
% Pattern Analysis and Machine Intelligence, 35(9), 2117-2130.
% 
% First written by debingzhang, Zhejiang Universiy, November 2012.

%% add path
close all; clear ; clc;
addpath pic ;
addpath mask ;
addpath function ;
addpath TNNR-admm;
addpath TNNR-apgl;

%% read image files directory information
admm_result = './TNNR-admm/result/synthetic';
apgl_result = './TNNR-apgl/result/synthetic';
if ~exist(admm_result, 'dir'),   mkdir(admm_result); end
if ~exist(apgl_result, 'dir'),   mkdir(apgl_result); end

%% parameter configuration
para.lost = 0.50;        % percentage of lost elements in matrix
para.save_eps = 0;       % save eps figure in result directory
para.min_R = 1;          % minimum rank of chosen image
para.max_R = 15;          % maximum rank of chosen image
% it requires to test all ranks from min_R to max_R, note that different
% images have different ranks, and various masks affect the ranks, too.

para.outer_iter = 200;     % maximum number of iteration
para.outer_tol = 3e-4;     % tolerance of iteration

para.admm_iter = 200;    % iteration of the ADMM optimization
para.admm_tol = 1e-4;    % epsilon of the ADMM optimization
para.admm_rho = 5e-2;    % rho of the the ADMM optimization

para.apgl_iter = 200;    % iteration of the APGL optimization
para.apgl_tol = 1e-4;    % epsilon of the APGL optimization
para.apgl_lambda = 1e-2; % lambda of the the APGL optimization
para.progress = 0;

%% generate synthetic data for experiment
image_name = 'synthetic_data';
m = 200;
n = 200;
dim = 3;
r = 10;
sigma = 0.3;    % [0.1, 0.9]

%% random loss
rnd_idx = randi([0, 100-1], m, n);
old_idx = rnd_idx;
lost = para.lost * 100;
fprintf('loss: %d%% elements are missing.\n', lost);
rnd_idx = double(old_idx < (100-lost));
mask = repmat(rnd_idx, [1 1 dim]); % index matrix of the known elements

L = randn(m, r);
R = randn(n, r);
noise = sigma * randn(m, n);
M = L * R' + noise * mask(:,:,1);
M = mapminmax(M, 0, 255);
X_full = repmat(M, [1 1 dim]);

%% run truncated nuclear norm regularization through ADMM
fprintf('ADMM optimization method to recovery an image with missing pixels\n');
t1 = tic;
[admm_res, X_rec]= admm_pic(admm_result, image_name, X_full, mask, para);

admm_rank = admm_res.best_rank;
admm_psnr = admm_res.best_psnr;
admm_erec = admm_res.best_erec ./ 255;
admm_time_cost = admm_res.time(admm_rank);
admm_iteration = admm_res.iterations(admm_rank, :);
admm_total_iter = admm_res.total_iter(admm_rank, :);

fprintf('\nTNNR-ADMM: ');
fprintf('rank=%d, psnr=%f, erec=%f, time=%f s, iteration=%d(%d),%d(%d),%d(%d),\n', ...
    admm_rank, admm_psnr, admm_erec, admm_time_cost, admm_iteration(1), ...
    admm_total_iter(1), admm_iteration(2), admm_total_iter(2), admm_iteration(3), ...
    admm_total_iter(3));
disp(' ');

figure('NumberTitle', 'off', 'Name', 'TNNR-ADMM result');
subplot(2, 2, 1);
plot(admm_res.Rank, admm_res.Psnr, 'o-');
xlabel('Rank');
ylabel('PSNR');

subplot(2, 2, 2);
plot(admm_res.Rank, admm_res.Erec ./ 255, 'diamond-');
xlabel('Rank');
ylabel('Recovery error');

subplot(2, 2, 3);
plot(admm_res.Psnr_iter, 'square-');
xlabel('Iteration');
ylabel('PSNR');

subplot(2, 2, 4);
plot(admm_res.Erec_iter ./ 255, '^-');
xlabel('Iteration');
ylabel('Recovery error');

if para.progress
    figure('NumberTitle', 'off', 'Name', 'TNNR-ADMM progress');
    num_iter = min(admm_iteration);
    X_rec = X_rec / 255;
    for i = 1 : num_iter
        imshow(X_rec(:, :, :, i));
        title(['iter ' num2str(i)]);
    end    % better set a breakpoint here, to display image step by step
end

%% record test results
outputFileName = fullfile(admm_result, 'parameters.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '****** %s ******\n', datestr(now,0));
fprintf(fid, '%s\n', ['image: '           image_name               ]);
fprintf(fid, '%s\n', ['loss ratio: '      num2str(para.lost)       ]);
fprintf(fid, '%s\n', ['min rank: '        num2str(para.min_R)      ]);
fprintf(fid, '%s\n', ['max rank: '        num2str(para.max_R)      ]);
fprintf(fid, '%s\n', ['max iteration: '   num2str(para.outer_iter) ]);
fprintf(fid, '%s\n', ['tolerance: '       num2str(para.outer_tol)  ]);
fprintf(fid, '%s\n', ['ADMM iteration: '  num2str(para.admm_iter)  ]);
fprintf(fid, '%s\n', ['ADMM tolerance: '  num2str(para.admm_tol)   ]);
fprintf(fid, '%s\n', ['ADMM rho: '        num2str(para.admm_rho)   ]);

fprintf(fid, '%s\n', ['sigma: '           num2str(sigma)           ]);
fprintf(fid, '%s\n', ['rank: '            num2str(admm_rank)       ]);
fprintf(fid, '%s\n', ['psnr: '            num2str(admm_psnr)       ]);
fprintf(fid, '%s\n', ['recovery error: '  num2str(admm_erec)       ]);
fprintf(fid, '%s\n', ['time cost: '       num2str(admm_time_cost)  ]);
fprintf(fid, 'outer iteration: %d, %d, %d\n',   admm_iteration(1), ...
    admm_iteration(2), admm_iteration(3));
fprintf(fid, 'total iteration: %d, %d, %d\n',   admm_total_iter(1), ...
    admm_total_iter(2), admm_total_iter(3));
fprintf(fid, '--------------------\n');
fclose(fid);

%% run truncated nuclear norm regularization through APGL
fprintf('APGL optimization method to recovery an image with missing pixels\n');
t2 = tic;
[apgl_res, X_rec]= apgl_pic(apgl_result, image_name, X_full, mask, para);

apgl_rank = apgl_res.best_rank;
apgl_psnr = apgl_res.best_psnr;
apgl_erec = apgl_res.best_erec ./ 255;
apgl_time_cost = apgl_res.time(apgl_rank);
apgl_iteration = apgl_res.iterations(apgl_rank, :);
apgl_total_iter = apgl_res.total_iter(apgl_rank, :);

fprintf('\nTNNR-APGL: ');
fprintf('rank=%d, psnr=%f, erec=%f, time=%f s, iteration=%d(%d),%d(%d),%d(%d),\n', ...
    apgl_rank, apgl_psnr, apgl_erec, apgl_time_cost, apgl_iteration(1), ...
    apgl_total_iter(1), apgl_iteration(2), apgl_total_iter(2), apgl_iteration(3), ...
    apgl_total_iter(3));
disp(' ');

figure('NumberTitle', 'off', 'Name', 'TNNR-APGL result');
subplot(2, 2, 1);
plot(apgl_res.Rank, apgl_res.Psnr, 'o-');
xlabel('Rank');
ylabel('PSNR');

subplot(2, 2, 2);
plot(apgl_res.Rank, apgl_res.Erec ./ 255, 'diamond-');
xlabel('Rank');
ylabel('Recovery error');

subplot(2, 2, 3);
plot(apgl_res.Psnr_iter, 'square-');
xlabel('Iteration');
ylabel('PSNR');

subplot(2, 2, 4);
plot(apgl_res.Erec_iter ./ 255, '^-');
xlabel('Iteration');
ylabel('Recovery error');

if para.progress
    figure('NumberTitle', 'off', 'Name', 'TNNR-APGL progress');
    num_iter = min(apgl_iteration);
    X_rec = X_rec / 255;
    for i = 1 : num_iter
        imshow(X_rec(:, :, :, i));
        title(['iter ' num2str(i)]);
    end    % better set a breakpoint here, to display image step by step
end

%% record test results
outputFileName = fullfile(apgl_result, 'parameters.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '****** %s ******\n', datestr(now,0));
fprintf(fid, '%s\n', ['image: '           image_name               ]);
fprintf(fid, '%s\n', ['loss ratio: '      num2str(para.lost)       ]);
fprintf(fid, '%s\n', ['min rank: '        num2str(para.min_R)      ]);
fprintf(fid, '%s\n', ['max rank: '        num2str(para.max_R)      ]);
fprintf(fid, '%s\n', ['max iteration: '   num2str(para.outer_iter) ]);
fprintf(fid, '%s\n', ['tolerance: '       num2str(para.outer_tol)  ]);
fprintf(fid, '%s\n', ['APGL iteration: '  num2str(para.apgl_iter)  ]);
fprintf(fid, '%s\n', ['APGL tolerance: '  num2str(para.apgl_tol)   ]);
fprintf(fid, '%s\n', ['APGL lambda: '     num2str(para.apgl_lambda)]);

fprintf(fid, '%s\n', ['sigma: '           num2str(sigma)           ]);
fprintf(fid, '%s\n', ['rank: '            num2str(apgl_rank)       ]);
fprintf(fid, '%s\n', ['psnr: '            num2str(apgl_psnr)       ]);
fprintf(fid, '%s\n', ['recovery error: '  num2str(apgl_erec)       ]);
fprintf(fid, '%s\n', ['time cost: '       num2str(apgl_time_cost)  ]);
fprintf(fid, 'outer iteration: %d, %d, %d\n',   apgl_iteration(1), ...
    apgl_iteration(2), apgl_iteration(3));
fprintf(fid, 'total iteration: %d, %d, %d\n',   apgl_total_iter(1), ...
    apgl_total_iter(2), apgl_total_iter(3));
fprintf(fid, '--------------------\n');
fclose(fid);
