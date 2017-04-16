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
admm_result = './TNNR-admm/result';
apgl_result = './TNNR-apgl/result';
if ~exist(admm_result, 'dir'),   mkdir(admm_result); end
if ~exist(apgl_result, 'dir'),   mkdir(apgl_result); end
image_list = {'re1.jpg', 're2.jpg', 're3.jpg', 're4.jpg', 're5.jpg', ...
              're6.jpg', 're7.jpg', 're8.jpg', 're9.jpg', 're10.jpg' };

file_list = dir('mask');
num_mask = length(file_list) - 2;
mask_list = cell(num_mask, 1);
for i = 1 : num_mask
    mask_list{i} = file_list(i+2).name;
end

% imagenum = 6;
% pic_name = pic_list{imagenum};
% 
% Xfull = double(imread(pic_name));
% [m, n, dim] = size(Xfull);

%% parameter configuration
image_id = 1;            % select an image for experiment
mask_id  = 4;            % select a mask for experiment

para.block = 1;          % 1 for block occlusion, 0 for random noise
para.lost = 0.50;        % percentage of lost elements in matrix
para.save_eps = 1;       % save eps figure in result directory
para.min_R = 10;         % minimum rank of chosen image
para.max_R = 15;         % maximum rank of chosen image
% it requires to test all ranks from min_R to max_R, note that different
% images have different ranks, and various masks affect the ranks, too.

para.outer_iter = 25;    % number of the outer iteration
para.outer_tol = 1e-3;   % epsilon of the outer iteration

para.admm_iter = 200;    % iteration of the ADMM optimization
para.admm_tol = 1e-4;    % epsilon of the ADMM optimization
para.admm_rho = 5e-2;    % rho of the the ADMM optimization

para.apgl_iter = 200;    % iteration of the APGL optimization
para.apgl_tol = 1e-4;    % epsilon of the APGL optimization
para.apgl_lambda = 1e-2; % lambda of the the APGL optimization

% %% ratio of missing pixels
% index = randi([0, 100-1], m, n);
% oldidx = index;
% lost = para.lost * 100;
% fprintf('%d%% elements are missing\n', lost);
% index = (oldidx < (100-lost));
% mask = repmat(index, [1 1 dim]);  % index matrix of the known elements

%% select an image and a mask for experiment
image_name = image_list{image_id};
X_full = double(imread(image_name));
[m, n, dim] = size(X_full);
fprintf('choose image: %s, ', image_name);

if para.block  
    % block occlusion
    mask = double(imread(mask_list{mask_id}));
    mask = mask ./ max(mask(:));       % index matrix of the known elements
    fprintf('mask: %s.\n', mask_list{mask_id});
else
    % random loss
    rnd_idx = randi([0, 100-1], m, n);
    old_idx = rnd_idx;
    lost = para.lost * 100;
    fprintf('loss: %d%% elements are missing.\n', lost);
    rnd_idx = double(old_idx < (100-lost));
    mask = repmat(rnd_idx, [1 1 dim]); % index matrix of the known elements
end

%% run truncated nuclear norm regularization through ADMM
fprintf('ADMM optimization method to recovery an image with missing pixels\n');
t1 = tic;
[admm_res, X_rec]= admm_pic(admm_result, image_name, X_full, mask, para);

% admm_iteration = admm_res.iterations(admm_res.rank);
% admm_psnr = admm_res.psnr;
% admm_time_cost = toc(t1);
% admm_rank = admm_res.rank;

admm_rank = admm_res.best_rank;
admm_psnr = admm_res.best_psnr;
admm_erec = admm_res.best_erec;
admm_time_cost = admm_res.time(admm_rank);
admm_iteration = admm_res.iterations(apgl_rank, :);

fpritnf('\nTNNR-ADMM: ');
fprintf('rank=%d, psnr=%f, erec=%f, time=%f s, iteration=(%d,%d,%d)\n', ...
    admm_rank, admm_psnr, admm_erec, admm_time_cost, admm_iteration(1), ...
    admm_iteration(2), admm_iteration(3));
disp(' ');

figure('NumberTitle', 'off', 'Name', 'TNNR-ADMM result');
subplot(2, 2, 1);
plot(admm_res.Rank, admm_res.Psnr, 'o-');
xlabel('Rank');
ylabel('PSNR');

subplot(2, 2, 2);
plot(admm_res.Rank, admm_res.Erec, 'diamond-');
xlabel('Rank');
ylabel('Recovery error');

subplot(2, 2, 3);
plot(admm_res.Psnr_iter, 'square-');
xlabel('Iteration');
ylabel('PSNR');

subplot(2, 2, 4);
plot(admm_res.Erec_iter, '^-');
xlabel('Iteration');
ylabel('Recovery error');

% fprintf('\nTNNR-ADMM: rank=%d, psnr=%.3f, time=%.1f s, iteration=%d\n', ...
%     admm_rank, admm_psnr, admm_time_cost, admm_iteration);
% disp(' ');
para.admm_iter = 200;    % iteration of the ADMM optimization
para.admm_tol = 1e-4;    % epsilon of the ADMM optimization
para.admm_rho = 5e-2;    % rho of the the ADMM optimization

%% record test results
outputFileName = fullfile(admm_result, 'parameters.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '****** %s ******\n', datestr(now,0));
fprintf(fid, '%s\n', ['image: '           image_name               ]);
fprintf(fid, '%s\n', ['mask: '            mask_list{mask_id}       ]);
fprintf(fid, '%s\n', ['block or noise: '  num2str(para.block)      ]);
fprintf(fid, '%s\n', ['loss ratio: '      num2str(para.lost)       ]);
fprintf(fid, '%s\n', ['save eps figure: ' num2str(para.save_eps)   ]);
fprintf(fid, '%s\n', ['min rank: '        num2str(para.min_R)      ]);
fprintf(fid, '%s\n', ['max rank: '        num2str(para.max_R)      ]);
fprintf(fid, '%s\n', ['max iteration: '   num2str(para.outer_iter) ]);
fprintf(fid, '%s\n', ['tolerance: '       num2str(para.outer_tol)  ]);
fprintf(fid, '%s\n', ['ADMM iteration: '  num2str(para.admm_iter)  ]);
fprintf(fid, '%s\n', ['ADMM tolerance: '  num2str(para.admm_tol)   ]);
fprintf(fid, '%s\n', ['ADMM rho: '        num2str(para.admm_rho)   ]);

fprintf(fid, '%s\n', ['rank: '            num2str(admm_rank)       ]);
fprintf(fid, '%s\n', ['psnr: '            num2str(admm_psnr)       ]);
fprintf(fid, '%s\n', ['recovery error: '  num2str(admm_erec)       ]);
fprintf(fid, '%s\n', ['time cost: '       num2str(admm_time_cost)  ]);
fprintf(fid, 'iteration: %d, %d, %d\n',   admm_iteration(1), ...
    admm_iteration(2), admm_iteration(3));
fprintf(fid, '--------------------\n');
fclose(fid);

%% run truncated nuclear norm regularization through APGL
fprintf('APGL optimization method to recovery an image with missing pixels\n');
t2 = tic;
[apgl_res, X_rec]= apgl_pic(apgl_result, image_name, Xfull, mask, para);

% apgl_iteration = apgl_res.iterations(apgl_res.rank);
% apgl_psnr = apgl_res.psnr;
% apgl_time_cost = toc(t2);
% apgl_rank = apgl_res.rank;
% 
% figure;
% plot(apgl_res.Rank, apgl_res.Psnr, 'diamond-');
% title('TNNR-APGL');
% xlabel('Rank');
% ylabel('PSNR');
% 
% fprintf('\nTNNR-APGL: rank=%d, psnr=%.3f, time=%.1f s, iterations=%d\n', ...
%     apgl_rank, apgl_psnr, apgl_time_cost, apgl_iteration);

apgl_rank = apgl_res.best_rank;
apgl_psnr = apgl_res.best_psnr;
apgl_erec = apgl_res.best_erec;
apgl_time_cost = apgl_res.time(apgl_rank);
apgl_iteration = apgl_res.iterations(apgl_rank, :);

fpritnf('\nTNNR-APGL: ');
fprintf('rank=%d, psnr=%f, erec=%f, time=%f s, iteration=(%d,%d,%d)\n', ...
    apgl_rank, apgl_psnr, apgl_erec, apgl_time_cost, apgl_iteration(1), ...
    apgl_iteration(2), apgl_iteration(3));
disp(' ');

figure('NumberTitle', 'off', 'Name', 'TNNR-APGL result');
subplot(2, 2, 1);
plot(apgl_res.Rank, apgl_res.Psnr, 'o-');
xlabel('Rank');
ylabel('PSNR');

subplot(2, 2, 2);
plot(apgl_res.Rank, apgl_res.Erec, 'diamond-');
xlabel('Rank');
ylabel('Recovery error');

subplot(2, 2, 3);
plot(apgl_res.Psnr_iter, 'square-');
xlabel('Iteration');
ylabel('PSNR');

subplot(2, 2, 4);
plot(apgl_res.Erec_iter, '^-');
xlabel('Iteration');
ylabel('Recovery error');

%% record test results
outputFileName = fullfile(apgl_result, 'parameters.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '****** %s ******\n', datestr(now,0));
fprintf(fid, '%s\n', ['image: '           image_name               ]);
fprintf(fid, '%s\n', ['mask: '            mask_list{mask_id}       ]);
fprintf(fid, '%s\n', ['block or noise: '  num2str(para.block)      ]);
fprintf(fid, '%s\n', ['loss ratio: '      num2str(para.lost)       ]);
fprintf(fid, '%s\n', ['save eps figure: ' num2str(para.save_eps)   ]);
fprintf(fid, '%s\n', ['min rank: '        num2str(para.min_R)      ]);
fprintf(fid, '%s\n', ['max rank: '        num2str(para.max_R)      ]);
fprintf(fid, '%s\n', ['max iteration: '   num2str(para.outer_iter) ]);
fprintf(fid, '%s\n', ['tolerance: '       num2str(para.outer_tol)  ]);
fprintf(fid, '%s\n', ['APGL iteration: '  num2str(para.apgl_iter)  ]);
fprintf(fid, '%s\n', ['APGL tolerance: '  num2str(para.apgl_tol)   ]);
fprintf(fid, '%s\n', ['APGL lambda: '     num2str(para.apgl_lambda)]);

fprintf(fid, '%s\n', ['rank: '            num2str(apgl_rank)       ]);
fprintf(fid, '%s\n', ['psnr: '            num2str(apgl_psnr)       ]);
fprintf(fid, '%s\n', ['recovery error: '  num2str(apgl_erec)       ]);
fprintf(fid, '%s\n', ['time cost: '       num2str(apgl_time_cost)  ]);
fprintf(fid, 'iteration: %d, %d, %d\n',   apgl_iteration(1), ...
    apgl_iteration(2), apgl_iteration(3));
fprintf(fid, '--------------------\n');
fclose(fid);
