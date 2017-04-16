% written by debingzhang
% if you have any questions, please fell free to contact
% debingzhangchina@gmail.com

% version 1.0, 2012.11.16

close all; clear ; clc;
addpath pic;
addpath TNNR-admm;
addpath TNNR-apgl;
%% for experiment use
pic_list = {'re1.jpg', 're2.jpg', 're3.jpg', 're4.jpg', 're5.jpg', 're6.jpg', ...
            're7.jpg', 're8.jpg', 're9.jpg', 're10.jpg', 're11.jpg' };

imagenum = 6;
pic_name = pic_list{imagenum};

Xfull = double(imread(pic_name));
[m, n, dim] = size(Xfull);

%% parameter configuration
para.lost = 0.10;        % percentage of lost elements in data matrix
para.save_eps = 1;       % save eps figure in result directory
para.min_R = 15;         % for others, it requires to test all small ranks
para.max_R = 20;         % as we test r=9 maybe the best for re10.jpg
para.outer_iter = 20;    % number of the outer iteration
para.outer_tol = 1e-3;   % epsilon of the outer iteration

para.admm_iter = 200;    % iteration of the ADMM optimization
para.admm_tol = 1e-4;    % epsilon of the ADMM optimization
para.admm_rho = 5e-2;    % rho of the the ADMM optimization

para.apgl_iter = 200;    % iteration of the APGL optimization
para.apgl_tol = 1e-4;    % epsilon of the APGL optimization
para.apgl_lambda = 1e-2; % lambda of the the APGL optimization

%% ratio of missing pixels
index = randi([0, 100-1], m, n);
oldidx = index;
lost = para.lost * 100;
fprintf('%d%% elements are missing\n', lost);
index = (oldidx < (100-lost));
mask = repmat(index, [1 1 dim]);  % index matrix of the known elements

%% run truncated nuclear norm regularization through ADMM
fprintf('ADMM optimization method to recovery an image with missing pixels\n');
t1 = tic;
[admm_res, Xrecover]= admm_pic(pic_name, Xfull, mask, para);

admm_iteration = admm_res.iterations(admm_res.rank);
admm_psnr = admm_res.psnr;
admm_time_cost = toc(t1);
admm_rank = admm_res.rank;

figure;
plot(admm_res.Rank, admm_res.Psnr, 'o-');
title('TNNR-ADMM');
xlabel('Rank');
ylabel('PSNR');

fprintf('\nTNNR-ADMM: rank=%d, psnr=%.3f, time=%.1f s, iteration=%d\n', ...
    admm_rank, admm_psnr, admm_time_cost, admm_iteration);
disp(' ');

%% run truncated nuclear norm regularization through APGL
fprintf('APGL optimization method to recovery an image with missing pixels\n');
t2 = tic;
[apgl_res, Xrecover]= apgl_pic(pic_name, Xfull, mask, para);

apgl_iteration = apgl_res.iterations(apgl_res.rank);
apgl_psnr = apgl_res.psnr;
apgl_time_cost = toc(t2);
apgl_rank = apgl_res.rank;

figure;
plot(apgl_res.Rank, apgl_res.Psnr, 'diamond-');
title('TNNR-APGL');
xlabel('Rank');
ylabel('PSNR');

fprintf('\nTNNR-APGL: rank=%d, psnr=%.3f, time=%.1f s, iterations=%d\n', ...
    apgl_rank, apgl_psnr, apgl_time_cost, apgl_iteration);
