clc
clear


%% regression: school dataset
opts.max_iter =500;
opts.rel_tol  =10^-3;
opts.rho=2;
K=9;
hyp = [2,8,2,3];
load('schoo_rep1.mat')
[U,V,f] = VSTG_MTL_regress(school_train_input,school_train_output,K,hyp,opts);
W = U*V;
for task=1:139
    school_test_output_hat{task} = school_test_input{task} * W(:,task);
    resi{task} = school_test_output{task} - school_test_output_hat{task};
    RMSE(task) = sqrt(mean(resi{task}.^2));
end
fprintf(sprintf('RMSE: %f\n',mean(RMSE)));
%% classification: mnist dataset 
% VSTG_MTL_logistic requires minFunc by Schmidit, 2005 
% (https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
addpath(genpath('minFunc_2012'))
load('mnistPCA_1k.mat')
hyp2= [2^-3,2^-3,2^-3,3];
[U_mnist,V_mnist,f_mnist] =VSTG_MTL_logistic(X_train,Y_train,K,hyp2,opts);
W_mnist = U_mnist*V_mnist;
T = size(X_test,1);
Y_test_hat = cell(T,1);

for t = 1:T
    Y_test_hat{t} = sign(1./(1+exp(-X_test{t}*W_mnist(:,t)))-0.5);
end
Y_test_vec = cell2mat(Y_test);
Y_test_hat_vec = cell2mat(Y_test_hat);
Err = mean(Y_test_vec~=Y_test_hat_vec);
fprintf(sprintf('Error Rate: %f\n',Err));
