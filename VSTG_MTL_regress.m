function [U_old,V_old,fun] = VSTG_MTL_regress(X_cell,Y_cell,K,hyp, opts)
% Variable Selection and Task Grouping Multi-task Learning for regression
% Objective function

% description
% D variables and T tasks
% \sum_{j=1}^T ||y_j -X_j * U * v_j||_2^2/N_j
% + gamma_1 ||U||_1 + gamma2 ||U||_1,inf + mu * sum_{j=1}^T (||v_j||_k^{sp})^2
% Input arguments
% X_cell: T \times 1 input cell matrix, 
%         where the j-th component is X_j \in N_j * D
% Y_cell: T \times 1 output cell matrix,
%         where the j-th component is y_j \in N_j \in R^{N_j}
% K: rank of coeffcient matrix = # of latent bases
% hyp=[gamma1, gamma2, mu, k]: regularization parameters
% opts: opts.max_iter, opts.rel_tol,

%% Initialization 
T = length(Y_cell);
D = size(X_cell{1},2);
opts.k = hyp(4);
if isfield (opts, 'U_init')
    U_old = opts.U_init;
    V_old = opts.V_init;
else
    for task=1:T
        left = (X_cell{task}'*X_cell{task} + norm(hyp(1:3),2)+eye(D));
        right = X_cell{task}'*Y_cell{task};
        W_init(:,task) = left\right;        
    end
    [U_temp,Lambda,V_temp] = svds(W_init,K);
    U_old = U_temp * sqrt(Lambda);
    V_old = sqrt(Lambda) *V_temp';    
end
fun = Obj_val(U_old,V_old);
iter=1;
%% main

while iter<opts.max_iter
%     disp(iter)
    
    % update U
    [U_new, history] = update_MTL_U_ADMM_regress(U_old, V_old,X_cell,Y_cell,hyp(1:2),opts);
    % update V       
    tic
    for task=1:T
        Z = X_cell{task}*U_new;
        N_task = length(Y_cell{task});
        opts.init_beta = V_old(:,task);
        
        V_new(:,task) = ksupport_FISTA_regress(Z,Y_cell{task}, hyp(3), opts);
    end      
    eval_time(iter,2) = toc;
    
    f_new = Obj_val(U_new,V_new);
    fun = cat(1,fun, f_new);
    
    % stopping criteria 
    if iter>=2 && abs(fun(end)-fun(end-1))<= opts.rel_tol*fun(end-1)
        break;
    end       
    
    U_old =U_new;
    V_old=V_new;  
    iter=iter+1;
        
end




    %%      

    function val = Obj_val(U,V)
        
        val = hyp(1) * norm(U,1) + hyp(2)*sum(max(abs(U),[],2)) +  hyp(3) * norm_overlap(U(:),opts.k)^2;
        
        for m2=1:T
            val = val + norm(Y_cell{m2} - X_cell{m2} * U*V(:,m2),2)^2 /(2*length(Y_cell{m2}));
        end             
    end

   




end

