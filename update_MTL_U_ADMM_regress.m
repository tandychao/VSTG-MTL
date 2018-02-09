function [ U_opt U_history ] = update_MTL_U_ADMM_regress(U_init, V,X_cell,Y_cell,hyp,opts);


%% initialization
M = length(Y_cell);
[D,K] = size(U_init);
U=U_init;
Z1=U;
Z2=U;
Z3=U;

L1=U;
L2=U;
L3=U;
iter=0;
  
rho =opts.rho;

%% update
while iter<=10
%     disp(iter)

    % update U
    U = (Z1+Z2+Z3 - L1-L2-L3)/3;
    
    % update auxiliary variables Z
    Z1_old = Z1;
    Z2_old= Z2;
    Z3_old = Z3;
    
    A = rho * eye(D*K);
    B = rho *(U(:) + L1(:));
    
    for t=1:M
        N_t = size(X_cell{t},1);
        A = A + 1/N_t * kron(V(:,t)*V(:,t)', X_cell{t}'*X_cell{t});
        b = X_cell{t}'*Y_cell{t} * V(:,t)';
        B = B + 1/N_t * b(:);
    end
    Z1 = reshape(linsolve(A,B),size(Z1));    
    Z2 = proximal_L11norm(U + L2, hyp(1)/rho);
    Z3 = proximal_L1infnorm(U + L3, hyp(2)/rho);
    
    % update Lagrangian variables L
    
    L1 = L1+ U - Z1;
    L2 = L2 + U - Z2;
    L3 = L3 + U - Z3;
    
    iter=iter+1;
    U_history.fun(iter) = fun_eval(Z1,Z2,Z3);
    U_history.r_norm(iter) = (norm(U-Z1,'fro')^2 + norm(U-Z2,'fro')^2 + norm(U-Z3,'fro')^2)^0.5;
    U_history.s_norm(iter) = rho * (norm(Z1-Z1_old,'fro')^2 + norm(Z2-Z2_old,'fro')^2 + norm(Z3-Z3_old,'fro')^2)^0.5; 
    if U_history.r_norm(iter) < 10^-1 && U_history.s_norm(iter)<= 10^-1
        break;    
    end
    
end
U_opt = Z2;
U_opt(abs(U_opt)<10^-4)=0;


    %%   private function
    function [X] = proximal_L11norm(D, tau)
        % min_X 0.5*||X - D||_F^2 + tau*||X||_{1,1}
        % where ||X||_{1,1} = sum_ij|X_ij|, where X_ij denotes the (i,j)-th entry of X
        X = sign(D).*max(0,abs(D)-tau);
    end

    function [X] = proximal_L1infnorm(D, tau)
    % min_X 0.5*||X - D||_F^2 + tau*||X||_{1,inf}
    % where ||X||_{1,inf} = sum_i||X^i||_inf, where X^i denotes the i-th row of X

    % X = D; n = size(D,2);
    % for ii = 1:size(D,1)
    %     [mu,~,~] = prf_lb(D(ii,:)', n, tau);
    %     X(ii,:) = D(ii,:) - mu';
    % end

    [m,n]=size(D);
    [temp,~,~]=prf_lbm(D,m,n,tau);
    X = D - temp;
    end
%     
    function val = fun_eval(Z1,Z2,Z3)
        val = hyp(1) * norm(Z2,1) + hyp(2)*sum(max(abs(Z3),[],2)); %                 
        for task=1:M
            val = val + norm(Y_cell{task} - X_cell{task} * Z1 * V(:,task),2)^2/(2*length(Y_cell{task}));
        end
    end
% %  
% 
end

