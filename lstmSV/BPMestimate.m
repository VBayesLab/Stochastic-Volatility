function obj = BPMestimate(obj,model,y)
%BPMESTIMATE Bayesian estimation for LSTM-SV using BPM

%   Copyright 2019 Nghia Nguyen (nghia.nguyen@sydney.edu.au)
%
%   https://github.com/VBayesLab
%
%   Version: 1.0
%   LAST UPDATE: Sep, 2019

% Model setting
N        = obj.NumParticles;
nloop    = obj.NumMCMC;
savename = obj.SaveFileName;
t_accept = obj.TargetAccept;
N_block  = obj.NumBlock;
N_corr   = obj.NumCovariance; 
verbose  = obj.Verbose;
save_num = obj.SaveAfter;
theta    = obj.ParamsInit;
n_params = model.NumParams;

% Static setting
T     = length(y);
Sigma = 0.01*eye(n_params);
scale = 1;

% Likelihood estimation 
u_pro   = randn(N,T);    % Random numbers for proposal
u_res   = rand(N,T);     % Random number for resampling
log_lik = obj.LogLikelihood(theta,N,y,u_pro,u_res);

% Prepare for the first iteration
[log_prior,log_jac,theta_proposal] = obj.logPriorFnc(model,theta);
log_post = log_prior + log_lik;
 
% Store parameters to calculate adaptive proposal covariance matrix
thetasave = zeros(nloop,n_params);
tic
for i=1:nloop
    %% Print training message
    if(verbose)
        model.PrintMsg(theta,i);
    end
        
    %% Update U using random blocks
    block_idx  = randi(N_block);                                 % Get random block index
    u_pro_star = obj.getBlock(u_pro,block_idx,N_block,'Normal'); 
    u_res_star = obj.getBlock(u_res,block_idx,N_block,'Uniform');

    %% Propose new parameters with random walk proposal
    theta_temp     = mvnrnd(theta_proposal,scale.*Sigma);
    theta_temp_inv = obj.invTransformFnc(model,theta_temp);
    
    %%  Calculate acceptance probability
    theta_star     = obj.toStruct(model.NameParams,theta_temp_inv);
    log_lik_star   = obj.LogLikelihood(theta_star,N,y,u_pro_star,u_res_star);
    [log_prior_star,log_jac_star,theta_star_proposal] = obj.logPriorFnc(model,theta_star);
    log_post_star  = log_prior_star + log_lik_star;
    r1 = exp(log_post_star-log_post + log_jac_star-log_jac);
    
    %% Rejection decision
    A1 = rand();      % Use this uniform random number to accept a proposal sample
    C1 = min(1,r1);    
    % If accept the new proposal sample
    if (A1 <= C1)
       theta_proposal = theta_star_proposal;
       theta    = theta_star;
       log_post = log_post_star;
       log_jac  = log_jac_star;
       u_pro    = u_pro_star;
       u_res    = u_res_star;
    end
    
    %% Adaptive random walk covariance
    thetasave(i,:) = theta_proposal;
    % Adaptive scale for proposal distribution
    if i > 50
        scale = utils.updateScale(scale,C1,t_accept,i,n_params);
        if (i > N_corr)
            Sigma = cov(thetasave(i-N_corr+1:i,:));
        else
            Sigma = cov(thetasave(1:i,:));
        end
%         V1 = jitChol(V1);
    end

    %% Store the output
    obj.Post.Theta(i,:) = cell2mat(struct2cell(theta));
    obj.Post.Scale(i)   = scale;
    if(~strcmp(savename,''))
        if mod(i,save_num)==0
            save(savename,'obj')
        end
    end
end
toc
obj.Post.CPU = toc;
end

