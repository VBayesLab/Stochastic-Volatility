function [llh,std_llh] = IS2fit(obj,model)
%IS2FIT Summary of this function goes here
%   Detailed explanation goes here
    y           = obj.Data;
    burnin      = obj.Burnin;
    M           = obj.NumISParticle;
    N           = obj.NumParticle;
    theta_mcmc  = model.Post.Theta(burnin+1:end,:);

    % Convert all parameters to real values (-inf,+inf)
    theta_proposal = obj.transform(model,theta_mcmc);
    
    % Fit a proposal model
    mdl_proposal = obj.Proposal(theta_proposal,'GM');
    
    logw = zeros(M,1);
    % For each sample of theta
    disp('Starting ....')
    
    parfor i = 1:M
        disp(['Iteration: ',num2str(i)]);
        theta = obj.RandomGenerator(mdl_proposal);
        
        % Transform the proposed theta to parameter space
        theta_inv = obj.inv_transform(model,theta);
        
        % Convert vector of params to a struct
        theta_struct = obj.toStruct(model.NameParams,theta_inv);

        % Estimate log-likelihood using particle filter
        log_lik = obj.LogLikelihood(theta_struct,N,y);

        % Calculate log-prior contribution 
        [log_prior,log_jac] = obj.logPrior(model,theta_struct);

        % Calculate log of proposal density contribution
        proposal_log_density = obj.LogProposal(mdl_proposal,theta);

        % Calculate weight for the current proposed sample of theta
        logw(i) = log_prior + log_jac + log_lik - proposal_log_density;
    end
    
    % Numerical stabability
    max_lw = max(logw);
    weights = exp(logw-max_lw);

    % Estimate log of marginal likelihood and its variance (using Delta method)
    llh = log(mean(weights)) + max_lw;
    variance_llh = (mean(exp(2*(logw-max_lw)))/(mean(weights))^2-1)/length(y);
    std_llh = sqrt(variance_llh);
    
    disp('End!')
end

