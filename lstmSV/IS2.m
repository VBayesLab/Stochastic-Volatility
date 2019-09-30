classdef IS2 < handle 
    %IS2 Class to perform marginal likelihood estimation using IS2
    
    properties
        ModelToFit
        NumParticle     % Number of particles in particle filter
        NumISParticle   % Number of IS particle
        LogLikelihood   % Handle of the function implementing log-likelihod
        Proposal        % Handle of the function estimating the proposal
        LogProposal     % Log of the pdf of the proposal distribution
        RandomGenerator % Handle of the function generating random vector from the proposal distribution
        Data            % Input data
        Burnin          % Length of burnin period of MCMC samples
        Marllh          % log-maginal likelihood estimation
        StdMarllh       % Variance of log-marginal likelihood estimation
        Seed            % Random seed
    end
    
    methods
        %% Constructor
        function obj = IS2(data,model,varargin)
            
            % Assign values for class properties
            obj.NumParticle     = 2000;       % Number of particles in particle filter
            obj.NumISParticle   = 5000;       % Number of IS particle
            obj.Data            = data;
            obj.Proposal        = @IS2proposal;
            obj.LogProposal     = @IS2LogProposalPdf;
            obj.RandomGenerator = @IS2randomGenerator;
            obj.Burnin     = 0;
            obj.Marllh     = NaN;
            obj.StdMarllh  = NaN;
            obj.Seed       = NaN;
            
            obj.ModelToFit = class(model);
            if(strcmp(obj.ModelToFit,'lstmSV'))
                obj.LogLikelihood = @lstmSVsmc;
            elseif(strcmp(obj.ModelToFit,'SV'))
                obj.LogLikelihood = @SVsmc;
            else
            end
            
            % Set user-specified settings
            if nargin>2
                obj = obj.setParams(obj,varargin);
            end
            
            % Set random seed if specified
            if(~isnan(obj.Seed))
                rng(obj.Seed);
            end
            
            % Run IS2 algorithm
            [obj.Marllh,obj.StdMarllh] = IS2fit(obj,model);
        end
        
        %% Set up user-defined settings
        function obj = setParams(obj,varargin)
            paramNames = {'NumParticle'       'NumISParticle'      'LogLikelihood' ...
                          'Proposal'          'Burnin'             'RandomGenerator',...
                          'LogProposal'       'Seed'};
                      
            paramDflts = {obj.NumParticle     obj.NumISParticle    obj.LogLikelihood ...
                          obj.Proposal        obj.Burnin           obj.RandomGenerator...
                          obj.LogProposal     obj.Seed};

           [obj.NumParticle,...
            obj.NumISParticle,...
            obj.LogLikelihood,...
            obj.Proposal,...
            obj.Burnin,...
            obj.RandomGenerator,...
            obj.LogProposal,...
            obj.Seed] = internal.stats.parseArgs(paramNames, paramDflts, varargin{2}{:});           
        end
        
        %% Calculate the log prior and log jacobian of the current model
        function [log_prior,log_jac] = logPrior(obj,model,theta)
            params      = model.Params;
            num_params  = model.NumParams;
            params_name = model.NameParams;
            log_prior = 0;
            log_jac = 0;
            for i=1:num_params
                prior_i = params.(params_name{i}).Prior;
                theta_i = theta.(params_name{i});
                param_i = params.(params_name{i});
                
                % For log-jacobian
                log_jac_offset = param_i.JacobianOffset(theta_i);
                log_jac = log_jac + prior_i.logJacobianRandomFnc(theta_i) + log_jac_offset;
                
                % For log-prior
                theta_i = param_i.PriorTransform(theta_i);
                log_prior = log_prior + prior_i.logPdfFnc(theta_i);
            end
        end
        
        %% Calculate the log prior and log jacobian of the current model
        % Theta is a matrix whose columns are mcmc samples of the
        % corresponding parameters
        function theta_trans = transform(obj,mdl,theta)
            params      = mdl.Params;
            num_params  = mdl.NumParams;
            params_name = mdl.NameParams;
            theta_trans = zeros(size(theta));
            for i=1:num_params
                prior_i = params.(params_name{i}).Prior;
                theta_i = theta(:,i);
                % Transformation for random-walk proposal
%                 theta_trans(:,i) = params.(params_name{i}).transform(theta_i,prior.name);
                theta_trans(:,i) = prior_i.transformForRandomWalkFnc(theta_i);
            end
        end
        
        % Inverse transform after random-walk proposal
        % theta is output from random-walk proposal
        function theta_inv = inv_transform(obj,model,theta)
            params      = model.Params;
            num_params  = model.NumParams;
            params_name = model.NameParams;
            theta_inv   = zeros(num_params,1);
            for i=1:num_params
                prior_i = params.(params_name{i}).Prior;
%                 theta_inv(i) = params.(params_name{i}).inv_transform(theta(i),prior.name);
                theta_inv(i) = prior_i.invTransformAfterRandomWalkFnc(theta(i));
            end
        end
        
        % Convert parameter array to struct if variable name is specified
        function theta_struct = toStruct(obj,var_name,theta)
            for i=1:length(var_name)
                theta_struct.(var_name{i}) = theta(i);
            end
        end    
    end
end

