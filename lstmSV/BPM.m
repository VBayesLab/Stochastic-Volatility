classdef BPM < handle & matlab.mixin.CustomDisplay
    %BPM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Model           % Instance of model to be fitted
        ModelToFit      % Name of model to be fitted
        NumParticles    % Number of particle in particle filter
        SeriesLength    % Length of the series    
        NumMCMC         % Number of MCMC iterations 
        TargetAccept    % Target acceptance rate
        NumCovariance   % Number of latest samples to calculate adaptive covariance matrix for random-walk proposal
        SaveFileName    % Save file name
        SaveAfter       % Save the current results after each 5000 iteration
        ParamsInit      % Initial values of lstmSV parameters
        Seed            % Random seed
        Post            % Struct to store estimation results
        Initialize      % Initialization method
        LogLikelihood   % Handle of the log-likelihood function
        PrintMessage    % Custom message during the sampling phase
        CPU             % Sampling time    
        Verbose         % Turn on of off printed message during sampling phase
        NumBlock        % Number of blocks
        BlockSize       % Number of time steps per block
        RandomNumberPro % Random numbers for proposal within Partical Filter
        RandomNumberRes % Random numbers for resampling within Particle Filter
    end
    
    % Method for object display
    methods (Access = protected)
       % Display object properties
       function propgrp = getPropertyGroups(~)
          proplist = {'ModelToFit','NumParticles','NumMCMC','TargetAccept',...
                      'NumCovariance','BlockSize','SaveFileName','SaveAfter',...
                      'Verbose','Initialize','Seed'};
          propgrp = matlab.mixin.util.PropertyGroup(proplist);
       end
       
       % Modify header message
       function header = getHeader(obj)
          if ~isscalar(obj)
             header = getHeader@matlab.mixin.CustomDisplay(obj);
          else
             newHeader1 = '    Blocking Pseudo Marginal sampler:';
             newHeader2 = '    ---------------------------------';
             newHeader = {newHeader1
                          newHeader2};
             header = sprintf('%s\n',newHeader{:});
          end
      end
    end
    
    methods
        %% Constructor
        function obj = BPM(varargin)
                        
            % Set default values
            obj.NumParticles   = 200;          
            obj.NumMCMC        = 100000;        
            obj.TargetAccept   = 0.25;         
            obj.NumCovariance  = 1000;         
            obj.SaveFileName   = '';          
            obj.SaveAfter      = 5000;        
            obj.ParamsInit     = [];          
            obj.Seed           = NaN;         
            obj.Post           = struct('Theta', [], 'Scale', [], 'CPU', NaN); % Results
            obj.Initialize     = 'Prior';    % Prior / Custom
            obj.NumBlock       = [];  
            obj.BlockSize      = 5;  
            obj.Verbose        = true;
            % 
            if nargin > 1
                %Parse additional options
                paramNames = {'NumParticles'    'NumMCMC'         'TargetAccept'      'BlockSize'   ...
                              'ParamsInit'      'SaveFileName'    'SaveAfter'         'Verbose'     ...
                              'NumCovariance'   'Seed'            'ModelToFit'        'LogLikelihood'};
                paramDflts = {obj.NumParticles  obj.NumMCMC       obj.TargetAccept    obj.BlockSize  ...
                              obj.ParamsInit    obj.SaveFileName  obj.SaveAfter       obj.Verbose   ...
                              obj.NumCovariance obj.Seed          obj.ModelToFit      obj.LogLikelihood};

                [obj.NumParticles,...
                 obj.NumMCMC,...
                 obj.TargetAccept,...
                 obj.BlockSize,...
                 obj.ParamsInit,...
                 obj.SaveFileName,...
                 obj.SaveAfter,...
                 obj.Verbose,...
                 obj.NumCovariance,...
                 obj.Seed,...
                 obj.ModelToFit,...
                 obj.LogLikelihood] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});

                if (~isempty(obj.ParamsInit))
                    obj.Initialize = 'Custom';
                end
            end
        end
        
        %% Bayesian Estimation 
        function model = estimate(obj,model,data,varargin)

            % User should at least specify the model and data
            if nargin < 2
                error(utils.errorMSG('error:ModelMustBeSpecified'))
            end
            
            % Set random seed if specified
            if(~isnan(obj.Seed))
                rng(obj.Seed);
            end
            
            % Assign model name
            obj.ModelToFit = class(model);
            obj.Model      = model;
            
            % Assign handle of the log-likelihood function
            if(strcmp(obj.ModelToFit,'lstmSV'))
                obj.LogLikelihood  = @lstmSVcorrSMC;
                obj.PrintMessage   = @lstmSVprint;     % Custom print messange during sampling phase
                model.StateInit    = log(var(data));   % Initialization of state 
            end
            
            if(strcmp(obj.ModelToFit,'SV'))
                obj.LogLikelihood = @SVcorrSMC;
            end
            
            % Make sure that data is a row vector
            data = reshape(data,1,length(data));
            
            % If users do not specify initial parameters
            % -> initialize parameters from their priors
            if (strcmp(obj.Initialize,'Prior'))
                obj.ParamsInit = initializeFnc(obj,model);
            end
            
            % Run BPM sampler
            obj.NumBlock = round(length(data))/obj.BlockSize;
            obj = BPMestimate(obj,model,data);
            model.Post = obj.Post;
        end
        
        %% Initialization. Model is an instance of a stochastic volatility model
        function theta_init = initializeFnc(obj,Model)
            % By default, BPM initializes parameters using their priors 
            if(strcmp(obj.Initialize,'Prior'))
                params      = Model.Params;
                num_params  = Model.NumParams;
                params_name = Model.NameParams;

                for i=1:num_params
                    param_i = params.(params_name{i});
                    prior = param_i.Prior;
                    invtransform = param_i.PriorInvTransform;
                    theta_init.(params_name{i}) = invtransform(prior.randomGeneratorFnc);
                end
            end
        end
        
        %% Calculate the log prior and log jacobian of the current model
        function [log_prior,log_jac,theta_trans] = logPriorFnc(obj,model,theta)
            params      = model.Params;
            num_params  = model.NumParams;
            params_name = model.NameParams;
            log_prior = 0;
            log_jac = 0;
            theta_trans = zeros(num_params,1);
            for i=1:num_params
                prior_i = params.(params_name{i}).Prior;
                theta_i = theta.(params_name{i});
                param_i = params.(params_name{i});
                
                % Transformation for random-walk proposal
                theta_trans(i) = prior_i.transformForRandomWalkFnc(theta_i);
                
                % For log-jacobian
                log_jac_offset = param_i.JacobianOffset(theta_i);
                log_jac = log_jac + param_i.Prior.logJacobianRandomFnc(theta_i) + log_jac_offset;
                
                % For log-prior
                theta_i = param_i.PriorTransform(theta_i);
                log_prior = log_prior + param_i.Prior.logPdfFnc(theta_i);
            end
        end

        %% Inverse transform after random-walk proposal
        function theta_inv = invTransformFnc(obj,model,theta)
            params      = model.Params;
            num_params  = model.NumParams;
            params_name = model.NameParams;
            theta_inv   = zeros(num_params,1);
            for i=1:num_params
                prior_i = params.(params_name{i}).Prior;
                theta_inv(i) = prior_i.invTransformAfterRandomWalkFnc(theta(i));
            end
        end
        
        %% Convert parameter array to struct if variable name is specified
        function theta_struct = toStruct(obj,var_name,theta)
            for i=1:length(var_name)
                theta_struct.(var_name{i}) = theta(i);
            end
        end 
        
        %% Get the correlated vector of u using block index
        function u_star = getBlock(obj,u,block_idx,N_block,type)
            u_star = u;
            n_col = size(u,2);                 % Divide blocks in columns
            n_rows = size(u,1);
            block_size = fix(n_col/N_block);   % Number of columns in 1 block
            idx_start = (block_idx-1)*block_size + 1;
            idx_stop = block_idx*block_size;

            if (strcmp(type,'Normal'))
                if block_idx == N_block
                    idx_stop = n_col;
                    u_star(:,idx_start:idx_stop) = randn(n_rows,idx_stop-idx_start+1);
                else
                    u_star(:,idx_start:idx_stop) = randn(n_rows,idx_stop-idx_start+1);
                end
            elseif (strcmp(type,'Uniform')) 
                if block_idx == N_block
                    idx_stop = n_col;
                    u_star(:,idx_start:idx_stop) = rand(n_rows,idx_stop-idx_start+1);
                else
                    u_star(:,idx_start:idx_stop) = rand(n_rows,idx_stop-idx_start+1);
                end
            else
                disp('You must specify random number type')
            end
        end
    end
end

