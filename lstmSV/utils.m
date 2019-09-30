classdef utils
    %UTILS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Static)
        
        %% Calculate activation function
        function output = activation(z,text)
            switch text
                case 'Linear'
                    output = z;
                case 'Sigmoid'
                    output = 1.0 ./ (1.0 + exp(-z));
                case 'Tanh'
                    output = tanh(z);
                case 'ReLU'
                    output = max(0,z);
            end
        end
        
        %% Calculate sigmoid function
        function output = sigmoid(z)
            output = 1.0 ./ (1.0 + exp(-z));
        end
        
        %% Calculate log pdf of some distribution
        function log_pdf = logpdf(theta,prior)
            switch prior{1}
                case 'normal'
                    log_pdf = utils.logNormal(theta,prior{2}(1),sqrt(prior{2}(2)));
                case 'IG'
                    log_pdf = utils.logIG(theta,prior{2}(1),prior{2}(2));
                case 'beta'
                    log_pdf = utils.logBeta(theta,prior{2}(1),prior{2}(2));
            end
        end
        % Inverse Gamma distribution
        function logpdf = logIG(x,a,b,varargin)
            logpdf = a*log(b) - gammaln(a) - (a+1)*log(x) - b./x;
        end
        
        % Normal distribution
        function logpdf = logNormal(x,mu,var,varargin)
            logpdf = log(normpdf(x,mu,sqrt(var)));
        end
        
        % Beta distribution
        function logpdf = logBeta(x,a,b,varargin)
            logpdf = log(betapdf(x,a,b));
        end
        
        %% Some log_jacobian handler
        % Note: value is the value of the original parameter, not the one
        % from the random-walk proposal
        function log_jac = log_jacobian(value,type)
            switch type
                case 'linear'
                    log_jac = 0;
                case 'log'
                    log_jac = log(value);
                case 'logit'
                    log_jac = log(value) +  log(1-value);
            end
        end
        
        %% Tranformation based on prior
        % obj is a Parameter object
        function out = transform(value,type)
            switch type
                case 'linear'
                    out = value;
                case 'log'
                    out = log(value);
                case 'logit'
                    out = log(value./(1-value));
            end
        end
        
        %% Inverse tranformation after obtain parameter from random walk proposal
        % obj is a Parameter object
        function out = inv_transform(value,type)
            switch type
                case 'linear'
                    out = value;
                case 'log'
                    out = exp(value);
                case 'logit'
                    out = utils.sigmoid(value);
            end
        end
        
        %% Random number generator
        function rng_out = random_generator(dist,varargin)
            switch dist.name
                case 'normal'
                    rng_out = normrnd(dist.val(1),sqrt(dist.val(2)));
                case 'gamma'
                    rng_out = random('gam',dist.val(1),dist.val(2));
                case 'IG'
                    rng_out = 1./random('gam',dist.val(1),1/dist.val(2));
                case 'beta'
                    temp = betarnd(dist.val(1),dist.val(2));
                    rng_out = 2*temp - 1; 
            end
        end
        
        %% Update scale factor of covariance matrix of the random-walk proposal
        function theta = updateScale(sigma2,acc,p,i,d)
            T = 200;
            alpha = -norminv(p/2);
            c = ((1-1/d)*sqrt(2*pi)*exp(alpha^2/2)/(2*alpha) + 1/(d*p*(1-p)));
            Theta = log(sqrt(abs(sigma2)));
            Theta = Theta + c*(acc-p)/max(T, i/d);
            theta = (exp(Theta));
            theta = theta^2;
        end
        
        function [B_var] = jitChol(B_var)
            % Cholesky decompostion
            [~,p] = chol(B_var);
            if p>0
                min_eig = min(eig(B_var));
                d       = size(B_var,1);
                delta   = max(0,-2*min_eig+10^(-5)).*eye(d);
                B_var   = B_var+delta;
            end
        end
        
        %% Resampling
        % Binomial resampling
        function indx = rs_multinomial(w)
            N = length(w);       % Number of particles
            indx = zeros(1,N);   % Preallocate 
            Q = cumsum(w);       % Cumulative sum
            u = sort(rand(1,N)); % Random numbers
            j = 1;
            for i=1:N
                while (Q(j)<u(i))
                    j = j+1;     % Climb the ladder
                end
                indx(i) = j;     % Assign index
            end
        end
        
        % Binomial resampling with sorting
        function indx = rs_multinomial_sort(particles,w,u)
            N = length(w);                       % Number of particles
            orig_index = (1:1:N);
            col = [particles',w',orig_index'];
            col_sort = sortrows(col,1);          % Hilbert sort  
            particles_sort = col_sort(:,1);  
            weight_sort = col_sort(:,2);         % Re-arrange weights according to idx of sorted particles
            orig_ind_sort = col_sort(:,3);
            indx_sort = zeros(1,N);              % Preallocate 
            Q = cumsum(weight_sort);             % Cumulative sum
            Q(end) = 1;                          % Make sure that sum of weights is 1
            u = sort(u);                         % Random numbers
            j = 1;
            for i = 1:N
                while (Q(j)<u(i))
                    j = j + 1;                   % Climb the ladder
                end
                indx_sort(i) = j;                % Assign index
            end
            indx = orig_ind_sort(indx_sort');
            indx = indx';
        end
        
        function indx = rs_multinomial_corr(w,u_res)
            N = length(w); % number of particles
            u_res = reshape(u_res,1,N);
            indx = zeros(1,N); % preallocate 
            Q = cumsum(w); % cumulative sum
            u = sort(u_res); % random numbers
            j = 1;
            for i=1:N
                while (Q(j)<u(i))
                    j = j+1; % climb the ladder
                end
                indx(i) = j; % assign index
            end
        end
        
        %% Forecast scores for Stochastic volatility models 
        function f = crps_normal(x,mu,sigma2)
            % Compute the predictive score (continuous ranked probability score - CRPS)
            % for normal distribution. The smaller CRPS the better prediction. 
            % See Gneiting, T., Raftery, A.: Strictly proper scoring rules, prediction, and
            % estimation. J. Am. Stat. Assoc. 102, 359–378 (2007)

            z = (x-mu)./sqrt(sigma2);
            f = sqrt(sigma2)*(1/sqrt(pi)-2*normpdf(z)-z.*(2*normcdf(z)-1));
        end
        
        function f = indicator_fun(y,quantile)
            if y<=quantile
                f = 1;
            else
                f = 0;
            end
        end
        
        %% Some helper function
        
        % Convert an array to a struct with each given field name for each
        % array element
        function theta_struct = array2struct(theta,name_list)
 
            % Make sure size is match
            theta = reshape(theta,1,length(theta));
            name_list = reshape(name_list,1,length(name_list));
            
            % Conver array to struct with given field names
            theta_struct = cell2struct(num2cell(theta),name_list,2);
        end
        
        function msg_out = errorMSG(identifier)

            switch identifier
                case 'error:DimensionalityPositiveInteger'
                    msg_out = 'Dimensionality must be integer';
                case 'error:DistributionNameMustBeString'
                    msg_out = 'Distribution Name must be string';
                case 'error:DistributionNameIsConstant'
                    msg_out = 'Distribution Name is constant';
                case 'error:NormalDistributionDimensionIncorrect'
                    msg_out = 'The parameter for a normal distribution should be a 1x2 array';
                case 'error:ModelMustBeSpecified'
                    msg_out = 'The fitted model and data must be specified';
                case 'error:DistributionMustBeBinomial'
                    msg_out = 'Binomial distribution option required';
                case 'error:MustSpecifyActivationFunction'
                    msg_out = 'Activation function type requied';
            end
        end
        
    end
    
end

