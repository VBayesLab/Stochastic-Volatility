classdef Distribution < handle
    %DISTRIBUTION A Superclass to define a probability distribution
    
    properties
        Name
        Parameter                     % Distribution parameters, must be array
        Dimension                     % Dimensionality, must be array
        TransformForRandomWalk
        InvTransformForRandomWalk
        LogJacobianForRandomWalk
        LogPdf
        RandomGenerator
    end
    
    %% Constructor
    methods
        function obj = Distribution(varargin)
            % Set up some default values
%             obj.Name = 'Normal';
            obj.Dimension = [1,1];
%             obj.Parameter = [0,0.1];
            
            % User-defined settings
            if nargin > 0
                paramNames = {'Name'                          'Parameter'   ...
                              'Dimension'                     'Transform'   ...
                              'InvTransform'                  'LogJacobian' ...
                              'LogPdf'                        'RandomGenerator'};
                          
                paramDflts = {obj.Name                        obj.Parameter ...
                              obj.Dimension                   obj.TransformForRandomWalk ...
                              obj.InvTransformForRandomWalk   obj.LogJacobianForRandomWalk ...
                              obj.LogPdf                      obj.RandomGenerator};

               [obj.Name,...
                obj.Parameter,...
                obj.Dimension,...
                obj.TransformForRandomWalk,...
                obj.InvTransformForRandomWalk,...
                obj.LogJacobianForRandomWalk,...
                obj.LogPdf,...
                obj.RandomGenerator] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});
                
                if(~isa(obj.TransformForRandomWalk,'function_handle'))
                    obj.TransformForRandomWalk = obj.setTransformForRandomWalkFnc();
                end

            end
        end
    end
    
    methods 
        %% Calculate log-pdf
        function log_pdf = logPdfFnc(obj,theta)
            if(isa(obj.LogPdf,'function_handle'))
                log_pdf = obj.LogPdf(obj.Parameter,theta);
            else
                a = obj.Parameter(1);
                b = obj.Parameter(2);
                switch obj.Name
                    case 'Normal'
                        log_pdf = log(normpdf(theta,a,sqrt(b)));
                    case 'IG'
                        log_pdf = a*log(b)-gammaln(a)-(a+1)*log(theta)-b./theta;
                    case 'Beta'
                        log_pdf = log(betapdf(theta,a,b));
                    case 'Gamma'
                        log_pdf = log(gampdf(theta,a,b));
                    case 'Cauchy'
                        log_pdf = log(b./(pi*(b.^2 +(theta-a).^2)));
                    case 'Uniform'
                        log_pdf = log(unifpdf(theta,a,obj.Parameter(2)));
                end
            end
        end

        %% Generate random number
        function rnd_number = randomGeneratorFnc(obj,varargin)
            if(isa(obj.RandomGenerator,'function_handle'))
                rnd_number = obj.RandomGenerator(obj.Parameter,varargin);
            else
                a = obj.Parameter(1);
                b = obj.Parameter(2);
                % Dimensionality of random number array
                if nargin > 1
                    dim = varargin{1};
                else
                    dim = [1,1];
                end

                switch obj.Name
                    case 'Normal'
                        rnd_number = normrnd(a,sqrt(b),dim);
                    case 'IG'
                        rnd_number = 1./random('gam',a,1/b,dim);
                    case 'Beta'
                        rnd_number = betarnd(a,b,dim);
                    case 'Gamma'
                        rnd_number = random('gam',a,b,dim);
                    case 'Cauchy'
                        rnd_number = a + b*tan(pi*(rand(dim)-1/2));
                    case 'Uniform'
                        rnd_number = rand(a,b,dim);
                end
            end
        end
        
        %% Transformation for random-walk proposal
        function numTransform  = transformForRandomWalkFnc(obj,value)
            % If user specify the transformation as a function handle, use it
            if(isa(obj.TransformForRandomWalk,'function_handle'))
                numTransform = obj.TransformForRandomWalk(value,varargin);
            else % Otherwise, use some pre-defined transformations
                transform_type = obj.TransformForRandomWalk;
                switch transform_type
                    case 'Linear'
                        numTransform = value;
                    case 'Log'
                        numTransform = log(value);
                    case 'Logit'
                        numTransform = log(value./(1-value));
                end
            end
        end

        %% Inverse of the transformation for random-walk proposal
        function numInvTransform = invTransformAfterRandomWalkFnc(obj,value)
            if(isa(obj.InvTransformForRandomWalk,'function_handle'))
                numInvTransform = obj.TransformForRandomWalk(value,varargin);
            else
                type = obj.TransformForRandomWalk;
                switch type
                    case 'Linear'
                        numInvTransform = value;
                    case 'Log'
                        numInvTransform = exp(value);
                    case 'Logit'
                        numInvTransform = utils.sigmoid(value);
                end
            end
        end
        
        %% Log of Jacobian for the transformation
        function logJacobian = logJacobianRandomFnc(obj,value)
            if(isa(obj.LogJacobianForRandomWalk,'function_handle'))
            else
                type = obj.TransformForRandomWalk;
                switch type
                    case 'Linear'
                        logJacobian = 0;
                    case 'Log'
                        logJacobian = log(value);
                    case 'Logit'
                        logJacobian = log(value) + log(1-value);
                end
            end
        end
    end
    
    %% Validating class properties
    methods
        % This will be automatically called when the property Name is assigned
        function transform = setTransformForRandomWalkFnc(obj)
            switch obj.Name
                case 'Normal'
                    transform = 'Linear';
                case 'IG'
                    transform = 'Log';
                case 'Beta'
                    transform = 'Logit';
                case 'Gamma'
                    transform = 'Log';
                case 'Cauchy'
                    transform = 'Log';
                case 'Uniform'
                    transform = 'Linear';
            end
        end
    end
end

%% Define custom validation functions
function mustBeString(a)
    if ~ischar(a)
        error('Distribution name must be a string')
    end
end

%% For property validation, check these links
% https://au.mathworks.com/help/matlab/matlab_oop/property-size-and-class-validation.html#bvklfs7-1
% https://au.mathworks.com/help/matlab/matlab_oop/property-validator-functions.html