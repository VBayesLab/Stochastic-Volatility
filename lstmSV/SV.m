classdef SV < handle & matlab.mixin.CustomDisplay
    %STOCHASTICVOLATILITY The class for stochastic Volatility Models
    
    %% Class properties
    properties 
        ModelName              % Model name 
        Distribution           % Distribution of the innovation 
        NumParams              % Number of parameters
        NameParams             % Parameter names
        Params                 % Struct to store model parameters
        Post                   % Struct to store posterior samples and information about estimation
        Forecast               % Struct to store forecast results
        Diagnosis              % Struct to store in-sample diagnotis
        StateInit              % The option for z0
        StateTransitionFcn     % Handle of function to define the state transition equation
        MeasurementFcn         % Handle of function to define the measurement equation
        LogLikelihood          % Handle of the log-likelihood function
        StateInitialization    % Start value for state
        ParamValues            % Current values of all parameters
    end

    %% Method for object display
    methods (Access = protected)
       % Display object properties
       function propgrp = getPropertyGroups(~)
          proplist = {'Distribution','NumParams'};
          propgrp = matlab.mixin.util.PropertyGroup(proplist);
       end
       
       % Display footer
       function header = getHeader(obj)
          if ~isscalar(obj)
             header = getHeader@matlab.mixin.CustomDisplay(obj);
          else
             newHeader1 = '    Stochastic Volatility Model:';
             newHeader2 = '    ----------------------------';
             newHeader = {newHeader1
                          newHeader2};
             header = sprintf('%s\n',newHeader{:});
          end
      end
    end
    
    methods         
        %% Class Constructor
        function obj = SV(varargin)
            obj.ModelName      = 'SV';
            obj.Distribution   = 'Gaussian';
            obj.Params.mu      = Parameter('Name','mu');
            obj.Params.sigma2  = Parameter('Name','sigma2' ,...
                                           'Prior', Distribution('Name','IG','Parameter',[2.5,0.25]));
            obj.Params.phi     = Parameter('Name','phi'   ,...
                                           'Prior', Distribution('Name','Beta','Parameter',[20,1.5]),...
                                           'PriorTransform',@(x)(x+1)/2,...
                                           'PriorInvTransform',@(x)2*x-1,...
                                           'JacobianOffset',@(x)log(0.5));
            obj.MeasurementFcn = @SVmeasurementFnc;
            obj.StateTransitionFcn = @SVstateTransitionFnc;
            obj = obj.setPropertiesFnc();
            
            % If user specifies custom state transision and measurement
            % equations
            if nargin > 0
                paramNames = {'MeasurementFcn'       'StateTransitionFcn'};
                paramDflts = {obj.MeasurementFcn     obj.StateTransitionFcn};

               [obj.MeasurementFcn,...
                obj.StateTransitionFcn] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                  
            end
        end
        
        function obj = setPropertiesFnc(obj)
            obj.NameParams     = fieldnames(obj.Params);
            obj.NumParams      = length(obj.NameParams);
        end
        
        %% Initialize state
        function StateInit = stateInitialize(obj,NumParticle)
            StateInit = obj.ParamValues.mu + ...
                        sqrt(obj.ParamValues.sigma2/(1-obj.ParamValues.phi^2))*randn(1,NumParticle);
        end
        
        %% Customize sampling message during the sampling phase
        function print(theta,iteration)
            disp(['Iteration: ',num2str(iteration), '|mu: ',num2str(theta.mu),'|phi: ',num2str(theta.phi),'|sigma2: ',num2str(theta.sigma2)]);    
        end
    end
    
    methods
        %% Forecast using samples from posterior distribution
        
        %% Model Diagnostics
        
        %% Plot some useful figures
    end
end

%% Define measurement function for SV model
function logYgivenZ = SVmeasurementFnc(Theta,CurrentState,CurrentObs)
    logYgivenZ = -0.5*log(2*pi) - 0.5*CurrentState - 0.5*CurrentObs^2.*exp(-CurrentState);
end

%% Define state transition function for SV model
function NewState = SVstateTransitionFnc(Theta,OldState,RandomNumber)
    NewState = Theta.mu + Theta.phi*(OldState - Theta.mu) + sqrt(Theta.sigma2)*RandomNumber;
end






