classdef ParticleFilter < handle
    %PARTICLEFILTER A class to implement particle filter to estimate
    %likelihood of a stochastic volatility model
    properties
        NumStateVariables          % Number of states
        NumParticles               % Number of particle to estimate state
        NumTimeSteps               % Number of time steps (length of the series)
        StateTransition            % State transition function
        StateInitialize            % State initialize function
        MeasurementLikelihood      % Measurement function           
        ResamplingMethod           % Resampling method
        ResamplingFnc              % Custom resampling method
        StateEstimationMethod      % The method to estimate the states from particle    
        Weights                    % Store matrix of weights of particles 
        State                      % Store matrix of particles of states
        AncestorIndex              % Store ancestor indexes 
        RandomProposal             % Store random numbers to generate states
        RandomResampling           % Store random numbers for resampling step
        CurrentTime                % Current time step index
        Data                       % Time series data
    end
    
    methods
        
        %% Constructor
        function obj = ParticleFilter(Model,data,varargin)
            obj.NumStateVariables     = 1;              % Univariate state
            obj.NumParticles          = 200;
            obj.NumTimeSteps          = length(data);
            obj.ResamplingMethod      = 'multinomial';  % Multinomial resample
            obj.ResamplingFnc         = @utils.rs_multinomial;
            obj.StateEstimationMethod = 'mean';         % Estimated states are mean of particles
            obj.Data                  = data;
            
            % User-defined setting
            if nargin > 2
                paramNames = {'NumParticles'       'StateTransitionFcn'      'MeasurementLikelihoodFcn' ...
                              'RandomProposal'     'RandomResampling'        'ResamplingFnc'};

                paramDflts = {obj.NumParticles     obj.StateTransition       obj.MeasurementLikelihood ...
                              obj.RandomProposal   obj.RandomResampling      obj.ResamplingFnc};

               [obj.NumParticles,...
                obj.StateTransition,...
                obj.MeasurementLikelihood,...
                obj.RandomProposal,...
                obj.RandomResampling,...
                obj.ResamplingFnc] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});   
            end
            
            % Assign state transition function of the Model
            if(isempty(obj.StateTransition) && ~isempty(Model.StateTransitionFcn))
                obj.StateTransition = Model.StateTransitionFcn;
            else
                disp('A handle of state transition equation must be specified!')
            end
            
            % Assign measurement function of the Model
            if(isempty(obj.MeasurementLikelihood) && ~isempty(Model.MeasurementFcn))
                obj.MeasurementLikelihood = Model.MeasurementFcn;
            else
                disp('A handle of measurement equation must be specified!')
            end
        end
        
        %% Equation to propose state particles in the next time step
        function ProposedState = stateProposalFnc(obj,Model) 
            Theta         = Model.ParamValues;
            OldState      = obj.State(:,obj.CurrentTime);
            RandomNumber  = randn(1,obj.NumParticles);
            ProposedState = obj.StateTransition(Theta,OldState,RandomNumber);
        end
        
        %% Run particle filter
        function obj = estimate(obj,model,data,varargin)
            obj = PFfilter(obj,model,data); 
        end       
    end
end

