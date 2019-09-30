function sir_llh = PFfilter(PFObj,Model)
% PFFILTER run a particle filter
% Input: 
%       PFObj    : A particle filter object storing specifications to run a
%                  particle filter
%       Model    : The model to run the particle filter
% Output:
%       sir_llh  : The estimate log-likelihood of the input Model

% Setting
N = PFObj.NumParticles;        % Number of particles to estimate state at each time step
T = PFObj.NumTimeSteps;        % Number of time steps
y = PFObj.Data;

% Preallocation
PFObj.State         = zeros(N,T);        % States 
PFObj.Weights       = zeros(N,T);        % Weight of each particle
PFObj.AncestorIndex = zeros(N,T);        % Store N resampling index in each of T time step

% sample particles at time t = 1 
t = 1; 
PFObj.CurrentTime = t;
PFObj.State(:,t)  = Model.stateInitialize(N);

% Calculate weights for particles at time = 1
logw = PFObj.MeasurementLikelihoodFcn(PFObj.State(:,t),y(t));

% Numerical stabability
PFObj.Weights(:,t) = exp(logw-max(logw));

% Estimate marginal likelihood
sir_llh = log(mean(PFObj.Weights(:,t))) + max(logw);

% Normalize weigths
PFObj.Weights(:,t) = PFObj.Weights(:,t)./sum(PFObj.Weights(:,t));

for t = 2:T
    PFObj.CurrentTime = t;
   % Resampling
   PFObj.AncestorIndex(:,t) = PFObj.ResamplingFnc(PFObj.Weights(:,t-1));
   PFObj.State(:,t)         = PFObj.StateTransitionFcn(PFObj,Model);
   
   % Calculate weights of particles at the current time step
   logw = PFObj.MeasurementLikelihoodFcn(PFObj);

   % Numerical stabability
   PFObj.Weights(:,t) = exp(logw-max(logw));

   % Estimate marginal likelihood
   sir_llh = sir_llh + log(mean(PFObj.Weights(:,t))) + max(logw);

   % Normalize weigths
   PFObj.Weights(:,t) = PFObj.Weights(:,t)./sum(PFObj.Weights(:,t));
end
      
end

