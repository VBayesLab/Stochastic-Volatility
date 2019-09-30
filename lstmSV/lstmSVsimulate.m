function [y,z,theta] = lstmSVsimulate(model,numObs,varargin)
%LSTMSVSIMULATE Generate samples from lstmSV model

% Set default values
NumPaths = 1;
z0 = [];
theta = [];

% Set-up user defined settings
if nargin>2
    paramNames = {'NumPaths'    'z0'    'Parameter'};
    paramDflts = {NumPaths       z0      theta};
   [NumPaths,...
    z0,...
    theta] = internal.stats.parseArgs(paramNames, paramDflts, varargin{1}{:});
end

% Check if theta is specified and is a vector, then convert theta to struct 
% with field names are model parameter names
if(~isempty(theta) && ~isstruct(theta))
    theta = utils.array2struct(theta,model.NameParams);
end

% If model parameters are not specified then use prior to generate a sample
% of model parameters
if(isempty(theta))
    params      = model.Params;
    num_params  = model.NumParams;
    params_name = model.NameParams;

    for i=1:num_params
    dist = params.(params_name{i}).prior;
    theta.(params_name{i}) = params.(params_name{i}).random_generator(dist);
    end
end

if(isempty(z0))
    z0 = zeros(NumPaths,1);
end

% Create a LSTM object
obj_lstm = LSTM(NumPaths,numObs,theta);


% Pre-allocation
y = zeros(NumPaths,numObs);
z = zeros(NumPaths,numObs);

% For t=1
t = 1;
z(:,t) = obj_lstm.eta(:,t) + theta.phi*z0;
y(:,t) = exp(0.5*z(:,t)).*randn(NumPaths,1);

% Simulation from t = 2
for t=2:numObs
    obj_lstm = forward(obj_lstm,theta,t-1);
    z(:,t)   = obj_lstm.eta(:,t) + theta.phi*z(:,t-1);
    y(:,t)   = exp(0.5*z(:,t)).*randn(NumPaths,1);
end

end

