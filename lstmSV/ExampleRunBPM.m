clear
clc

%% Load data
data = load('../Data/SP500_weekly.mat');
T = 1000;                % In-sample size
y = data.y(1:T);         % In-sample data

% Define a save name to store the results during sampling phase
idx = 1;
DateVector = datevec(date);
[~, MonthString] = month(date);
date_time = ['_',num2str(DateVector(3)),'_',MonthString,'_'];
ver = ['v',num2str(idx)];
name = ['Results_lstmSV_SP500',date_time,ver];

%% Bayesian inference
% Create a lstmSV object with defauts properties
Model = lstmSV();

% Create a Blocking Pseudo-Marginal object, setting random seed property
sampler = BPM('Seed',1,...
              'SaveAfter',100,...
              'NumMCMC',100000);

% Set saving name (optional)
sampler.SaveFileName = name;

% Estimate using BPM 
lstmSV_fit = estimate(sampler,Model,y);

%% Estimate marginal likelihood with IS2
lstmSV_fit.Post.IS2 = IS2(y,lstmSV_fit,...
                         'NumParticle',1000,...
                         'NumISParticle',5000,...
                         'Burnin',10000,...
                         'Seed',1);
disp(['Marginal likelihood: ',num2str(lstmSV_fit.Post.IS2.Marllh)]);

%% Forecast


