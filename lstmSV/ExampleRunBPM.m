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
% Create a lstmSV object with defauts attributes
Model = lstmSV();

% Create a Blocking Pseudo-Marginal object, setting random seed attribute
sampler = BPM('Seed',1,...
              'SaveAfter',100,...
              'NumMCMC',200);

% Set saving name (optional)
sampler.SaveFileName = name;

% Estimate using BPM 
lstmSV_fit = estimate(sampler,Model,y);

%% Estimate marginal likelihood with IS2
lstmSV_fit.Post.IS2 = IS2(y,lstmSV_fit,...
                         'NumParticle',100,...
                         'NumISParticle',200,...
                         'Burnin',100,...
                         'Seed',1);
disp(['Marginal likelihood: ',num2str(lstmSV_fit.Post.IS2.Marllh)]);

%% Forecast

