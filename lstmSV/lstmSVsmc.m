function sir_llh = lstmSVsmc(mdl,N,y)
% Particle Filter for the LSTM-SV model

T = length(y);       
z = zeros(N,T); 
h = zeros(N,T);
eta = zeros(N,T);
memory_cell = zeros(N,T);
weights = zeros(N,T);
indx = zeros(N,T);                                % Store N resampling index in each of T time step

% sample particles at time t = 1                               
t = 1; 

% Calculate weights for particles at time = 1
eta(:,t) = mdl.beta0 + normrnd(0,sqrt(mdl.sigma2),N,1);
z(:,t) = eta(:,t) + mdl.phi*log(var(y));
logw = -0.5*log(2*pi) - 0.5*z(:,t) - 0.5*(y(t))^2*exp(-z(:,t));

% Numerical stabability
weights(:,t) = exp(logw-max(logw));

% Estimate marginal likelihood
sir_llh = log(mean(weights(:,t))) + max(logw);

% Normalize weigths
weights(:,t) = weights(:,t)./sum(weights(:,t));

for t = 2:T
   % Calculate resampling index
   indx(:,t) = utils.rs_multinomial(weights(:,t-1));
   
   % Resampling
   z(:,t-1) = z(indx(:,t),t-1);
   eta(:,t-1) = eta(indx(:,t),t-1);
   h(:,t-1) = h(indx(:,t),t-1);
   memory_cell(:,t-1) = memory_cell(indx(:,t),t-1);

   % Generate particles at time t>=2. 
   x_d = utils.activation(mdl.v_d*eta(:,t-1) + mdl.w_d*h(:,t-1) + mdl.b_d,'Tanh');    % data input
   g_i = utils.activation(mdl.v_i*eta(:,t-1) + mdl.w_i*h(:,t-1) + mdl.b_i,'Sigmoid'); % input gate
   g_o = utils.activation(mdl.v_o*eta(:,t-1) + mdl.w_o*h(:,t-1) + mdl.b_o,'Sigmoid'); % output gate
   g_f = utils.activation(mdl.v_f*eta(:,t-1) + mdl.w_f*h(:,t-1) + mdl.b_f,'Sigmoid'); % output gate

   memory_cell(:,t) = g_i.*x_d + g_f.*memory_cell(:,t-1); % update recurrent cell
   h(:,t) = g_o.*tanh(memory_cell(:,t));
   % Calculate weights for particles at time t>=2. 
   eta(:,t) = mdl.beta0 + mdl.beta1*h(:,t) + normrnd(0,sqrt(mdl.sigma2),N,1);    
   z(:,t) = eta(:,t) + mdl.phi*z(:,t-1); 
   logw = -0.5*log(2*pi) - 0.5*z(:,t) - 0.5*(y(t))^2*exp(-z(:,t));

   % Numerical stabability
   weights(:,t) = exp(logw-max(logw));

   % Estimate marginal likelihood
   sir_llh = sir_llh + log(mean(weights(:,t))) + max(logw);

   % Normalize weigths
   weights(:,t) = weights(:,t)./sum(weights(:,t));
end
      
end
