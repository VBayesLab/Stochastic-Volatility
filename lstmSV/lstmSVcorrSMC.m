function sir_llh = lstmSVcorrSMC(theta,N,y,u_pro,u_res)
  
T = length(y);

z = zeros(1,N,T); 
h = zeros(1,N,T);
eta = zeros(1,N,T);
memory_cell = zeros(1,N,T);
weights = zeros(T,N);
indx = zeros(N,T);              % Store N resampling index in each of T time step

% sample particles at time t = 1                               
t = 1; 
eta(:,:,t) = theta.beta0 + sqrt(theta.sigma2)*u_pro(:,t)';
z(:,:,t) = eta(:,:,t)  + theta.phi*log(var(y));

% Calculate weights for particles at time = 1
logw = -0.5*log(2*pi) - 0.5*z(:,:,t) - 0.5*y(t)^2*exp(-z(:,:,t));

% Numerical stabability
weights(t,:) = exp(logw-max(logw));

% Estimate marginal likelihood
sir_llh = log(mean(weights(t,:))) + max(logw);

% Normalize weigths
weights(t,:) = weights(t,:)./sum(weights(t,:));

for t = 2:T
   % Resampling
   indx(:,t) = utils.rs_multinomial_sort(z(:,1:N,t-1),weights(t-1,:),u_res(:,t)');
   z(:,:,t-1) = z(:,indx(:,t),t-1);
   eta(:,:,t-1) = eta(:,indx(:,t),t-1);
   h(:,:,t-1) = h(:,indx(:,t),t-1);
   memory_cell(:,:,t-1) = memory_cell(:,indx(:,t),t-1);
   
   % Generate particles at time t>=2. 
   z_d = utils.activation(theta.v_d*eta(:,:,t-1) + theta.w_d*h(:,:,t-1) + theta.b_d,'Tanh');    % Data input
   g_i = utils.activation(theta.v_i*eta(:,:,t-1) + theta.w_i*h(:,:,t-1) + theta.b_i,'Sigmoid'); % Input gate
   g_o = utils.activation(theta.v_o*eta(:,:,t-1) + theta.w_o*h(:,:,t-1) + theta.b_o,'Sigmoid'); % Output gate
   g_f = utils.activation(theta.v_f*eta(:,:,t-1) + theta.w_f*h(:,:,t-1) + theta.b_f,'Sigmoid'); % Forget gate
   memory_cell(:,:,t) = g_i.*z_d + g_f.*memory_cell(:,:,t-1); % Update recurrent cell
   h(:,:,t) = g_o.*tanh(memory_cell(:,:,t));
   eta(:,:,t) = theta.beta0 + theta.beta1*h(:,:,t) + sqrt(theta.sigma2)*u_pro(:,t)';
   z(:,:,t) = eta(:,:,t) + theta.phi*z(:,:,t-1);

   % Calculate weights for particles at time t>=2. 
   logw = -0.5*log(2*pi) - 0.5*z(:,:,t) - 0.5*y(t)^2*exp(-z(:,:,t));

   % Numerical stabability
   weights(t,:) = exp(logw-max(logw));

   % Estimate marginal likelihood
   sir_llh = sir_llh + log(mean(weights(t,:))) + max(logw);

   % Normalize weigths
   weights(t,:) = weights(t,:)./sum(weights(t,:));
end
end
