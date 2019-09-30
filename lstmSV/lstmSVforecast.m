function out = lstmSVforecast(y,z0,theta,mdl)

N = mdl.forecast.N;
t0 = mdl.forecast.t0;
alpha = mdl.forecast.alpha;
K1 = mdl.forecast.K1;
K2 = mdl.forecast.K2;

T = length(y);
volatility_forecast = NaN(T,1);
volatility_filtered = NaN(T,1);
violate = zeros(T,1);

z = zeros(N,T); 
h = zeros(N,T);
eta = zeros(N,T);
memory_cell = zeros(N,T);
weights = zeros(N,T);
indx = zeros(N,T);     

crps = 0; % predictve score
pps = 0;
quantile_score = 0;
hit = 0; % percentage of y instances below forecast VaR
mae = 0; % Min absolute error

% sample particles at time t = 1                               
t = 1; 
eta(:,t) = theta.beta0 + normrnd(0,sqrt(theta.sigma2),N,1);
z(:,t) = eta(:,t) + theta.phi*z0;
logw = -0.5*log(2*pi) - 0.5*z(:,t) - 0.5*(y(t))^2*exp(-z(:,t));
   
% Numerical stabability
weights(:,t) = exp(logw-max(logw));

% Estimate marginal likelihood
sir_llh = log(mean(weights(:,t))) + max(logw);

% Normalize weigths
weights(:,t) = weights(:,t)./sum(weights(:,t));

for t = 2:T
    % Resampling
    indx(:,t) = rs_multinomial(weights(:,t-1));
    z(:,t-1) = z(indx(:,t),t-1);
    h(:,t-1) = h(indx(:,t),t-1);
    eta(:,t-1) = eta(indx(:,t),t-1);
    memory_cell(:,t-1) = memory_cell(indx(:,t),t-1);

    volatility_filtered(t-1) = mean(exp(z(:,t-1)));

    % Generate particles at time t>=2. 
    x_d = activation(theta.v_d*eta(:,t-1) + theta.w_d*h(:,t-1) + theta.b_d,'Tanh'); % data input
    g_i = activation(theta.v_i*eta(:,t-1) + theta.w_i*h(:,t-1) + theta.b_i,'Sigmoid'); % input gate
    g_o = activation(theta.v_o*eta(:,t-1) + theta.w_o*h(:,t-1) + theta.b_o,'Sigmoid'); % output gate
    g_f = activation(theta.v_f*eta(:,t-1) + theta.w_f*h(:,t-1) + theta.b_f,'Sigmoid'); % output gate

    memory_cell(:,t) = g_i.*x_d + g_f.*memory_cell(:,t-1); % update recurrent cell
    h(:,t) = g_o.*tanh(memory_cell(:,t));

    % Calculate weights for particles at time t>=2. 
    eta(:,t) = theta.beta0 + theta.beta1*h(:,t) + normrnd(0,sqrt(theta.sigma2),N,1);    
    z(:,t) = eta(:,t) + theta.phi*z(:,t-1); 
                    
    if t>=t0
        z_forecast = z(:,t); 
        eta_forecast = eta(:,t); 
        h_forecast = h(:,t); 
        memory_cell_forecast = memory_cell(:,t);                        
        for k = 2:K1               
            [z_forecast,eta_forecast,h_forecast,memory_cell_forecast] = lstmSVforward(theta,z_forecast,eta_forecast,h_forecast,memory_cell_forecast);
        end
        volatility_forecast(t) = mean(exp(z_forecast));
        if t+K1-1<=T
            if abs(y(t+K1-1))>norminv(1-alpha/2)*sqrt(volatility_forecast(t))
                violate(t) = 1;
            else
                violate(t) = 0;
            end
            pps = pps+0.5*log(2*pi)+0.5*log(volatility_forecast(t))+(y(t+K1-1))^2/2/volatility_forecast(t);
            crps = crps+crps_normal(y(t+K1-1),0,volatility_forecast(t));
            VaR_t = norminv(alpha)*sqrt(volatility_forecast(t));
            quantile_score = quantile_score+(y(t+K1-1)-VaR_t)*(alpha-indicator_fun(y(t+K1-1),VaR_t));
            hit = hit+indicator_fun(y(t+K1-1),VaR_t);
            mae = mae + abs(normrnd(0,0.5*exp(volatility_forecast(t))) - y(t));
        end
    end     
    
    logw = -0.5*log(2*pi) - 0.5*z(:,t) - 0.5*(y(t))^2*exp(-z(:,t));

    % Numerical stabability
    weights(:,t) = exp(logw-max(logw));

    % Estimate marginal likelihood
    sir_llh = sir_llh + log(mean(weights(:,t))) + max(logw);

    % Normalize weigths
    weights(:,t) = weights(:,t)./sum(weights(:,t));   
end
indx_T = rs_multinomial(weights(:,T));
z(:,T) = z(indx_T,T);
h(:,T) = h(indx_T,T);
eta(:,T) = eta(indx_T,T);
memory_cell(:,T) = memory_cell(indx_T,T);
         
volatility_filtered(T) = mean(exp(z(:,T)));
memory_cell(:,T) = memory_cell(indx_T,T);
h(:,T) = h(indx_T,T);

z_forecast = z(:,T); 
eta_forecast = eta(:,T); 
h_forecast = h(:,T); 
memory_cell_forecast = memory_cell(:,T);     
volatility_out_of_sample = zeros(K2,1);
for k = 1:K2               
    [z_forecast,eta_forecast,h_forecast,memory_cell_forecast] = lstmSVforward(theta,z_forecast,eta_forecast,h_forecast,memory_cell_forecast);
    volatility_out_of_sample(k) = mean(exp(z_forecast));
end
           

out.volatility_forecast = volatility_forecast;
out.volatility_filtered = volatility_filtered;
out.violate_array = violate;
out.violate = sum(violate);
out.particles = z;
out.weights = weights;
out.indx = indx;
out.llh = sir_llh;
out.crps = crps;
out.pps = pps/(T-t0+1);
out.volatility_out_of_sample = volatility_out_of_sample;
out.quantile_score = quantile_score/(T-t0+1);
out.hit = hit/(T-t0+1);
out.mae = mae/(T-t0+1);
out.h = h;
out.volatility_out_of_sample = volatility_out_of_sample;
out.m = memory_cell;
out.eta = eta;
end
