function [z_t] = ADMM(X,y,option)
% X = double(NewTrain_DAT);
% y = double(NewTest_DAT(:,1));
% X  =  X./( repmat(sqrt(sum(X.*X)), [size(X,1),1]) );
% y  =  y./( repmat(sqrt(sum(y.*y)), [size(y,1),1]) );

iter = option.iter;
threshold= option.threshold;
rho = option.rho;
c_t = zeros(size(X,2),size(y,2));
z_t = zeros(size(X,2),size(y,2));
delta_t= zeros(size(X,2),size(y,2));


for t = 1:iter
%     c_t1 = inv(X'*X+rho/2*eye(size(X,2)))*(X'*y+rho/2*z_t+0.5*delta_t);
    %when N>>D
    c_t1 = (2/rho*eye(size(X,2))-((2/rho)^2)*X'*(inv(eye(size(X,1))+(2/rho)*X*X')*X))*(X'*y+rho/2*z_t+0.5*delta_t);
    z_t1 = max(0,c_t1-1/rho*delta_t);
    delta_t1 = delta_t+rho*(z_t1 - c_t1);
%     norm(c_t1 - c_t)
    if ((norm(c_t1 - c_t)<threshold)&&(norm(z_t - z_t1)<threshold) &&(norm(z_t - c_t)<threshold))
        break;
    end
    c_t = c_t1;
    z_t = z_t1;
    delta_t = delta_t1;

end

end