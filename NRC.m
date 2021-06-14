function [id]= NRC(D,c,y,Dlabels)
%------------------------------------------------------------------------
% NRC classification function
for ci = 1:max(Dlabels)
    coef_c   =  c(Dlabels==ci);
    Dc       =  D(:,Dlabels==ci);
    error(ci) = norm(y-Dc*coef_c,2)^2;
end

index      =  find(error==min(error));
id         =  index(1);