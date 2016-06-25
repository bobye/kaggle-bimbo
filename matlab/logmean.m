function [ y ] = logmean( x )
%
y=exp(mean(log(x+1)))-1;

end

