function x = inverse_demand_map( id )
%id=floor(2*log(x+ones(size(x))))+ones(size(x));

x=exp((id - 2*ones(size(id)))/2) - ones(size(id));
end

