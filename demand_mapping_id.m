function id = demand_mapping_id( x )
id=floor(2*log(x+ones(size(x))))+2*ones(size(x));
id(isinf(id))=1;
end

