function mm_trans = estimate_mm(units)

states=demand_mapping_id(units);

max_state=max(states(:));
mm_trans=zeros(max_state, max_state);

for i=1:6
    [xu, ~, k]=unique([states(i,:);states(i+1,:)]', 'rows');
    count = histc(k, 1:size(xu, 1));
    ind=sub2ind([max_state, max_state], xu(:,1), xu(:,2));
    mm_trans(ind) = mm_trans(ind)+ count(:);    
end
mm_trans=mm_trans + ones(max_state);
mm_trans=bsxfun(@times, mm_trans, 1./sum(mm_trans, 2));
end