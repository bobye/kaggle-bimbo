function data = get_client_data(client_id )
global conn;
curs = exec(conn,['select Producto_ID,Agencia_ID,Canal_ID,Semana,avg(Demanda_uni_equil)', ...
    ' from train where Cliente_ID=', num2str(client_id), ...
    ' group by Semana,Agencia_ID,Canal_ID,Producto_ID']);
curs = fetch(curs);
cols = cell2mat(curs.Data);
close(curs);
product_ids=unique(myhash(cols(:,1:3)));
data.prod_u=-ones(7,length(product_ids));
if size(data.prod_u,2)>0
for i=1:7
    mdata=cols(cols(:,4)==i+2,:);
    if ~isempty(mdata)
        mem=ismember(product_ids, unique(myhash(mdata(:,1:3))));        
        data.prod_u(i,mem)= mdata(:,end);
    end
end
end
end

