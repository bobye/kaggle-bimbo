function data = get_product_data( product_id, agency_id, channel_id)
global conn

curs = exec(conn,['select Cliente_ID,Semana,avg(Demanda_uni_equil) from train where Agencia_ID=', num2str(agency_id), ...
    ' and Canal_ID=', num2str(channel_id), ' and Producto_ID=', num2str(product_id), ...
    ' group by Cliente_ID, Semana' ]);
curs = fetch(curs);
cols = cell2mat(curs.Data);
close(curs);
client_ids=unique(cols(:,1));
% data.cli_v=zeros(7,length(client_ids));
% data.cli_d=zeros(7,length(client_ids));
data.cli_u=-ones(7,length(client_ids));
if size(data.cli_u, 2)>0
for i=1:7       
    mdata=cols(cols(:,2)==i+2,:);
    if ~isempty(mdata)
        data.cli_u(i,ismember(client_ids, unique(mdata(:,1))))= mdata(:,3);
    end
end
end
end

