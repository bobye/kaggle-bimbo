function estimate=single_test_sample(id, varargin)
global conn;
global product_mm_map client_mm_map
global test_data

% curs=exec(conn, ['select * from test where id=', num2str(id)]);
% curs=fetch(curs);
% test_sample=cell2mat(curs.Data);
% close(curs);

%%
id=id+1;

if nargin==1
    Semana=num2str(test_data.Semana(id));
    Agencia_ID=num2str(test_data.Agencia_ID(id));
    Canal_ID=num2str(test_data.Canal_ID(id));
    Producto_ID=num2str(test_data.Producto_ID(id));
    Cliente_ID=num2str(test_data.Cliente_ID(id));    
    curs=exec(conn, ['select avg(Demanda_uni_equil) from train where',...
        ' Cliente_ID=', Cliente_ID, ...
        ' and Semana=9 and Agencia_ID=',Agencia_ID, ...
        ' and Canal_ID=',Canal_ID , ...
        ' and Producto_ID=',Producto_ID, ...
        ]);
    curs=fetch(curs);
    if ischar(curs.Data{1}) || isnan(curs.Data{1})
        infer_sample=-1;
    else
        infer_sample=cell2mat(curs.Data);
    end
    close(curs);
else
    infer_sample=varargin{1};
end

%%
% curs=exec(conn, ['select mm from product where Agencia_ID=', Agencia_ID, ...
%     ' and Canal_ID=', Canal_ID, ...
%     ' and Producto_ID=', Producto_ID]);
% curs=fetch(curs);
% mm2=typecast(curs.Data{1}, 'double');
% close(curs);
% mm2=reshape(mm2, [sqrt(length(mm2)), sqrt(length(mm2))]);
% trans2=mm2(demand_mapping_id(infer_sample),:);



%%
% data=get_product_data(Producto_ID, Agencia_ID, Canal_ID);
% mm2=estimate_mm(data.cli_u);
mm2=product_mm_map(myhash([test_data.Producto_ID(id), test_data.Agencia_ID(id), test_data.Canal_ID(id)]));
if test_data.Semana(id)==11
    mm2=mm2*mm2;
end
trans2=mm2(demand_mapping_id(infer_sample),:);
%%
% data=get_client_data(Cliente_ID);
% mm1=estimate_mm(data.prod_u);
mm1=client_mm_map(test_data.Cliente_ID(id));
if test_data.Semana(id)==11
    mm1=mm1*mm1;
end
trans1=mm1(demand_mapping_id(infer_sample),:);
%%
map_table=[
           0
           2
           5
          10
          18
          31
          52
          86
         143
         237
         391
         646
        1067
        1760
        2903
        4787
        7893
        9914]/2;
    
if size(mm2,1) >1 && size(mm1,1) > 1    
ll=min(length(trans1),length(trans2));
trans=sqrt(trans1(1:ll).*trans2(1:ll));
trans=trans(2:end)./sum(trans(2:end));
estimate=sum(trans'.*map_table(1:length(trans)));

elseif size(mm1,1) > 1
trans=trans1;        
trans=trans(2:end)./sum(trans(2:end));
estimate=sum(trans'.*map_table(1:length(trans)));
elseif size(mm2,1) > 1
trans=trans2;        
trans=trans(2:end)./sum(trans(2:end));
estimate=sum(trans'.*map_table(1:length(trans)));
else
estimate=3.9;
end
