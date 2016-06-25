n=length(test_data.id);
estimate=zeros(n,1);

reverseStr = '';
tic;
batch_size=100;
for i=1:n
    if mod(i,batch_size) == 1
%         curs=exec(conn, ...
%             ['select test.id,exp(avg(log(train.Demanda_uni_equil+1)))-1 from test join train on test.Cliente_ID=train.Cliente_ID and test.Producto_ID=train.Producto_ID and test.Agencia_ID=train.Agencia_ID and test.Canal_ID=train.Canal_ID where ', ...
%               num2str(i-1) '<=test.id and test.id < ' num2str(i-1+batch_size),' group by test.id']);
%           curs=exec(conn, ...
%             ['select test.id,exp(avg(log(train.Demanda_uni_equil+1)))-1 from test join train on test.Cliente_ID=train.Cliente_ID and test.Producto_ID=train.Producto_ID where ', ...
%               num2str(i-1) '<=test.id and test.id < ' num2str(i-1+batch_size),' group by test.id']);  
        curs=exec(conn, ...
            ['select test.id,train.Demanda_uni_equil,if(test.Agencia_ID=train.Agencia_ID and test.Canal_ID=train.Canal_ID, 1, 0) from test join train on test.Cliente_ID=train.Cliente_ID and test.Producto_ID=train.Producto_ID where ', ...
              num2str(i-1) '<=test.id and test.id < ' num2str(i-1+batch_size)]);  
        curs=fetch(curs);
        id_and_demand=cell2mat(curs.Data);
        close(curs);
        demand=-ones(batch_size,1);
        demand(mod(id_and_demand(:,1),batch_size)+1)=id_and_demand(:,2);                

%         curs=exec(conn, ...
%             ['select test.id,avg(train.Demanda_uni_equil) from test join train on test.Cliente_ID=train.Cliente_ID and test.Producto_ID=train.Producto_ID and test.Agencia_ID=train.Agencia_ID and test.Canal_ID=train.Canal_ID where train.Semana=9 and ', ...
%               num2str(i-1) '<= test.id and test.id <' num2str(i-1+1000) ' group by test.id']);
%         curs=fetch(curs);
%         id_and_demand=cell2mat(curs.Data);
%         close(curs);
%         demand=-ones(1000,1);
%         demand(mod(id_and_demand(:,1),1000)+1)=id_and_demand(:,2);
    end
    if (demand(mod(i-1,batch_size)+1) == -1)        
        estimate(i)=single_test_sample(i-1, -1);
    else
        estimate(i)=logmean(id_and_demand(id_and_demand(:,1) == (i-1) & id_and_demand(:,3)==1, 2));
        if isnan(estimate(i))
            estimate(i)=logmean(id_and_demand(id_and_demand(:,1) == (i-1), 2));
        end        
    end
%    estimate(i)=single_test_sample(i-1, demand(mod(i-1,1000)+1));
    if mod(i,1000) == 1
        percentDone = 100 * i / n;
        msg = sprintf('Percent done: %3.1f', percentDone); %Don't forget this semicolon
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg)); 
    end        
end
toc;


%%
save('estimate.mat', 'estimate');
fid = fopen('submit.csv','wt');
fprintf(fid, 'id,Demanda_uni_equil\n');
fclose(fid);
dlmwrite('submit.csv', [(0:n-1)', round(10*estimate)/10],'precision',8, '-append');
