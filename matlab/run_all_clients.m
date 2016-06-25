curs=exec(conn, 'select Cliente_ID from client');
curs=fetch(curs);
CLIs=cell2mat(curs.Data);
close(curs);
n=size(CLIs,1);
client=cell(n,1);
client_mm_map=containers.Map('KeyType','double','ValueType','any');
reverseStr = '';
err_idx=[];
%%
tic;
for i=1:n
    try
        data=get_client_data(CLIs(i));
        client_mm_map(CLIs(i))=estimate_mm(data.prod_u);
%         update(conn, 'product', {'mm'}, {typecast(mm(:), 'uint8')}, ...
%             ['where Producto_ID=', num2str(PIDs(i,1)), ...
%              ' and Agencia_ID=', num2str(PIDs(i,2)), ...
%              ' and Canal_ID=', num2str(PIDs(i,3))]);
    catch ME
        warning('not successful for %d\n', i);
        err_idx(end+1)=i;
        rethrow(ME);
    end
    if mod(i,100) == 1
        percentDone = 100 * i / n;
        msg = sprintf('Percent done: %3.1f', percentDone); %Don't forget this semicolon
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));  
    end
end
disp '[done]'
toc;

save('client_mm_map.mat', 'client_mm_map');
