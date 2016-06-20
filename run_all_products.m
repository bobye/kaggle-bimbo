curs=exec(conn, 'select Producto_ID,Agencia_ID,Canal_ID from product');
curs=fetch(curs);
PIDs=cell2mat(curs.Data);
close(curs);
n=size(PIDs,1);
product=cell(n,1);
product_mm_map=containers.Map('KeyType','double','ValueType','any');
reverseStr = '';
err_idx=[];
%%
tic;
for i=1:n
    try
        data=get_product_data(PIDs(i,1), PIDs(i,2), PIDs(i,3));
        mm=estimate_mm(data.cli_u); 
        product_mm_map(myhash(PIDs(i,1:3)))=mm;
        update(conn, 'product', {'mm'}, {typecast(mm(:), 'uint8')}, ...
            ['where Producto_ID=', num2str(PIDs(i,1)), ...
             ' and Agencia_ID=', num2str(PIDs(i,2)), ...
             ' and Canal_ID=', num2str(PIDs(i,3))]);
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

save('product_mm_map.mat', 'product_mm_map');
