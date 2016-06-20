% 
% load train.mat
% load test.mat
% PIDs=unique([train_data.Producto_ID, train_data.Agencia_ID, train_data.Canal_ID], 'rows');
% CLIs=unique(test_data.Cliente_ID);
% 
% clear train_data
% clear test_data
%%

% for i=1:7
%     filt=train_data.Semana==i+2;
%     train.Agencia_ID{i}=train_data.Agencia_ID(filt);
%     train.Canal_ID{i}=train_data.Canal_ID(filt);
%     train.Ruta_SAK{i}=train_data.Ruta_SAK(filt);
%     train.Cliente_ID{i}=train_data.Cliente_ID(filt);
%     train.Venta_hoy{i}=train_data.Cliente_ID(filt);
%     train.Venta_uni_hoy{i}=train_data.Venta_uni_hoy(filt);
%     train.Dev_proxima{i}=train_data.Dev_proxima(filt);
%     train.Dev_uni_proxima{i}=train_data.Dev_uni_proxima(filt);
%     train.Producto_ID{i}=train_data.Producto_ID(filt);
%     train.Semana{i}=train_data.Semana(filt);
%     train.Demanda_uni_equil{i}=train_data.Demanda_uni_equil(filt);
% end
%%
global conn;
conn = database('kaggle_inventory','root','bob1989','com.mysql.jdbc.Driver', 'jdbc:mysql://localhost/');

%%


%%
