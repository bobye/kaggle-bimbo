#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>
#include <tuple>

#include "auxilary.hh"

inline void prt_progress_bar(float progress) {
  using namespace std;
  int barWidth = 70;
  cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos) cout << "=";
    else if (i == pos) cout << ">";
    else cout << " ";
  }
  cout << "] " << int(progress * 100.0) << " %\r";
  cout.flush();
}

inline float get_logmean(size_t jj, short int *demands, size_t * next_id) {
  float logmean=0;
  int n_logmean=0;
  while (jj != 0) {
    logmean += log(1+demands[jj]);
    n_logmean++;
    jj=next_id[jj];
  }
  logmean /= n_logmean;
  return logmean;
}

int main() {
  using namespace std;

  /* scan through all data */
  FILE *train_file;
  train_file=fopen("../train.csv", "r");
  if (train_file == NULL)
    exit(EXIT_FAILURE);

  /* basic line reader utility */
  char *line = NULL; 
  size_t len = 0;
  ssize_t read;
  read = getline(&line, &len, train_file);   // skip the first line

  /* basic data structure */
  unordered_map<tuple<int, int, int, char>, size_t> last_group;
  size_t* next_id;
  short int* demands; 
  char* months; 
  size_t count = 1, max_count=74180465;
  next_id = (size_t*) calloc(max_count, sizeof(size_t));
  demands = (short int*)    malloc(max_count * sizeof(short int));
  months  = (char*)    malloc(max_count * sizeof(char));


  /* scanning training file */
  cout << "File Scan:\n";
  while ((read = getline(&line, &len, train_file)) != -1 && count < max_count) {
    int Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Venta_uni_hoy,Dev_uni_proxima,Demanda_uni_equil;
    float Venta_hoy,Dev_proxima;

    sscanf(line, "%d,%d,%d,%d,%d,%d,%d,%f,%d,%f,%d", &Semana,&Agencia_ID,&Canal_ID,&Ruta_SAK,&Cliente_ID,&Producto_ID,&Venta_uni_hoy,&Venta_hoy,&Dev_uni_proxima,&Dev_proxima,&Demanda_uni_equil);

    months[count]= Semana;
    demands[count] = Demanda_uni_equil;
    auto key = make_tuple(Cliente_ID, Producto_ID, Agencia_ID, (char) Canal_ID);
    auto itr = last_group.find(key);
    if (itr == last_group.end()) {
      last_group[key] = count;
    } else {
      next_id[count] = itr->second;
      itr->second = count;
    }
    if (count%10000==0 || count == max_count) {
      prt_progress_bar((float) count / (float) max_count);
    }
    count ++;
  }
  fclose(train_file);
  free(line);
  printf("\n");


  /* write aggregate data for matrix factorization */
  FILE *aggregate_file;
  aggregate_file = fopen("group.csv", "w");
  count = 1;
  size_t size_of_group = last_group.size();
  cout << "Write Logmean:\n";
  for (auto itr = last_group.begin(); itr != last_group.end(); ++itr) {
    fprintf(aggregate_file, "%d,%d,%d,%d,", get<0>(itr->first),get<1>(itr->first),get<2>(itr->first),(int) get<3>(itr->first));
    size_t jj = itr->second;
    float logmean;
    logmean = get_logmean(jj, demands, next_id);
    fprintf(aggregate_file, "%.2f\n", exp(logmean)-1);

    if (count % 10000 == 0 || count == size_of_group) {
      prt_progress_bar((float) count / (float) size_of_group);
    } 
    count++;
  }
  fclose(aggregate_file);
  printf("\n");


  /* write submit files */
  FILE *test_file, *submit_file;
  test_file = fopen("../test.csv", "r");
  submit_file = fopen("submit.csv", "w");

  if (test_file == NULL)
    exit(EXIT_FAILURE);
  read = getline(&line, &len, train_file);
  count = 1; 
  max_count = 6999252;
  cout << "Write Test Submit:\n";
  fprintf(submit_file, "id,Demanda_uni_equil\n");
  while ((read = getline(&line, &len, test_file)) != -1 && count < max_count) {
    int id,Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID;
    sscanf(line, "%d,%d,%d,%d,%d,%d,%d", &id,&Semana,&Agencia_ID,&Canal_ID,&Ruta_SAK,&Cliente_ID,&Producto_ID);
    auto key = make_tuple(Cliente_ID, Producto_ID, Agencia_ID, Canal_ID);
    auto itr = last_group.find(key);
    float logmean=0;
    if (itr == last_group.end()) {      
      // do something non-trivial 
      fprintf(submit_file, "%d,%.2f\n", id, 0.);    
    } else {
      logmean = get_logmean(itr->second, demands, next_id);
      fprintf(submit_file, "%d,%.2f\n", id, exp(logmean)-1);    
    }
    if (count % 10000 == 0 || count == max_count) {
      prt_progress_bar((float) count / (float) max_count);
    } 
    count++;
  }  
  fclose(test_file);
  fclose(submit_file);
  free(line);
  printf("\n");

  return 0;
}
