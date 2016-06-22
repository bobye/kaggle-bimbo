#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <numeric>

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <math.h>
#include <assert.h>
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

void linear_regression(double *xx, double *yy, size_t n, size_t p, double *ww) {
  double xi, yi, ei, chisq;
  gsl_matrix *X, *cov;
  gsl_vector *y, *w, *c;

  X = gsl_matrix_alloc (n, p);
  y = gsl_vector_alloc (n);

  c = gsl_vector_alloc (p);
  cov = gsl_matrix_alloc (p, p);
  for (size_t i=0; i<n; ++i) {
    for (size_t j=0; j<p; ++j)
      gsl_matrix_set(X, i, j, xx[i*p+j]);
    gsl_vector_set(y, i, yy[i]);
  }
  {
    gsl_multifit_linear_workspace *work =gsl_multifit_linear_alloc(n, p);
    gsl_multifit_linear (X, y, c, cov, &chisq, work);
    gsl_multifit_linear_free (work);
  }

  for (size_t i=0; i<p; ++i) 
    if (isnan(ww[i] = gsl_vector_get( c, i))) {}

  gsl_matrix_free (X);
  gsl_vector_free (y);
  gsl_vector_free (c);
  gsl_matrix_free (cov);  
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
  unordered_map<int, tuple<float, float, char> > client_group;
  unordered_map<tuple<int, int>, size_t> product_group;
  unordered_map<tuple<int, int>, tuple<float, float, float> > product_group_coeff;
  size_t *next_id, *next_id_prod;
  short int* demands; 
  char* months; 
  int* client_ids;
  size_t count = 1, max_count=74180465, num_of_products=2592;
  next_id = (size_t*) calloc(max_count, sizeof(size_t));
  next_id_prod = (size_t*) calloc(max_count, sizeof(size_t));
  demands = (short int*)    malloc(max_count * sizeof(short int));
  months  = (char*)    malloc(max_count * sizeof(char));
  client_ids = (int*) malloc(max_count * sizeof(int));

  /* scanning training file */
  cout << "File Scan:\n";
  while ((read = getline(&line, &len, train_file)) != -1 && count < max_count) {
    int Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Venta_uni_hoy,Dev_uni_proxima,Demanda_uni_equil;
    float Venta_hoy,Dev_proxima;

    sscanf(line, "%d,%d,%d,%d,%d,%d,%d,%f,%d,%f,%d", &Semana,&Agencia_ID,&Canal_ID,&Ruta_SAK,&Cliente_ID,&Producto_ID,&Venta_uni_hoy,&Venta_hoy,&Dev_uni_proxima,&Dev_proxima,&Demanda_uni_equil);

    months[count]= Semana;
    demands[count] = Demanda_uni_equil;
    client_ids[count] = Cliente_ID;
    {
      auto key = make_tuple(Cliente_ID, Producto_ID, Agencia_ID, (char) Canal_ID);
      auto itr = last_group.find(key);
      if (itr == last_group.end()) {
	last_group[key] = count;
      } else {
	next_id[count] = itr->second;
	itr->second = count;
      }
    }

    {
      auto key = make_tuple(Producto_ID, Agencia_ID);
      auto itr = product_group.find(key);
      if (itr == product_group.end()) {
	product_group[key] = count;
      } else {
	next_id_prod[count] = itr->second;
	itr->second = count;
      }
    }
    

    {
      auto itr = client_group.find(Cliente_ID);
      if (Venta_hoy > 0 || Dev_proxima > 0) {
	if (itr == client_group.end()) {
	  client_group[Cliente_ID] = make_tuple(Venta_hoy, Dev_proxima, 0x01 << (Semana - 3));
	} else {
	  get<0>(itr->second) += Venta_hoy;
	  get<1>(itr->second) += Dev_proxima;
	  get<2>(itr->second) |= 0x01 << (Semana - 3);
	}
      }
    }


    if (count%10000==0 || count == max_count) {
      prt_progress_bar((float) count / (float) max_count);
    }
    count ++;
  }
  fclose(train_file);
  printf("\n");

  /* load product weights */
  /*
  FILE *product_file;
  unordered_map<int, int> p_weight;
  product_file = fopen("product_weight.csv", "r");
  if (product_file == NULL)
    exit(EXIT_FAILURE);
  for (int i=0; i<num_of_products; ++i) {
    int id, w;
    fscanf(product_file, "%d,%d", &id, &w);
    p_weight[id]=w;
  }
  fclose(product_file);
  */

  FILE *aggregate_file;
  size_t size_of_group;
  /* write aggregate data for regression */
  /*
  aggregate_file = fopen("group.csv", "w");
  count = 1;
  size_of_group = last_group.size();
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
  */

  /* write aggregate data for client */
  aggregate_file = fopen("client.csv", "w");
  count = 1;
  size_of_group = client_group.size();
  cout << "Write Client:\n";
  for (auto itr = client_group.begin(); itr != client_group.end(); ++itr) {
    unsigned char value=get<2>(itr->second);
    int denominator = 0;
    while(value > 0) {
      if ( (value & 1) == 1 ) 
	denominator++;
      value >>= 1;
    }
    assert(denominator>0 && denominator<=7);   

    get<0>(itr->second)/=denominator;
    get<1>(itr->second)/=denominator;

    fprintf(aggregate_file, "%d,%.2f,%.2f\n", itr->first, get<0>(itr->second),get<1>(itr->second));
    if (count % 10000 == 0 || count == size_of_group) {
      prt_progress_bar((float) count / (float) size_of_group);
    } 
    count++;
  }
  fclose(aggregate_file);
  printf("\n");


  /* write regression missing data */
  aggregate_file = fopen("regression_data.csv", "w");
  count = 1;
  size_of_group = product_group.size();
  cout << "Write Regression Data:\n";
  for (auto itr = product_group.begin(); itr != product_group.end(); ++itr) {
    size_t jj = itr->second;
    vector<double> x;
    vector<double> y;
    double w[3]; 
    while (jj!=0) {
      int client_id=client_ids[jj];
      x.push_back(1.);
      x.push_back(log(get<0>(client_group[client_id])+1));
      x.push_back(log(get<1>(client_group[client_id])+1));
      y.push_back(log(demands[jj]+1));
      jj = next_id_prod[jj];
    }
    float avg = accumulate(y.begin(), y.end(), 0) / y.size();
    if (y.size() > 10) {
      linear_regression(&x[0], &y[0], y.size(), 3, w);
      if (fabs(w[0]) < 5) {
	fprintf(aggregate_file, "%d,%d,%ld,%lf,%lf,%lf\n", 
		get<0>(itr->first), get<1>(itr->first), 
		y.size(), w[0], w[1], w[2]);	
	product_group_coeff[itr->first] = make_tuple(w[0], w[1], w[2]);
      } else {
	product_group_coeff[itr->first] = make_tuple(avg, 0., 0.);
      }
    } else {      
	product_group_coeff[itr->first] = make_tuple(avg, 0., 0.);
    }
    x.clear(); y.clear();
    if (count % 1000 == 0 || count == size_of_group) {
      prt_progress_bar((float) count / (float) size_of_group);
    } 
    count++;
  }
  fclose(aggregate_file);
  printf("\n");
  product_group.clear();
  free(next_id_prod);
  free(client_ids);
  
  /* write submit files */
  FILE *test_file, *submit_file;
  test_file = fopen("../test.csv", "r");
  submit_file = fopen("submit.csv", "w");

  if (test_file == NULL || submit_file == NULL)
    exit(EXIT_FAILURE);
  read = getline(&line, &len, train_file);
  count = 1; 
  max_count = 6999252;
  cout << "Write Test Submit:\n";
  fprintf(submit_file, "id,Demanda_uni_equil\n");
  while ((read = getline(&line, &len, test_file)) != -1 && count < max_count) {
    int id,Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID;
    sscanf(line, "%d,%d,%d,%d,%d,%d,%d", &id,&Semana,&Agencia_ID,&Canal_ID,&Ruta_SAK,&Cliente_ID,&Producto_ID);
    auto key = make_tuple(Cliente_ID, Producto_ID, Agencia_ID, (char) Canal_ID);
    auto itr = last_group.find(key);
    float logmean=0;
    if (itr != last_group.end()) {      
      logmean = get_logmean(itr->second, demands, next_id);
      fprintf(submit_file, "%d,%.2f\n", id, exp(logmean)-1);    
    } else {
      auto regress=product_group_coeff.find(make_tuple(Producto_ID, Agencia_ID));
      if (regress != product_group_coeff.end()) {
	float estimate = get<0>(regress->second)
	  + get<1>(regress->second) * log(get<0>(client_group[Cliente_ID])+1)
	  + get<2>(regress->second) * log(get<0>(client_group[Cliente_ID])+1);
	if (estimate > 0 && estimate < 15)
	  fprintf(submit_file, "%d,%.2f\n", id, exp(estimate)-1);
	else if (estimate < 0) {
	  fprintf(submit_file, "%d,%.2f\n", id, 0.);
	} else {
	  fprintf(submit_file, "%d,%.2f\n", id, 3.92);    
	}
      } else {
        // do something non-trivial 
	fprintf(submit_file, "%d,%.2f\n", id, 3.92);    
      }
    }
    if (count % 10000 == 0 || count == max_count) {
      prt_progress_bar((float) count / (float) max_count);
    } 
    count++;
  }  
  fclose(test_file);
  fclose(submit_file);
  printf("\n");


  free(next_id); 
  free(demands); free(months); 
  return 0;
}
