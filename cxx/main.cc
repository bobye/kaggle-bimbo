#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <numeric>
#include <algorithm>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <math.h>
#include <assert.h>
#include "auxilary.hh"

#define MISSING (-999)


/* basic data structure */
// <Cliente_ID, Producto_ID, Agencia_ID, Canal_ID> => index
std::unordered_map<std::tuple<int, int, int, char>, size_t> last_group;

// Cliente_ID => <Venta_hoy, Dev_uni_proxima, Semana>
std::unordered_map<int, std::tuple<float, float, char> > client_group;
std::unordered_map<int, std::tuple<float, float> > client_group_ro;

// <Producto_ID, Agencia_ID> => index
std::unordered_map<std::tuple<int, int>, size_t> product_group;

// <Producto_ID, Agencia_ID> => LinearRegression<w0,w1,w2>
std::unordered_map<std::tuple<int, int>, std::tuple<float, float, float> > product_group_coeff;

// <Producto_ID> => weight
std::unordered_map<int, float> p_weight;

// <Producto_ID> => count
std::unordered_map<int, float> p_popularity;

// <field, ID> => index
std::unordered_map<std::tuple<char, int>, size_t> feat_index;
size_t *next_id, *next_id_prod;
short int* demands, *sales, *returns; 
char* months; 
int* client_ids;

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


inline unsigned char get_historical_data(size_t jj, float *historical_data, int current_month) {
  int n=0, count_month=0, month=-1;
  unsigned char histo = 0;
  historical_data[0]=0;
  while (jj != 0 && n <6) {
    if (months[jj] != month && month!=-1) {
      historical_data[n]/=count_month; 
      historical_data[n+6]/=count_month; 
      n++; if (n>=6) break;
      count_month=0;
      historical_data[n] = 0;
    }; 
    month = months[jj];
    histo |= 1 << (month - (current_month - 6));
    count_month++;
    historical_data[n] += log(sales[jj]+1);
    historical_data[n+6] += log(returns[jj]+1);
    jj=next_id[jj];    
  }
  for (; jj!=0&&n<6; ++n) {
    historical_data[n] = MISSING;
    historical_data[n+6] = MISSING;
  }
  return histo;
}

inline float get_logmean(size_t jj) {
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

inline float get_loggap(size_t jj) {
  float tmp=0;
  float max=0, min=10000;
  while (jj != 0) {
    tmp = log(1+demands[jj]);
    if (tmp > max) max = tmp;
    if (tmp < min) min = tmp;
    jj=next_id[jj];
  }
  return max-min;
}

inline float get_median(size_t jj) {
  using namespace std;
  vector<short int> cache;
  while (jj != 0) {
    cache.push_back(demands[jj]);
    jj=next_id[jj];
  }
  sort(cache.begin(), cache.end());
  int n=cache.size();
  if (n % 2 == 0) {
    return (cache[n/2] + cache[n/2-1])/2.0;
  } else {
    return cache[n/2];
  }
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



void prepare_features(std::ofstream &out, int Semana, int Cliente_ID, int Producto_ID, int Agencia_ID, int Canal_ID, int Ruta_SAK) {
  using namespace std;
  {
    float tmps[5] = {static_cast<float>(Cliente_ID), 
		     static_cast<float>(Producto_ID), 
		     static_cast<float>(Agencia_ID), 
		     static_cast<float>(Canal_ID), 
		     static_cast<float>(Ruta_SAK)};
    out.write((char*) tmps, sizeof(tmps));
  }
  {
    float historical_data[12]={MISSING, MISSING, MISSING, MISSING, MISSING, MISSING, 
			       MISSING, MISSING, MISSING, MISSING, MISSING, MISSING};
    auto key = make_tuple(Cliente_ID, Producto_ID, Agencia_ID, (char) Canal_ID);
    auto itr = last_group.find(key);
    float logmean, histo = 0;
    if (itr != last_group.end()) {
      histo = get_historical_data(itr->second, historical_data, Semana);
    }
    out.write((char*)historical_data, sizeof(historical_data));
    if (itr != last_group.end()) {
      logmean=get_logmean(itr->second);
    } else {
      logmean=MISSING;
    }
    out.write((char*) &histo, sizeof(float));
    out.write((char*) &logmean, sizeof(float));
  }
  {
    float c[2] ={log(get<0>(client_group[Cliente_ID])+1), log(get<1>(client_group[Cliente_ID])+1)};
    out.write((char*)c, sizeof(c));
    auto regress=product_group_coeff.find(make_tuple(Producto_ID, Agencia_ID));
    float estimate;
    if (regress != product_group_coeff.end()) {
      estimate = get<0>(regress->second)
	+ get<1>(regress->second) * log(get<0>(client_group[Cliente_ID])+1)
	+ get<2>(regress->second) * log(get<1>(client_group[Cliente_ID])+1);
      if (estimate <= 0 || estimate > 15)
	estimate = MISSING;
    } else {
      estimate = MISSING;
    }
    out.write((char*)&estimate, sizeof(float));
  }

  {
    float w;
    if (p_weight.find(Producto_ID) != p_weight.end()) w=p_weight[Producto_ID]; 
    else w=MISSING;
    out.write((char*)&w, sizeof(float));
    if (p_popularity.find(Producto_ID) != p_popularity.end()) w=log(p_popularity[Producto_ID]+1); 
    else w=MISSING;
    out.write((char*)&w, sizeof(float));    
  }
}

void write_ffm_data(std::ofstream &ffm, int Cliente_ID, int Producto_ID, int Agencia_ID, int Canal_ID, int Ruta_SAK, size_t &feat_count) {
  using namespace std;
  auto key = make_tuple(1, Cliente_ID);
  if (feat_index.find(key) == feat_index.end()) {
    feat_index[key] = feat_count++;
  } 
  ffm << "0:" << feat_index[key] << ":1\t";
  key = make_tuple(2, Producto_ID);
  if (feat_index.find(key) == feat_index.end()) {
    feat_index[key] = feat_count++;
  }
  ffm << "1:" << feat_index[key] << ":1\t";
  key = make_tuple(3, Agencia_ID*100 + Canal_ID);
  if (feat_index.find(key) == feat_index.end()) {
    feat_index[key] = feat_count++;
  }
  ffm << "2:" << feat_index[key] << ":1\t";      
  key = make_tuple(4, Ruta_SAK);
  if (feat_index.find(key) == feat_index.end()) {
    feat_index[key] = feat_count++;
  }
  ffm << "3:" << feat_index[key] << ":1\n";      
}

void write_ffm_data_s(std::ofstream &ffm, int Cliente_ID, int Producto_ID, int Agencia_ID, int Canal_ID, int Ruta_SAK, size_t &feat_count) {
  using namespace std;
  ffm << "0:0:" << log(get<0>(client_group_ro[Cliente_ID])+1) << "\t";
  ffm << "0:1:" << log(get<1>(client_group_ro[Cliente_ID])+1) << "\t";
  
  auto key = make_tuple(2, Producto_ID);
  if (feat_index.find(key) == feat_index.end()) {
    feat_index[key] = feat_count++;
  }
  ffm << "1:" << feat_index[key] << ":1\t";
  key = make_tuple(3, Agencia_ID*100 + Canal_ID);
  if (feat_index.find(key) == feat_index.end()) {
    feat_index[key] = feat_count++;
  }
  ffm << "2:" << feat_index[key] << ":1\t";      
  key = make_tuple(4, Ruta_SAK);
  if (feat_index.find(key) == feat_index.end()) {
    feat_index[key] = feat_count++;
  }
  ffm << "3:" << feat_index[key] << ":1\n";      
}


int main(int argc, char* argv[]) {
  using namespace std;

  /* use validation */
  bool use_valid, write_ffm, write_ffm_s, read_knn;
  int valid_month;
  assert(argc == 2);
  if (argv[1][0] >= '0' && argv[1][0] <='9')  {
    use_valid = true;
    valid_month = argv[1][0] - '0';
  }
  else if (argv[1][0] == 't')  use_valid = false;
  else assert(false);

  if (argv[1][1] == 'w') write_ffm = true;
  else if (argv[1][1] == 'r') write_ffm = false;

  if (argv[1][2] == 'w') write_ffm_s = true;
  else if (argv[1][2] == 'r') write_ffm_s = false;

  if (argv[1][3] == 'r') read_knn = true;
  else read_knn = false;
  /* basic line reader utility */
  char *line = NULL; 
  size_t len = 0;
  ssize_t read;
  
  size_t count, t_count, max_count=74180465, num_of_products=2592;
  next_id = (size_t*) calloc(max_count, sizeof(size_t));
  next_id_prod = (size_t*) calloc(max_count, sizeof(size_t));
  demands = (short int*)    malloc(max_count * sizeof(short int));
  sales = (short int*)    malloc(max_count * sizeof(short int));
  returns = (short int*)    malloc(max_count * sizeof(short int));
  months  = (char*)    malloc(max_count * sizeof(char));
  client_ids = (int*) malloc(max_count * sizeof(int));

  int Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Venta_uni_hoy,Dev_uni_proxima,Demanda_uni_equil;
  float Venta_hoy,Dev_proxima;

  if (write_ffm_s) {
    FILE *client_group_ro_file;
    int id; float c0, c1;
    client_group_ro_file = fopen("client_ro.csv", "r");
    assert(client_group_ro_file);
    while (fscanf(client_group_ro_file, "%d,%f,%f\n", &id, &c0, &c1) != EOF)
      client_group_ro[id]=make_tuple(c0, c1);    
    fclose(client_group_ro_file);
  }

  /* scanning training file */
  ifstream train_file_bin; train_file_bin.open("/home/jxy198/kaggle-inventory/cxx/train.bin", ios::binary); assert(train_file_bin);
  ofstream ffm_tr; if (write_ffm) ffm_tr.open("ffm_tr.txt");
  ofstream ffm_tr_s; if (write_ffm_s) ffm_tr_s.open("ffm_tr.s.txt");
  cout << "File Scan:\n";
  t_count = 1; 
  size_t feat_count = 0, feat_count_s=2;
  while (t_count < max_count) {
    train_file_bin.read( (char*) &Semana, sizeof(int) );
    train_file_bin.read( (char*) &Agencia_ID, sizeof(int) );
    train_file_bin.read( (char*) &Canal_ID, sizeof(int) );
    train_file_bin.read( (char*) &Ruta_SAK, sizeof(int) );
    train_file_bin.read( (char*) &Cliente_ID, sizeof(int) );
    train_file_bin.read( (char*) &Producto_ID, sizeof(int) );
    train_file_bin.read( (char*) &Venta_uni_hoy, sizeof(int) );
    train_file_bin.read( (char*) &Venta_hoy, sizeof(float) );
    train_file_bin.read( (char*) &Dev_uni_proxima, sizeof(int) );
    train_file_bin.read( (char*) &Dev_proxima, sizeof(float) );
    train_file_bin.read( (char*) &Demanda_uni_equil, sizeof(int) );
    
    
    if (Semana == valid_month && use_valid) break;

    months[t_count]= Semana;
    demands[t_count] = Demanda_uni_equil;
    sales[t_count] = Venta_uni_hoy;
    returns[t_count] = Dev_uni_proxima;
    client_ids[t_count] = Cliente_ID;
    {
      auto key = make_tuple(Cliente_ID, Producto_ID, Agencia_ID, (char) Canal_ID);
      auto itr = last_group.find(key);
      if (itr == last_group.end()) {
	last_group[key] = t_count;
      } else {
	next_id[t_count] = itr->second;
	itr->second = t_count;
      }
    }

    {
      auto key = make_tuple(Producto_ID, Agencia_ID);
      auto itr = product_group.find(key);
      if (itr == product_group.end()) {
	product_group[key] = t_count;
      } else {
	next_id_prod[t_count] = itr->second;
	itr->second = t_count;
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

    {
      auto itr = p_popularity.find(Producto_ID);
      if (use_valid || Semana > 3) {
	if (itr == p_popularity.end()) {
	  p_popularity[Producto_ID] = Demanda_uni_equil;
	} else {
	  p_popularity[Producto_ID] += Demanda_uni_equil;
	}
      }
    }

    {
      if (write_ffm) {
	ffm_tr << log(Demanda_uni_equil+1) << "\t";
	write_ffm_data(ffm_tr, Cliente_ID, Producto_ID, Agencia_ID, Canal_ID, Ruta_SAK, feat_count);
      }
    }

    {
      if (write_ffm_s) {
	ffm_tr_s << log(Demanda_uni_equil+1) << "\t";
	write_ffm_data_s(ffm_tr_s, Cliente_ID, Producto_ID, Agencia_ID, Canal_ID, Ruta_SAK, feat_count_s);
      }
    }

    if (t_count%10000==0 || t_count == max_count-1) {
      prt_progress_bar((float) t_count / (float) (max_count-1));
    }
    t_count ++;
  }  
  printf("\n");
  if (write_ffm) ffm_tr.close();
  if (write_ffm_s) ffm_tr_s.close();
  /* load product weights */

  FILE *product_file;
  product_file = fopen("/home/jxy198/kaggle-inventory/cxx/product_weight.csv", "r");
  if (product_file == NULL)
    exit(EXIT_FAILURE);
  for (int i=0; i<num_of_products; ++i) {
    int id, w;
    fscanf(product_file, "%d,%d", &id, &w);
    p_weight[id]=w;
  }
  fclose(product_file);


  FILE *aggregate_file;
  size_t size_of_group;

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
      if (fabs(w[0]) < 5 && w[1] >= 0) {
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


  /* re-scan for validation */
  if (use_valid) {
  const int nfold=5; 
  hash<int> int_hash;
  ofstream fold_file; fold_file.open("folds.txt");

  cout << "File Scan Resume:\n";
  ofstream valid_file; if (!write_ffm && !write_ffm_s) valid_file.open("valid.bin", ios::out | ios::binary);
  ofstream ffm_te; if (write_ffm) ffm_te.open("ffm_te.txt");
  ofstream ffm_te2; if (write_ffm) ffm_te2.open("ffm_te2.txt");
  ofstream ffm_te_s; if (write_ffm_s) ffm_te_s.open("ffm_te.s.txt");
  ofstream ffm_te2_s; if (write_ffm_s) ffm_te2_s.open("ffm_te2.s.txt");

  ifstream ffm_te_pred; if (!write_ffm) ffm_te_pred.open("ffm_te_pred.txt");
  ifstream ffm_te_pred_recent; if (!write_ffm) ffm_te_pred_recent.open("ffm_te_pred.last3.txt");
  ifstream ffm_te_pred_s; if (!write_ffm_s) ffm_te_pred_s.open("ffm_te_pred.s.txt");
  ifstream knn_te_pred; if (read_knn) knn_te_pred.open("knn_te_pred.txt");
  bool first_line_valid=true;
  do {
    if (first_line_valid) {
    first_line_valid=false;
    }
    else{
    train_file_bin.read( (char*) &Semana, sizeof(int) );
    train_file_bin.read( (char*) &Agencia_ID, sizeof(int) );
    train_file_bin.read( (char*) &Canal_ID, sizeof(int) );
    train_file_bin.read( (char*) &Ruta_SAK, sizeof(int) );
    train_file_bin.read( (char*) &Cliente_ID, sizeof(int) );
    train_file_bin.read( (char*) &Producto_ID, sizeof(int) );
    train_file_bin.read( (char*) &Venta_uni_hoy, sizeof(int) );
    train_file_bin.read( (char*) &Venta_hoy, sizeof(float) );
    train_file_bin.read( (char*) &Dev_uni_proxima, sizeof(int) );
    train_file_bin.read( (char*) &Dev_proxima, sizeof(float) );
    train_file_bin.read( (char*) &Demanda_uni_equil, sizeof(int) );
    }
    if (Semana == valid_month && use_valid) {      
      float tmp, tmp2;
      if (!write_ffm && !write_ffm_s && read_knn) {
	// write regular features
	{
	  prepare_features(valid_file, valid_month, Cliente_ID, Producto_ID, Agencia_ID, Canal_ID, Ruta_SAK);
	}

	// write different ffm features
	if (!write_ffm && ffm_te_pred.is_open() && ffm_te_pred_recent.is_open()) {
	  ffm_te_pred >> tmp;
	  valid_file.write((char*) &tmp, sizeof(float));
	  ffm_te_pred_recent >> tmp2; tmp2 -= tmp;
	  valid_file.write((char*) &tmp2, sizeof(float));
	}
	if (!write_ffm_s && ffm_te_pred_s.is_open()) {
	  ffm_te_pred_s >> tmp;
	  valid_file.write((char*) &tmp, sizeof(float));	
	}
	if (read_knn && knn_te_pred.is_open()) {
	  knn_te_pred >> tmp >> tmp2;
	  valid_file.write((char*) &tmp, sizeof(float));
	  valid_file.write((char*) &tmp2, sizeof(float));
	}

	// write labels
	{
	  tmp=Demanda_uni_equil;
	  valid_file.write((char*) &tmp, sizeof(float));
	}
      }

      if (write_ffm) {
	ffm_te << log(Demanda_uni_equil+1) << "\t";
	write_ffm_data(ffm_te, Cliente_ID, Producto_ID, Agencia_ID, Canal_ID, Ruta_SAK, feat_count);
	if (last_group.find(make_tuple(Cliente_ID, Producto_ID, Agencia_ID, (char) Canal_ID))
	    == last_group.end()) {
	  ffm_te2 << log(Demanda_uni_equil+1) << "\t";
	  write_ffm_data(ffm_te2, Cliente_ID, Producto_ID, Agencia_ID, Canal_ID, Ruta_SAK, feat_count);
	}
      }

      if (write_ffm_s) {
	ffm_te_s << log(Demanda_uni_equil+1) << "\t";
	write_ffm_data_s(ffm_te_s, Cliente_ID, Producto_ID, Agencia_ID, Canal_ID, Ruta_SAK, feat_count_s);
	if (last_group.find(make_tuple(Cliente_ID, Producto_ID, Agencia_ID, (char) Canal_ID))
	    == last_group.end()) {
	  ffm_te2_s << log(Demanda_uni_equil+1) << "\t";
	  write_ffm_data_s(ffm_te2_s, Cliente_ID, Producto_ID, Agencia_ID, Canal_ID, Ruta_SAK, feat_count_s);
	}
      }

      fold_file << int_hash(Cliente_ID) % nfold << endl;
    }

    if (t_count%10000==0 || t_count == max_count) {
      prt_progress_bar((float) t_count / (float) max_count);
    }
    t_count ++;
  }
  while (t_count < max_count);
  fold_file.close();
  if (!write_ffm && !write_ffm_s)  valid_file.close();
  if(write_ffm) {  ffm_te.close();   ffm_te2.close(); }
  if(write_ffm_s) {  ffm_te_s.close();   ffm_te2_s.close(); }
  if(!write_ffm && ffm_te_pred_s.is_open() && ffm_te_pred_recent.is_open()) 
    { ffm_te_pred.close(); ffm_te_pred_recent.close();}
  if(!write_ffm_s && ffm_te_pred_s.is_open()) { ffm_te_pred_s.close();}
  if (read_knn) knn_te_pred.close();
  printf("\n");
  }
  train_file_bin.close();

  if (!use_valid) {
  /* write submit files */
  ofstream submit_file;
  submit_file.open("/home/jxy198/kaggle-inventory/cxx/test_feature.bin", ios::out | ios::binary);
  ifstream test_file_bin; test_file_bin.open("/home/jxy198/kaggle-inventory/cxx/test.bin", ios::binary); assert(test_file_bin);
  count = 1; 
  max_count = 6999252;
  cout << "Write Test Submit:\n";
  ofstream ffm_te; if (write_ffm) ffm_te.open("ffm_te.txt");
  ofstream ffm_te_s; if (write_ffm_s) ffm_te_s.open("ffm_te.s.txt");

  ifstream ffm_te_pred; if (!write_ffm) ffm_te_pred.open("ffm_te_pred.txt");
  ifstream ffm_te_pred_recent; if (!write_ffm) ffm_te_pred_recent.open("ffm_te_pred.last3.txt");
  ifstream ffm_te_pred_s; if (!write_ffm_s) ffm_te_pred_s.open("ffm_te_pred.s.txt");
  ifstream knn_te_pred; if (read_knn) knn_te_pred.open("knn_te_pred.txt");
  while (count < max_count) {
    int id,Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID;
    float tmp, tmp2;
    test_file_bin.read((char*) &id, sizeof(int));
    test_file_bin.read((char*) &Semana, sizeof(int));
    test_file_bin.read((char*) &Agencia_ID, sizeof(int));
    test_file_bin.read((char*) &Canal_ID, sizeof(int));
    test_file_bin.read((char*) &Ruta_SAK, sizeof(int));
    test_file_bin.read((char*) &Cliente_ID, sizeof(int));
    test_file_bin.read((char*) &Producto_ID, sizeof(int));

    prepare_features(submit_file, 10, Cliente_ID, Producto_ID, Agencia_ID, Canal_ID, Ruta_SAK);
    if (!write_ffm && ffm_te_pred.is_open() && ffm_te_pred_recent.is_open()) {
      ffm_te_pred >> tmp;
      submit_file.write((char*) &tmp, sizeof(float));
      ffm_te_pred_recent >> tmp2; tmp2 -= tmp;
      submit_file.write((char*) &tmp2, sizeof(float));
    }
    if (!write_ffm_s && ffm_te_pred_s.is_open()) {
      ffm_te_pred_s >> tmp;
      submit_file.write((char*) &tmp, sizeof(float));
    }
    if (read_knn && knn_te_pred.is_open()) {
      knn_te_pred >> tmp >> tmp2;
      submit_file.write((char*) &tmp, sizeof(float));
      submit_file.write((char*) &tmp2, sizeof(float));
    }

    if (write_ffm) {
      ffm_te << 0 << "\t"; // write dummy label
      write_ffm_data(ffm_te, Cliente_ID, Producto_ID, Agencia_ID, Canal_ID, Ruta_SAK, feat_count);
    }    
    if (write_ffm_s) {
      ffm_te_s << 0 << "\t"; // write dummy label
      write_ffm_data(ffm_te_s, Cliente_ID, Producto_ID, Agencia_ID, Canal_ID, Ruta_SAK, feat_count_s);
    }

    if (count % 10000 == 0 || count == max_count-1) {
      prt_progress_bar((float) count / (float) (max_count-1));
    } 
    count++;
  }  
  submit_file.close();
  test_file_bin.close();
  if(write_ffm) {  ffm_te.close(); }
  if(write_ffm_s) {  ffm_te_s.close(); }
  if(!write_ffm && ffm_te_pred.is_open() && ffm_te_pred_recent.is_open()) 
    { ffm_te_pred.close(); ffm_te_pred_recent.close();}
  if(!write_ffm_s && ffm_te_pred_s.is_open()) { ffm_te_pred_s.close();}
  if(read_knn) {knn_te_pred.close();}
  printf("\n");
  }

  free(next_id); 
  free(demands); free(sales); free(returns); free(months); 
  return 0;
}
