#include <stdio.h>
#include <fstream>
#include <string>
#include <assert.h>

int main (int argc, char** argv) {
  using namespace std;
  FILE *read_file;
  ofstream write_file_bin_index, write_file_bin_label; 
  int index[4], tmp;
  float label;

  read_file = fopen((string("ffm_") + string(argv[1]) + string(".txt")).c_str(), "r");
  assert(read_file);
  write_file_bin_index.open(string("eenn_") + string(argv[1]) + string(".index.bin"), 
			    ios::binary | ios::out);
  write_file_bin_label.open(string("eenn_") + string(argv[1]) + string(".label.bin"), 
			    ios::binary | ios::out);
  while (fscanf(read_file, "%f %d:%d:%d %d:%d:%d %d:%d:%d %d:%d:%d", 
		&label, 
		&tmp, index, &tmp, 
		&tmp, index+1, &tmp, 
		&tmp, index+2, &tmp, 
		&tmp, index+3, &tmp) == 13) {
    //printf("%.5f %d %d %d %d\n", label, index[0], index[1], index[2], index[3]);
    write_file_bin_label.write((char*) &label, sizeof(float));
    write_file_bin_index.write((char*) &index, sizeof(index));
  }
  fclose(read_file);
  write_file_bin_index.close();
  write_file_bin_label.close();

  return 0;
}
