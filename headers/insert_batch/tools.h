#pragma once

using namespace std;

void check_fin(ifstream &fin, string file_path);
void check_gpu_mem();
string get_word(ifstream &fin);
void display_array_content(int *array, int size);
void aggregate_index(int *array, int *res, int size);
void aggregate_index_for_fileoffsets(int* array, int* res, int size, int* split_indexes, int split_num);