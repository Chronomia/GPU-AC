# include <iostream>
# include <stdio.h>
# include <string>
# include <vector>
# include <fstream>
# include <cmath>
# include <algorithm>
# include <unordered_map>
# include <set>
# include <thrust/device_vector.h>

# include "../../headers/insert_sequence/tools.h"
# include "../../headers/insert_sequence/record.h"

# define NEW_STACK_SIZE 5000
# define BUILD_BLOCK_SIZE 512
# define QUERY_BLOCK_SIZE 512
// # define DEBUG

using namespace::std;

unsigned long *bitmap;

// cpu data structures
int rules_content_size;
vector<int> word_lengths;
vector<int> rules_content;

// query
int* query_file_indexes;
int* query_insert_offsets;
string insert_strings;
vector<struct insert_query>* insert_querys;

// insert records
__device__ char* insert_strings_device;
__device__ int* string_start_indexes_device;
__device__ struct record* records_device;
__device__ unsigned long long* element_bitmap_device;
__device__ int curr_record_num_device;
__device__ int* relation_map_device;
__device__ int* insert_file_split_indexes_device;

// gpu data structures
__constant__ int word_num_device, rule_num_device, file_split_num_device;
__constant__ int hashtable_size_device;
__constant__ int records_size_device;

__device__ int* rules_content_device;
__device__ int* word_lengths_device;
__device__ int* rule_lengths_device;

__device__ int* rule_split_indexes_device;
__device__ int* file_split_indexes_device;

__device__ unsigned int* root_rule_start_offsets_device;


// functions
extern __global__ void insert(int *file_indexes, int offsets, int query_num);
__global__ void get_offsets(int root_rule_size, int* root_rule_offsets_device);

int main(int argc, char** argv){
	int file_split_num;
	
	// --- IO --- //

	clock_t time1 = clock();
	string input_file_path = argv[1];

	// process fileYyNO.txt

	string fileYyNo_path = input_file_path + "fileYyNO.txt";
	ifstream fin_filesplit(fileYyNo_path);
	check_fin(fin_filesplit, fileYyNo_path);
	
	fin_filesplit >> file_split_num;
	cudaMemcpyToSymbol(file_split_num_device, &file_split_num, sizeof(int));

	int file_split_word[file_split_num];
	int temp = 0;
	for(int i = 0; i < file_split_num; i ++){	
		fin_filesplit >> temp >> file_split_word[i];
	}

	fin_filesplit.close();


	// process rowCol.dic

	string rowCol_path = input_file_path + "rowCol.dic";
	ifstream fin_rules(rowCol_path);
	check_fin(fin_rules, rowCol_path);

	int word_num, rule_num = 0;
	fin_rules >> word_num >> rule_num;
	cudaMemcpyToSymbol(word_num_device, &word_num, sizeof(int));
	cudaMemcpyToSymbol(rule_num_device, &rule_num, sizeof(int));
	cout << "word number : " << word_num << ", rule number : " << rule_num << ", file number : "<< file_split_num << endl;

	int rule_size, element;
	int* rule_split_indexes = (int* )malloc(sizeof(int) * (rule_num + 1));
	rule_split_indexes[0] = 0;

	for(int i = 1; i <= rule_num; i ++){
		fin_rules >> rule_size;
		rule_split_indexes[i] = rule_size + rule_split_indexes[i - 1];

		for(int j = 0; j < rule_size; j ++){
			fin_rules >> element;
			rules_content.push_back(element);
		}
	}

	rules_content_size = rules_content.size();
	fin_rules.close();


	// process dictionary.dic

	string dictionary_path = input_file_path + "dictionary.dic";
	ifstream fin_worddic(dictionary_path);
	check_fin(fin_worddic, dictionary_path);

	int word_index;
	string word;
	string *word_collection = new string[word_num];

	while(!fin_worddic.eof()){
		fin_worddic >> word_index;
		fin_worddic.get();
		// fin_worddic >> word; // not working 
		word = get_word(fin_worddic); // super nb
	
		word_collection[word_index] = word;
		word_lengths.push_back(word.length());
	}

	fin_worddic.close();

	// get file split index

	int split_curr = 0;
	int* file_split_indexes = (int* )malloc(sizeof(int) * (file_split_num + 1));
	file_split_indexes[0] = 0;

	for(int i = rule_split_indexes[0]; i < rule_split_indexes[1]; i ++){
		if(rules_content[i] == file_split_word[split_curr] && split_curr < file_split_num){
			file_split_indexes[split_curr + 1] = i;
			split_curr ++;
		}
	}
	int root_rule_size = file_split_indexes[file_split_num];

	clock_t time2;
	time2 = clock();
	cout << endl;
	cout << "IO time : " << (double)(time2 - time1) / CLOCKS_PER_SEC << "s" << endl;
	cout << "===============" << endl;

	// --- insert query --- //

	int query_size;
	
	string query_path = "../../query/insert_1_query.txt";
	ifstream fin_query(query_path);
	check_fin(fin_query, query_path);

	fin_query >> query_size;
	// cout << "total query size : " << query_size << endl;

	int query_malloc_size = query_size * sizeof(int);

	query_file_indexes = (int *)malloc(query_malloc_size);
	query_insert_offsets = (int* )malloc(query_malloc_size);
	int* insert_string_lengths = (int* )malloc(query_malloc_size);

	insert_querys = new vector<struct insert_query> [file_split_num]; // in files

	int file_index, offset;
	string temp_string;
	int insert_string_total_length;

	insert_strings = "";

	for(int i = 0; i < query_size; i ++){
		fin_query >> file_index >> offset >> temp_string;

		// update struct
		struct insert_query temp_insert_query;
		temp_insert_query.file_index = file_index;
		temp_insert_query.insert_offset = offset;
		temp_insert_query.string_index = i;

		insert_querys[file_index].push_back(temp_insert_query);


		query_file_indexes[i] = file_index;
		query_insert_offsets[i] = offset;
		insert_strings += temp_string;
		
		int string_length = temp_string.length();
		insert_string_lengths[i] = string_length;
		insert_string_total_length += string_length;

		// cout << file_index << " " << temp_string << endl;
	}
	fin_query.close();
	//printf("haha\n");
	// --- process insert queries --- //
	int* insert_file_split_indexes = (int* )malloc(sizeof(int) * (file_split_num + 1));
	insert_file_split_indexes[0] = 0;

	// put queries into sequence by file index
	int* new_query_file_indexes = (int* )malloc(query_malloc_size);
	int* new_query_insert_offsets = (int* )malloc(query_malloc_size);
	int* new_string_indexes = (int* )malloc(query_malloc_size);

	int insert_index = 0;
	for(int i = 0; i < file_split_num; i ++){
		insert_file_split_indexes[i + 1] = insert_file_split_indexes[i] + insert_querys[i].size(); 
		
		for(auto it : insert_querys[i]){
			new_query_file_indexes[insert_index] = it.file_index;
			new_query_insert_offsets[insert_index] = it.insert_offset;
			new_string_indexes[insert_index] = it.string_index;
			insert_index += 1;
		}
	}


	// --- malloc for insert --- //

	// rules content in device
	int* temp_rules_malloc;
	cudaMalloc(&temp_rules_malloc, sizeof(int) * rules_content_size);
	cudaMemcpy(temp_rules_malloc, &rules_content[0], sizeof(int) * rules_content_size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(rules_content_device, &temp_rules_malloc, sizeof(temp_rules_malloc));

	// file and rule split indexes (start from 0)
	int* temp_rule_split_indexes_device;
	int* temp_file_split_indexes_device;

	cudaMalloc(&temp_rule_split_indexes_device, sizeof(int) * (rule_num + 1));
	cudaMalloc(&temp_file_split_indexes_device, sizeof(int) * (file_split_num + 1));

	cudaMemcpy(temp_rule_split_indexes_device, rule_split_indexes, sizeof(int) * (rule_num + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(temp_file_split_indexes_device, file_split_indexes, sizeof(int) * (file_split_num + 1), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(rule_split_indexes_device, &temp_rule_split_indexes_device, sizeof(temp_rule_split_indexes_device));
	cudaMemcpyToSymbol(file_split_indexes_device, &temp_file_split_indexes_device, sizeof(temp_file_split_indexes_device));

	// load file indexes to device
	int *query_file_indexes_device;
	cudaMalloc(&query_file_indexes_device, query_malloc_size);
	cudaMemcpy(query_file_indexes_device, new_query_file_indexes, query_malloc_size, cudaMemcpyHostToDevice);
	// load insert offsets to device
	int *query_insert_offsets_device;
	cudaMalloc(&query_insert_offsets_device, query_malloc_size);
	cudaMemcpy(query_insert_offsets_device, new_query_insert_offsets, query_malloc_size, cudaMemcpyHostToDevice);
	// load string indexes to device
	int *insert_string_indexes_device;
	cudaMalloc(&insert_string_indexes_device, query_malloc_size);
	cudaMemcpy(insert_string_indexes_device, new_string_indexes, query_malloc_size, cudaMemcpyHostToDevice);
	// insert file split indexes
	int *temp_insert_file_split_indexes;
	cudaMalloc(&temp_insert_file_split_indexes, sizeof(int) * (file_split_num + 1));
	cudaMemcpy(temp_insert_file_split_indexes, insert_file_split_indexes, sizeof(int) * (file_split_num + 1), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(insert_file_split_indexes_device, &temp_insert_file_split_indexes, sizeof(temp_insert_file_split_indexes));

	char* temp_insert_strings_device;
	cudaMalloc(&temp_insert_strings_device, sizeof(char) * insert_string_total_length);
	cudaMemcpy(temp_insert_strings_device, &insert_strings[0], sizeof(char) * insert_string_total_length, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(insert_strings_device, &temp_insert_strings_device, sizeof(temp_insert_strings_device));

	int* string_split_indexes = (int* )malloc(sizeof(int) * (query_size + 1));
	aggregate_index(insert_string_lengths, string_split_indexes, query_size);
	
	int* temp_string_start_indexes;
	cudaMalloc(&temp_string_start_indexes, sizeof(int) * (query_size + 1));
	cudaMemset(temp_string_start_indexes, 0x00, sizeof(int) * (query_size + 1));
	cudaMemcpyToSymbol(string_start_indexes_device, &temp_string_start_indexes, sizeof(temp_string_start_indexes));

	// records on device
	struct record* temp_records = create_record_set(query_size);
	cudaMemcpyToSymbol(records_device, &temp_records, sizeof(temp_records));

	// bitmap on device
	// cout << "size of unsigned long long : " << sizeof(unsigned long long) << endl;
	unsigned long long* temp_element_bitmap;
	int bitmap_size = rule_split_indexes[rule_num];
	cudaMalloc(&temp_element_bitmap, sizeof(unsigned long long) * ((bitmap_size >> 6) + 1));
	cudaMemset(temp_element_bitmap, 0x00, sizeof(unsigned long long) * ((bitmap_size >> 6) + 1));
	cudaMemcpyToSymbol(element_bitmap_device, &temp_element_bitmap, sizeof(temp_element_bitmap));

	int* temp_word_lengths_device;
	cudaMalloc(&temp_word_lengths_device, sizeof(int) * word_num);
	cudaMemcpy(temp_word_lengths_device, &word_lengths[0], sizeof(int) * word_num, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(word_lengths_device, &temp_word_lengths_device, sizeof(temp_word_lengths_device));

	int* temp_rule_lengths_device;
	cudaMalloc(&temp_rule_lengths_device, sizeof(int) * rule_num);
	cudaMemset(temp_rule_lengths_device, 0x00, sizeof(int) * rule_num);
	cudaMemcpyToSymbol(rule_lengths_device, &temp_rule_lengths_device, sizeof(temp_rule_lengths_device));

	// root rule offsets:
	int* temp_root_rule_offsets;
	cudaMalloc(&temp_root_rule_offsets, sizeof(int) * root_rule_size);
	// cudaMemcpyToSymbol(root_rule_offsets_device, &temp_root_rule_offsets, sizeof(temp_root_rule_offsets)); 

	// --- run dfs --- //

	cudaError_t stat;

	// check_gpu_mem();
	stat = cudaDeviceSetLimit(cudaLimitStackSize, NEW_STACK_SIZE);
	cout << "if stack successfully allocated : " << (stat == 0) << endl;

	int build_block_size = BUILD_BLOCK_SIZE;
	int build_grid_size = (root_rule_size + build_block_size - 1) / build_block_size;


	get_offsets<<<build_grid_size, build_block_size>>>(root_rule_size, temp_root_rule_offsets);
	cudaDeviceSynchronize();

	// # ifdef DEBUG
	// printf("after get offset : %s\n",cudaGetErrorString(cudaGetLastError()));
	// # endif

	clock_t time3 = clock();
	cout << endl;
	cout << "GPU building time : " << (double)(time3 - time2) / CLOCKS_PER_SEC << "s" << endl;
	cout << "===============" << endl;

	// copy back offsets and aggregrate
	int* root_rule_offsets_host = (int* )malloc(sizeof(int) * root_rule_size);
	stat = cudaMemcpy(root_rule_offsets_host, temp_root_rule_offsets, sizeof(int) * root_rule_size, cudaMemcpyDeviceToHost);
	
	int* root_rule_start_offsets = (int* )malloc(sizeof(int) * (root_rule_size + 1));
	// aggregate_index(root_rule_offsets_host, root_rule_start_offsets, root_rule_size);
	aggregate_index_for_fileoffsets(root_rule_offsets_host, root_rule_start_offsets, root_rule_size, file_split_indexes, file_split_num);

	// set device int* 
	unsigned int* temp_root_rule_start_offsets;
	cudaMalloc(&temp_root_rule_start_offsets, sizeof(unsigned int) * (root_rule_size + 1));
	cudaMemcpy(temp_root_rule_start_offsets, root_rule_start_offsets, sizeof(unsigned int) * (root_rule_size + 1), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(root_rule_start_offsets_device, &temp_root_rule_start_offsets, sizeof(temp_root_rule_start_offsets));

	// set curr record number
	int temp_curr_record_num = 0;
	cudaMemcpyToSymbol(curr_record_num_device, &temp_curr_record_num, sizeof(int));

	// relation_map
	int* relation_map;
	cudaMalloc(&relation_map, sizeof(int) * bitmap_size);
	cudaMemset(relation_map, 0x00, sizeof(int) * bitmap_size);
	cudaMemcpyToSymbol(relation_map_device, &relation_map, sizeof(relation_map));

	cudaFree(temp_root_rule_offsets);
	free(root_rule_offsets_host);
	free(root_rule_start_offsets);

	// --- run insert --- //

	int insert_block_size = QUERY_BLOCK_SIZE;
	int insert_grid_size  = (file_split_num + insert_block_size - 1) / insert_block_size; // split by file

	# ifdef DEBUG
	printf("before insert : %s\n",cudaGetErrorString(cudaGetLastError()));
	# endif

	clock_t time4 = clock();
	insert<<<insert_grid_size, insert_block_size>>>(query_file_indexes_device, query_insert_offsets_device, insert_string_indexes_device);
	cudaDeviceSynchronize();

	# ifdef DEBUG
	printf("after insert : %s\n",cudaGetErrorString(cudaGetLastError()));
	# endif

	clock_t time5 = clock();
	cout << endl;
	cout << "INSERT time(s): " << (double)(time5 - time4) / CLOCKS_PER_SEC << endl;
	cout << "AVGLatency(s): " << (double)(time5 - time4) / CLOCKS_PER_SEC / query_size << endl;
    cout << "AVGLatency(us): " << (double)(time5 - time4) / CLOCKS_PER_SEC / query_size * 1000000
         << endl;
    cout << "Throughput(op/s): " << query_size * CLOCKS_PER_SEC / (double)(time5 - time4)
         << endl;
	cout << "===============" << endl;
	


	return 0;
}


__device__ int dfs_for_offset(int element){
	if(element < word_num_device){
		return word_lengths_device[element];
	}
	else{
		int rule_index = element - word_num_device;
		int rule_start_index = rule_split_indexes_device[rule_index];
		int rule_end_index = rule_split_indexes_device[rule_index + 1];

		int temp_curr_offset = 0;
		for(int i = rule_start_index; i < rule_end_index; i ++){
			int temp_element = rules_content_device[i];
			int subtree_offset = dfs_for_offset(temp_element);

			temp_curr_offset += subtree_offset;
		}
		return temp_curr_offset;
	}

	return 0;
}

__global__ void get_offsets(int root_rule_size, int* root_rule_offsets_device){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid >= root_rule_size){
		return;
	}

	int element = rules_content_device[tid];

	int element_offset = dfs_for_offset(element);
	root_rule_offsets_device[tid] = element_offset;
}
