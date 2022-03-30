#pragma once

using namespace std;

struct record {
    int file_index;
    int file_offset;
    int rule_start_offset;
    int no;
    int rule_index;
    int rule_location;
    int replace_word;
    // string content;
	char* content;
	int content_length; // needed !
};

struct insert_update_record{
	int file_index;
	int root_insert_index;
	int insert_length;
	int insert_offset;
};

struct record* create_record_set(int size);

__global__ void insert(int *file_indexes, int* offset, struct insert_update_record* insert_update_records_device, int query_num);

__device__ int insert_into_rule(int rule_index, int insert_index, int& curr_offset, int file_index, int insert_offset, struct insert_update_record* insert_update_records_device, int string_index);

__global__ void insert_update_offsets(struct insert_update_record* insert_update_records_device, int root_size);

__global__ void insert_update_records(struct insert_update_record* insert_update_records_device, int query_size);

__device__ int get_file_index(int root_rule_index);

__host__ void host_insert(int *file_indexes, int* offset, int query_num, int tid);

__host__ int host_insert_into_rule(int rule_index, int insert_index, int& curr_offset, int file_index, int insert_offset, int tid);