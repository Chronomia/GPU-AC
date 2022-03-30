#pragma once

# define UPDATE_BLOCK_SIZE 512

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

struct record* create_record_set(int size);

__global__ void insert(int *file_indexes, int* offset, int* string_indexes);

__device__ void insert_per_query(int file_index, int offset, int string_index);

__device__ int insert_into_rule(int rule_index, int insert_index, int& curr_offset, int file_index, int insert_offset, int tid);

__global__ void update_root_start_offsets(int start_index, int add, int range);