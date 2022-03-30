# include <iostream>
# include <stdio.h>
# include <string>
# include <vector>

# include "../../headers/insert_batch/record.h"

# define BLOCK_SIZE 512

extern __constant__ int records_size_device;
extern __constant__ int word_num_device, file_split_num_device;

extern __device__ struct record* records_device;
extern __device__ int* rules_content_device;
extern __device__ int* word_lengths_device;

extern __device__ int* string_start_indexes_device;
extern __device__ char* insert_strings_device;

extern __device__ unsigned int* root_rule_start_offsets_device;
extern __device__ unsigned long long* element_bitmap_device;
extern __device__ int* file_split_indexes_device;
extern __device__ int* rule_split_indexes_device;
extern __device__ int curr_record_num_device;
extern __device__ int* relation_map_device;

struct record* create_record_set(int size){
	int malloc_size = sizeof(struct record) * size;
	// cout << "malloc size is : " << malloc_size << endl;
	struct record* temp_records_device;
	cudaError_t stat;

	stat = cudaMalloc(&temp_records_device, malloc_size);
	cudaMemset(temp_records_device, 0x00, malloc_size);

	if(stat){
		cout << endl;
		cout << "cudamalloc records failed with stat : " << stat << endl;
		return NULL;
	}
	// stat = cudaMemcpyToSymbol(records_device, &temp_records_device, sizeof(temp_records_device));
	cudaMemcpyToSymbol(records_size_device, &size, sizeof(int));
	
	return temp_records_device;
}

__global__ void insert(int* file_indexes, int* insert_offsets, struct insert_update_record* insert_update_records_device, int query_num){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid >= query_num){
		return;
	}
	int file_index = file_indexes[tid];
	int insert_offset = insert_offsets[tid];

	int file_start_index = file_split_indexes_device[file_index];
	int file_end_index   = file_split_indexes_device[file_index + 1];
	// printf("%d %d\n", file_index, file_split_indexes_device[1]);

	if((insert_offsets < 0) || (root_rule_start_offsets_device[file_end_index - 1] <= insert_offset)){
		// printf("%d %d\n", file_end_index - 1, root_rule_start_offsets_device[file_end_index - 1]);
		printf("insert invalid. \n");
	}

	int search_start = file_start_index;
	int search_end   = file_end_index;
	int search_mid   = (search_start + search_end) / 2;

	while(root_rule_start_offsets_device[search_mid] > insert_offset || 
		  root_rule_start_offsets_device[search_mid + 1] <= insert_offset){
		if(search_start == search_mid){
			break;
		}

		// int last_search_start = search_start;
		// int last_search_end   = search_end;
		int last_search_mid   = search_mid;

		if(insert_offset < root_rule_start_offsets_device[search_mid]){
			search_end = last_search_mid - 1;
		}	
		else{
			search_start = last_search_mid;
		}
		search_mid = (search_start + search_end) / 2;
	}

	int root_rule_index =  search_mid; // the index in root rule to insert
	int element_index = rules_content_device[search_mid];

	if(element_index < word_num_device){
		// if element is word
		struct record temp_record;
		if(element_bitmap_device[root_rule_index >> 6] & (1ul << (root_rule_index & 0x3f))){
			temp_record.no = relation_map_device[root_rule_index];
		}
		else{
			element_bitmap_device[root_rule_index >> 6] = 
			element_bitmap_device[root_rule_index >> 6] & (1ul << (root_rule_index & 0x3f));
			temp_record.no = -1;
		}

		temp_record.file_index = file_index;
		temp_record.file_offset = insert_offset;
		temp_record.rule_index = 0;
		temp_record.rule_start_offset = 0;
		temp_record.rule_location = root_rule_index;
		temp_record.replace_word = element_index; // original word
		temp_record.content = &insert_strings_device[string_start_indexes_device[tid]];
		int insert_string_length = string_start_indexes_device[tid] - string_start_indexes_device[tid + 1];
		temp_record.content_length = insert_string_length;

		int record_index = atomicAdd(&curr_record_num_device, 1);

		records_device[record_index] = temp_record;
		relation_map_device[root_rule_index] = record_index;	

		struct insert_update_record temp_update_record;
		temp_update_record.file_index = file_index;
		temp_update_record.insert_offset = insert_offset;
		temp_update_record.root_insert_index = root_rule_index;
		temp_update_record.insert_length = insert_string_length;
		insert_update_records_device[record_index] = temp_update_record;

		// no update 
	}
	else{
		// if element is rule
		int curr_offset = root_rule_start_offsets_device[search_mid];
		insert_into_rule(element_index - word_num_device, search_mid, curr_offset, file_index, insert_offset, insert_update_records_device, tid);
	}

	return;
}


__device__ int insert_into_rule(int rule_index, int insert_index, int& curr_offset, int file_index, int insert_offset, struct insert_update_record* insert_update_records_device, int tid){ // search mid is insert index(root rule index)
	// int file_end_index = file_split_indexes_device[file_index + 1];
	int rule_start_offset = curr_offset; // curr offset is the start offset of rule to insert

	int rule_start_index = rule_split_indexes_device[rule_index];
	int rule_end_index = rule_split_indexes_device[rule_index + 1];

	for(int i = rule_start_index; i < rule_end_index; i ++){
		int element_index = rules_content_device[i];
		if(element_index < word_num_device && !(element_bitmap_device[i >> 6] & (1ul << (i & 0x3f)))){
			// if is word and bit map not set
			int new_offset = curr_offset + word_lengths_device[element_index];
			if(insert_offset > new_offset){
				// keep searching
				curr_offset = new_offset;
				continue;
			}
			else{ // end searching
				struct record temp_record;
				element_bitmap_device[(i >> 6)] = element_bitmap_device[(i >> 6)] | (1ul << (i & 0x3f));
				temp_record.no = -1;

				// get record_index
				int insert_record_index = atomicAdd(&curr_record_num_device, 1);
				relation_map_device[i] = insert_record_index;

				temp_record.file_index = file_index;
				temp_record.file_offset = insert_offset;
				temp_record.rule_start_offset = rule_start_offset;
				temp_record.rule_index = rule_index;
				temp_record.rule_location = i;
				temp_record.replace_word = element_index;
				temp_record.content = &insert_strings_device[string_start_indexes_device[tid]];
				int insert_string_length = string_start_indexes_device[tid + 1] - string_start_indexes_device[tid];
				temp_record.content_length = insert_string_length;

				records_device[insert_record_index] = temp_record;

				struct insert_update_record temp_update_record;
				temp_update_record.file_index = file_index;
				temp_update_record.insert_offset = insert_offset;
				temp_update_record.root_insert_index = insert_index;
				temp_update_record.insert_length = insert_string_length;
				insert_update_records_device[insert_record_index] = temp_update_record;

				// break;
				return 1; // inserted, for end
			}
			
		}
		else if(element_index < word_num_device && element_bitmap_device[i >> 6] && (1ul << (i & 0x3f))){ // if is word but bitmap set
			int record_index = relation_map_device[i]; // the last record index of this rule
			int content_size = 0;

			struct record temp_record = records_device[record_index];
			if(temp_record.file_index == file_index &&
				temp_record.rule_start_offset == rule_start_offset){ // if in the same location
				content_size += temp_record.content_length;
			}	

			while(temp_record.no >= 0){ // view back every record in the same location
				record_index = temp_record.no;
				// set new temp record
				temp_record = records_device[record_index];
				if(temp_record.file_index == file_index && 
					temp_record.rule_start_offset == rule_start_offset){
					content_size += temp_record.content_length;
				}

			}

			content_size += word_lengths_device[records_device[record_index].replace_word];
		
			int new_offset = curr_offset + content_size;
			if(insert_offset > new_offset){ // keep searching
				curr_offset = new_offset;
				continue;
			}
			else{ // end searching
				struct record insert_record;
				int insert_record_index = atomicAdd(&curr_record_num_device, 1);

				insert_record.no = relation_map_device[i];
				relation_map_device[i] = insert_record_index;

				insert_record.file_index = file_index;
				insert_record.file_offset = insert_offset;
				insert_record.rule_start_offset = rule_start_offset;
				insert_record.rule_index = rule_index;
				insert_record.rule_location = i;
				insert_record.replace_word = element_index;
				insert_record.content = &insert_strings_device[string_start_indexes_device[tid]];
				int insert_string_length = string_start_indexes_device[tid + 1] - string_start_indexes_device[tid];
				insert_record.content_length = insert_string_length;

				records_device[insert_record_index] = insert_record;

				struct insert_update_record temp_update_record;
				temp_update_record.file_index = file_index;
				temp_update_record.insert_offset = insert_offset;
				temp_update_record.root_insert_index = insert_index;
				temp_update_record.insert_length = insert_string_length;
				insert_update_records_device[record_index] = temp_update_record;

				// to update root offset and record offset
				// for(int j = insert_index + 1; j < file_end_index; j ++){
				// 	// root_rule_start_offsets_device[j] += insert_string_length;
				// 	atomicAdd(&root_rule_start_offsets_device[j], insert_string_length);
				// }
				// for(int j = 0; j < insert_record_index; j ++){
				// 	struct record search_record = records_device[j];
				// 	if(file_index == search_record.file_index){
				// 		// if record file index hits, update file offset
				// 		if(search_record.file_offset > insert_offset){
				// 			// records_device[j].file_offset += insert_string_length;
				// 			atomicAdd(&records_device[j].file_offset, insert_string_length);
				// 		}
				// 		if(search_record.rule_start_offset > insert_offset){
				// 			// records.device[j].rule_start_offset += insert_string_length;
				// 			atomicAdd(&records_device[j].rule_start_offset, insert_string_length);
				// 		}
				// 	}
				// }

				return 1; // record inserted
			}
			
		}
		else if(element_index >= word_num_device){
			// if is still rule
			int if_inserted = insert_into_rule(element_index - word_num_device, insert_index, curr_offset, file_index, insert_offset, insert_update_records_device, tid);
			if(if_inserted){
				return 1;
			}
		}
	}
	
	return 0; // if not inserted
}

__device__ int get_file_index(int root_rule_index){
	int file_index = 0;
	
	for(int i = 0; i < file_split_num_device; i ++){
		if(root_rule_index < file_split_indexes_device[i + 1]){
			break;
		}
		file_index += 1;
	}

	return file_index;
}

__global__ void update_root_start_offsets(int range, int record_index, int tid_file_index, struct insert_update_record* insert_update_records_device){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid >= range){
		return;
	}

	struct insert_update_record temp_update_record = insert_update_records_device[tid];
	int file_index = temp_update_record.file_index;
	int root_insert_index = temp_update_record.root_insert_index;
	int insert_length = temp_update_record.insert_length;

	if(root_insert_index < record_index && file_index == tid_file_index){
		// root_rule_start_offsets_device[tid] += offset_update;
		atomicAdd(&root_rule_start_offsets_device[record_index], insert_length);
	}

}

__global__ void insert_update_offsets(struct insert_update_record* insert_update_records_device, int root_size){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid >= root_size){
		return;
	}

	int tid_file_index = get_file_index(tid);
	int offset_update = 0;

	// int block_size = BLOCK_SIZE;
	// int grid_size  = (records_size_device + block_size - 1) / block_size;

	// update_root_start_offsets<<<grid_size, block_size>>>(records_size_device, tid, tid_file_index, insert_update_records_device);

	// to process the root_rule_start_offset[tid]
	for(int i = 0; i < records_size_device; i ++){
		struct insert_update_record temp_update_record = insert_update_records_device[i];
		int file_index = temp_update_record.file_index;
		int root_insert_index = temp_update_record.root_insert_index;
		int insert_length = temp_update_record.insert_length;

		if(root_insert_index < tid && file_index == tid_file_index){
			offset_update += insert_length;
		}
	}

	root_rule_start_offsets_device[tid] += offset_update;

	return;

}

__global__ void insert_update_records(struct insert_update_record* insert_update_records_device, int query_size){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid >= query_size){
		return;
	}

	struct record this_record = records_device[tid];
	struct insert_update_record this_update_record = insert_update_records_device[tid];

	// to process the records[tid]

	for(int i = 0; i < records_size_device; i ++){
		struct insert_update_record temp_update_record = insert_update_records_device[i];

		if(temp_update_record.file_index == this_update_record.file_index){

			if(temp_update_record.insert_offset < this_update_record.insert_offset){
				this_record.file_offset += temp_update_record.insert_length;
			}
			if(temp_update_record.insert_offset < root_rule_start_offsets_device[this_update_record.root_insert_index]){
				this_record.rule_start_offset += temp_update_record.insert_length;
			}
			
		}
	}

	records_device[tid] = this_record;

	return;
}