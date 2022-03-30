# include <iostream>
# include <fstream>
# include <string>
# include <stdio.h>

# include "../../headers/insert_batch/tools.h"

using namespace std;

void check_fin(ifstream &fin, string file_path){
	if(!fin){
		cout << "file " << file_path << " cannot open." << endl;
	}
	return;
}

string get_word(ifstream &fin){
	int c;
    char t;
    string word;

    t = fin.get();
    if (t == ' ' || t == '\t' || t == '\n') {
        // fin.get();
        return string(&t, 1);
    }
    word += t;

    while (!fin.eof()) {
        c = fin.get();

        if (c == '\n') {
            break;
        }
        word += c;
    }

	return word;
}

void aggregate_index(int* array, int* res, int size){
	res[0] = 0;
	for(int i = 1; i <= size; i ++){
		res[i] = array[i - 1] + res[i - 1];
	}

	return;
}

void display_array_content(int* array, int size){
	for(int i = 0; i < size; i ++){
		cout << i << " : " << array[i] << endl;
	}
	// cout << endl;
	return;
}

void check_gpu_mem(){

    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    size_t used = total - avail;
    
	std::cout << "==============="<< std::endl;
    std::cout << "Avail memory : " << avail << std::endl;
    std::cout << "Used  memory : " << used << std::endl;
    std::cout << "Total memory : " << total << std::endl;
    std::cout << "==============="<< std::endl;
	std::cout << std::endl;

}

void aggregate_index_for_fileoffsets(int* array, int* res, int size, int* split_indexes, int split_num){
	res[0] = 0;
	int file_index = 0;
	
	for(int i = 1; i <= size; i ++){
		res[i] = array[i - 1] + res[i - 1];
		// if hit split index
		if(file_index < split_num && i == split_indexes[file_index + 1]){
			// cout << res[i] << endl;

			file_index += 1;
			res[i] = 0;
		}
	}

	return;
}
