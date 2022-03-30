#include <fstream>
#include <iostream>
#include <string>

#define FILE_MAX 1
#define OFFSET_MAX 100
#define STRING_LENGTH_MAX 50

using namespace ::std;

int main(int argc, char **argv) {
    string filename = argv[1];
    ofstream fout(filename + ".txt");

    srand((unsigned)time(NULL));
    int n = 10000;
    fout << n << endl;
	// char ___;
    for (int i = 0; i < n; i++) {
        fout << rand() % FILE_MAX << " " << rand() % OFFSET_MAX + 1 << " " << string(rand() % STRING_LENGTH_MAX + 1, 'a' + (rand() % 26)) << endl;
		// cout << (int)___ << endl;;
	}
    fout.close();
}
