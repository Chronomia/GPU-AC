## GPU random access on hierarchically compressed data

We currently submit the implementation of the insert operation, which consists of two functions, `insert in batch `and `insert in sequence`. The source code of these two functions locates in the directory `src/insert_batch`and`src/insert_sequence`. 

We offer one dataset named COV19\_seq in the directory `input` for evaluation, which is also available on the internet. You need to decompress the dataset file `cov19.tar.gz` before running the example functions. 

Directory `headers` includes related `.h` files used in random operation functions.

Directory `query` includes randomly generated queries for the two functions. 

Directory `bin` includes binary executable files for the two functions. 

----

We provide bash file `src/run_insert.sh` to make a fast evaluation of the two operation functions. You can call `bash src/run_insert.sh` to see the evaluation result.

