make: 
	nvcc -c record.cu -rdc=true 
	nvcc -c tools.cu -rdc=true
	nvcc -c insert1.cu -rdc=true
	nvcc record.o tools.o insert1.cu -rdc=true -o ../../bin/insert_sequence
# ./insert ~/input/COV19_seq/

clean: 
	rm -f *.o