echo "Insert In Batch: "
cd insert_batch
make
../../bin/insert_batch ../../input/COV19_seq/
make clean
cd ../

echo ""
echo ""

echo "Insert In Sequence: "
cd insert_sequence
make
../../bin/insert_sequence ../../input/COV19_seq/
make clean
cd ../
