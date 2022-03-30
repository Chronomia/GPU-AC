#include <algorithm>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unordered_map>
#include <vector>
//#define RES

#include <fcntl.h> // for O_BINARY
#include <fstream>
#include <limits.h>
#include <sys/times.h>
#include <unistd.h>

using namespace std;

unordered_map<int, int> mapRelation;
unsigned long *bitmap;
int indicator;

struct Record {
    int fileID;
    int fileOffset;
    int ruleStartOffset;
    int no;
    int ruleID;
    int ruleLocation;
    int replaceWord;
    string content;
};
vector<struct Record> recordSets;

int recordCurrent;
string finalOutput;

int globalTmp = 0;
int *splitLocation;
int *rootOffset;
vector<int> fileLength;

vector<string> dictionary_use;
// unordered_map<ulong,string> dictionary_use;
int word1, word2, word3;
set<int> splitSet;
map<string, int> *seqCount;
struct WordCount {
    int word;
    int count;
};
bool sortByCount(const struct WordCount &a, const struct WordCount &b) {
    return a.count > b.count;
}

double timestamp() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

struct RULE {
    unordered_map<int, int> rule_index;
    unordered_map<int, int> word_index;
};
// int* countP;
// vector<list<int> > rule_full;
// list<int>* rule_full;
// map<int, int>* rule_full;
struct RULE *rule_full;
// bool* ruleok;
int *rule_split_indexes;
// int* rules_content;
vector<int> rules_content;
int words, rules;
int total;

// 20170325

int startInRule;
int searchLen;

struct LOC {
    int file;
    int start;
    int end;
};

vector<struct LOC> *rule2location;
// vector<vector<struct LOC> > rule2location;
int currOffset;
vector<int> wordLength;

int correctRule, correctStart;

int goDeep(int start, int end, int ruleU) {
    int rule = ruleU - words;
    //  cout<<"rule="<<rule<<" start:"<<start<<" end:"<<end<<" RULEstart:"<<
    //  rule_split_indexes[rule]<<" RULEend:"<<rule_split_indexes[rule+1]<<endl;
    for (int i = rule_split_indexes[rule]; i < rule_split_indexes[rule + 1]; i++) {
        int tmp = rules_content[i];
        //    cout<<tmp;
        if (tmp >= words) {
            //     cout<<"rule\n";
            tmp = tmp - words;

            bool indicator = true;
            for (vector<struct LOC>::iterator it = rule2location[tmp].begin();
                 it < rule2location[tmp].end(); it++) {
                //      cout<<"("<<it->start<<","<<it->end<<")\n";
                if ((it->start <= start) && ((it->end) >= end)) {
                    //       cout<<"correct: rule: "<<tmp<<" loc:
                    //       "<<it->start<<" "<<it->end<<endl;

                    correctRule = tmp;
                    correctStart = it->start;

                    //      cout<<"start to inner dive\n";
                    goDeep(start, end, tmp + words);
                    return 0;
                } else if (it->start > start) {
                    break;
                }
                if (it->start <= start) {
                    indicator = false;
                }
            }
            if (indicator == true)
                return 0;
        }
    }
}

void scanZwift2(int rule, int newMid, int &currentOffset, int file_index,
                int insert_offset, string &str, int row0End) {
    //        cout<<"scanZwift2"<<endl;

    int no = rule - words;
    int ruleStartOffset = currentOffset;
    for (int i = rule_split_indexes[no]; i < rule_split_indexes[no + 1]; i++) {

        /*
      if(!(bitmap[i>>6]&(1ul<<(i&0x3f)))){
        cout<<"a"<<endl;
      }else if ((bitmap[i>>6]&(1ul<<(i&0x3f)))){
        cout<<"b"<<endl;
      }
      */

        int element_index = rules_content[i];
        //    cout<<(bitmap[i>>6]&(1ul<<(i&0x3f)))<<endl;
        if ((element_index < words) && (!(bitmap[i >> 6] & (1ul << (i & 0x3f))))) {

            // double time11=timestamp();

            //        cout<<"case 1 before"<<endl;

            int newOffset = currentOffset + wordLength[element_index];
            int tmp = insert_offset - newOffset;
            if (tmp > 0) {
                currentOffset = newOffset;
                continue;
            } else {

                // double time13=timestamp();
                // cout<<"case 1"<<endl;
                Record tempRecord;

                bitmap[(i >> 6)] = bitmap[(i >> 6)] | (1ul << ((i)&0x3f));
                tempRecord.no = -1;
                mapRelation[i] = recordCurrent;
                /*
                if( (bitmap[i>>6]&(1ul<<(i&0x3f)) )){
                  tempRecord.no=mapRelation[i];
                }else{
                  bitmap[(i>>6)]=bitmap[(i>>6)]|(1ul<<((i)&0x3f));
                  tempRecord.no=-1;
                }
                mapRelation[i] = recordCurrent;
                */

                // rules_content[i]=recordCurrent;
                recordCurrent++;

                // tempRecord.no=0;

                tempRecord.fileID = file_index;
                tempRecord.fileOffset = insert_offset;
                tempRecord.ruleStartOffset = ruleStartOffset;
                tempRecord.ruleID = no;
                tempRecord.ruleLocation = i;
                tempRecord.replaceWord = element_index;
                tempRecord.content = str;
                int contentSize = str.size();
                //  double time13_5=timestamp();
                for (int j = newMid; j < row0End; j++) {
                    // for(int j=newMid+1; j<row0End; j++){
                    rootOffset[j] += contentSize;
                }
                // double time14=timestamp();

                for (vector<struct Record>::iterator it = recordSets.begin();
                     it != recordSets.end(); it++) {
                    if (file_index == it->fileID) {
                        if (it->fileOffset > insert_offset)
                            it->fileOffset += contentSize;
                        if (it->ruleStartOffset > insert_offset)
                            it->ruleStartOffset += contentSize;
                        // cout<<it->fileID<<" "<<it->fileOffset<<"
                        // "<<it->ruleID<<" "<<it->ruleLocation<<"
                        // "<<it->replaceWord<<endl;
                    }
                }

                recordSets.push_back(tempRecord);
                indicator = 1;
                /*
          double time15=timestamp();
          cout<<"Part 1 Latency(us): "<<(time14-time13)*1000000<<" seconds:
          "<<time14-time13<<endl; cout<<" Part 1_1 Latency(us):
          "<<(time13_5-time13)*1000000<<" seconds: "<<time13_5-time13<<endl;
          cout<<" Part 1_2 Latency(us): "<<(time14-time13_5)*1000000<<" seconds:
          "<<time14-time13_5<<endl; cout<<"Part 2 Latency(us):
          "<<(time15-time14)*1000000<<" seconds: "<<time15-time14<<endl;
          */
                break;
            }

            // double time12=timestamp();
            // cout<<"firstloop innerLatency(us): "<<(time12-time11)*1000000<<"
            // seconds: "<<time12-time11<<endl;

        } else if ((element_index < words) &&
                   (bitmap[i >> 6] & (1ul << (i & 0x3f)))) {
            //  double time11=timestamp();
            //       cout<<"case 2 before"<<endl;

            // int recordIDtemp=element_index - words - rules;
            int recordIDtemp = mapRelation[i];
            int contentSize = 0;

            if ((recordSets[recordIDtemp].fileID == file_index) &&
                (recordSets[recordIDtemp].ruleStartOffset == ruleStartOffset)) {
                contentSize += recordSets[recordIDtemp].content.size();
            }
            // cout<<"new:"<<endl;
            while (recordSets[recordIDtemp].no >= 0) {
                // while(recordSets[recordIDtemp].replaceWord>=(words+rules)){
                recordIDtemp = recordSets[recordIDtemp].no;
                //      cout<<recordIDtemp<<endl;
                // recordIDtemp=recordSets[recordIDtemp].replaceWord-words-rules;
                if ((recordSets[recordIDtemp].fileID == file_index) &&
                    (recordSets[recordIDtemp].ruleStartOffset ==
                     ruleStartOffset)) {
                    contentSize += recordSets[recordIDtemp].content.size();
                }
            }
            //     cout<<"while-finished"<<endl;
            contentSize += wordLength[recordSets[recordIDtemp].replaceWord];

            int newOffset = currentOffset + contentSize;
            int tmp = insert_offset - newOffset;
            if (tmp > 0) {
                currentOffset = newOffset;
                continue;
            } else { // important
                //        cout<<"case 2"<<endl;
                Record tempRecord;
                tempRecord.no = mapRelation[i];
                mapRelation[i] = recordCurrent;

                // rules_content[i]=recordCurrent;
                recordCurrent++;

                // tempRecord.no=recordSets[element_index-words-rules].no+1;
                tempRecord.fileID = file_index;
                tempRecord.fileOffset = insert_offset;
                tempRecord.ruleStartOffset = ruleStartOffset;
                tempRecord.ruleID = no;
                tempRecord.ruleLocation = i;
                tempRecord.replaceWord = element_index;
                tempRecord.content = str;
                int contentSize = str.size();
                for (int j = newMid; j < row0End; j++) {
                    // for(int j=newMid+1; j<row0End; j++){
                    rootOffset[j] += contentSize;
                }
                for (vector<struct Record>::iterator it = recordSets.begin();
                     it != recordSets.end(); it++) {
                    if (file_index == it->fileID) {
                        if (it->fileOffset > insert_offset)
                            it->fileOffset += contentSize;
                        if (it->ruleStartOffset > insert_offset)
                            it->ruleStartOffset += contentSize;
                        // cout<<it->fileID<<" "<<it->fileOffset<<"
                        // "<<it->ruleID<<" "<<it->ruleLocation<<"
                        // "<<it->replaceWord<<endl;
                    }
                }

                recordSets.push_back(tempRecord);
                indicator = 1;
                break;
            }

            //  double time12=timestamp();
            //  cout<<"secondloop innerLatency(us):
            //  "<<(time12-time11)*1000000<<" seconds: "<<time12-time11<<endl;

        } else if ((element_index < words + rules) && (element_index >= words)) {
            scanZwift2(element_index, newMid, currentOffset, file_index, insert_offset,
                       str, row0End);
            if (indicator == 1)
                break;
        }
        /*
        else{
          //how to check if the inserted node is the required node

          int recordIDtemp=element_index - words - rules;
          int contentSize=0;

          if( (recordSets[recordIDtemp].fileID==file_index) &&
        (recordSets[recordIDtemp].ruleStartOffset == ruleStartOffset)){
            contentSize+=recordSets[recordIDtemp].content.size();
          }
          //cout<<"new:"<<endl;
          while(recordSets[recordIDtemp].replaceWord>=(words+rules)){
            // cout<<recordSets[recordIDtemp].replaceWord-words-rules<<endl;
            recordIDtemp=recordSets[recordIDtemp].replaceWord-words-rules;
            if( (recordSets[recordIDtemp].fileID==file_index) &&
        (recordSets[recordIDtemp].ruleStartOffset == ruleStartOffset)){
              contentSize+=recordSets[recordIDtemp].content.size();
            }
          }
          contentSize+=wordLength[recordSets[recordIDtemp].replaceWord];

          int newOffset = currentOffset + contentSize;
          int tmp = insert_offset - newOffset;
          if(tmp>0){
            currentOffset = newOffset;
            continue;
          }
          else{
            rules_content[i]=recordCurrent;
            recordCurrent++;
            Record tempRecord;

            tempRecord.no=recordSets[element_index-words-rules].no+1;
            tempRecord.fileID=file_index;
            tempRecord.fileOffset=insert_offset;
            tempRecord.ruleStartOffset = ruleStartOffset;
            tempRecord.ruleID=no;
            tempRecord.ruleLocation=i;
            tempRecord.replaceWord=element_index ;
            tempRecord.content=str;
            int contentSize=str.size();
            for(int j=newMid+1; j<row0End; j++){
              rootOffset[j]+=contentSize;
            }
            for(vector<struct Record> ::iterator it=recordSets.begin();
        it!=recordSets.end(); it++){ if(it->fileOffset > insert_offset)
                it->fileOffset += contentSize;
              if(it->ruleStartOffset > insert_offset)
                it->ruleStartOffset  += contentSize;
              //cout<<it->fileID<<" "<<it->fileOffset<<" "<<it->ruleID<<"
        "<<it->ruleLocation<<" "<<it->replaceWord<<endl;
            }

            recordSets.push_back(tempRecord);

          }
        }
        */
    }
    // here
}

void dfs(int rule, int splitNo) {
    // cout<<rule<<endl;
    if ((rule < words)) {

        currOffset += wordLength[rule];

        /*
      if(splitNo==3){
        cout<<dictionary_use[rule]<<"("<<currOffset<<")";//zf
      }
           */

        // if((splitSet.find(rule)==splitSet.end() )){
        total++;
        /*
        word1=word2;
        word2=word3;
        word3=rule;
        if(total>2){
          char keyStr[100];
          sprintf(keyStr,"%d|%d|%d",word1,word2,word3);
          //         cout<<keyStr<<endl;
          seqCount[splitNo][string(keyStr)]++;
        }
        */
    } else {
        int no = rule - words;
        for (int i = rule_split_indexes[no]; i < rule_split_indexes[no + 1]; i++) {

            LOC a;
            a.file = splitNo;
            a.start = currOffset;

            dfs(rules_content[i], splitNo);

            if (rules_content[i] >= words) {
                a.end = currOffset;
                rule2location[rules_content[i] - words].push_back(a);
            }
        }
    }
}

string getWordDictionary(ifstream &in) {
    int c;
    /*
        static bool has_special=false;
        static char special;
        if(has_special==true){
            has_special=false;

            return string(&special);
        }
    */
    char t;
    string word;
    t = in.get();
    if (t == ' ' || t == '\t' || t == '\n') {
        // in.get();
        return string(&t, 1);
    }
    word += t;

    while (!in.eof()) {
        c = in.get();

        if (c == '\n') {
            break;
        }
        word += c;
    }

    // in.get();
    //    cout<<"\n************* "<<word<<"*************\n";
    return word;
}

int insert(int file_index, int insert_offset, string &str) {
    indicator = 0;

    // cout<<"\n\nINPUT 3 numbers (fileID, startOffset, length):";
    // cin>>file_index>>insert_offset>>searchLen;
    int searchEnd = insert_offset + searchLen; // search start is in offset(including length !!!)

    finalOutput.clear(); // set string to ""
    // double time4=timestamp();

    /*
    bool find=false;
    for(vector<struct LOC>:: iterator it=rule2location[0].begin();
    it<rule2location[0].end(); it++){ if( it->file == file_index){ if(
    (it->start<=insert_offset) && (it->end>=searchEnd) ){ correctRule=0;
          correctStart=it->start;
          find=true;
          break;
        }
      }
    }
    if(find==false){
      cout<<"WRONG! Range exceeded!\n";
      return -1;
    }
    */

    int row0Start, row0End = splitLocation[file_index];
    if (file_index == 0)
        row0Start = 0;
    else
        row0Start = splitLocation[file_index - 1];

    if ((0 > insert_offset) || (rootOffset[row0End - 1] < searchEnd)) {
        // if( (rootOffset[row0Start]>insert_offset) ||
        // (rootOffset[row0End-1]<searchEnd) ){
        cout << "rootOffset[" << row0Start << "]=" << rootOffset[row0Start]
             << " " << insert_offset << endl;
        cout << "rootOffset[" << row0End - 1 << "]=" << rootOffset[row0End - 1]
             << " " << searchEnd << endl;
        cout << "WRONG! Range exceeded!\n";
        return -1;
    }

    int newHead = row0Start; // file start index
    int newEnd = row0End; // file end index
    int newMid = (newHead + newEnd) / 2; // ???? binary search????
    while ((rootOffset[newMid] > insert_offset) ||
           (rootOffset[newMid + 1] <= insert_offset)) {
        if (rootOffset[newHead] == rootOffset[newMid]) // to find the location, need root offset
            break;
        int oldHead = newHead;
        int oldMid = newMid;
        int oldEnd = newEnd;
        if (insert_offset < rootOffset[newMid]) {
            newEnd = oldMid - 1;
        } else {
            newHead = oldMid;
        }
        newMid = (newHead + newEnd) / 2;
    }
    //  rootOffset[newMid-1] < insert_offset < rootOffset[newMid]
    while (rootOffset[newMid] < insert_offset)
        newMid++;

    int element_index = rules_content[newMid]; // a number that is larger than words
    if ((element_index < words)) { // if root element is word
        // if((element_index < words)||(element_index>=(words+rules))){

        Record tempRecord;

        if (bitmap[newMid >> 6] & (1ul << (newMid & 0x3f))) { // if bitmap set
            tempRecord.no = mapRelation[newMid];
        } else { // if bitmap not set
            bitmap[(newMid >> 6)] =
                bitmap[(newMid >> 6)] | (1ul << ((newMid)&0x3f)); // set bitmap
            tempRecord.no = -1;
        }
        mapRelation[newMid] = recordCurrent; // set root index - record_index

        //   rules_content[newMid]=recordCurrent;
        recordCurrent++;

        /*
           if(element_index < words)
           tempRecord.no=0;
           else{
           tempRecord.no=recordSets[element_index-words-rules].no+1;
           }
           */

        tempRecord.fileID = file_index;
        tempRecord.fileOffset = insert_offset;
        tempRecord.ruleID = 0;
        tempRecord.ruleStartOffset = 0;
        tempRecord.ruleLocation = newMid;
        tempRecord.replaceWord = element_index;
        tempRecord.content = str;
        int contentSize = str.size();

        for (int i = newMid; i < row0End; i++) {
            // for(int i=newMid+1; i<row0End; i++){
            rootOffset[i] += contentSize; // update root offset
        }
        for (vector<struct Record>::iterator it = recordSets.begin();
             it != recordSets.end(); it++) {
            if (file_index == it->fileID) {
                if (it->fileOffset > insert_offset)
                    it->fileOffset += contentSize;
                if (it->ruleStartOffset > insert_offset)
                    it->ruleStartOffset += contentSize; // update past records
                // cout<<it->fileID<<" "<<it->fileOffset<<" "<<it->ruleID<<"
                // "<<it->ruleLocation<<" "<<it->replaceWord<<endl;
            }
        }

        recordSets.push_back(tempRecord); // new set
        indicator = 1;
    } else { // if is rule
        //  double time11=timestamp();
        int currentOffset = rootOffset[newMid - 1];
        scanZwift2(element_index, newMid, currentOffset, file_index, insert_offset, str,
                   row0End);
        // scanZwift2(element_index, newMid, rootOffset[newMid-1], file_index,
        // insert_offset, str, row0End );
        // double time12=timestamp();
        // cout<<"innerLatency(us): "<<(time12-time11)*1000000<<" seconds:
        // "<<time12-time11<<endl;
        // scanZwift(element_index  );
    }
	return true;
}

int main(int argc, char **argv) {

    double time1 = timestamp();

    int splitNum;
    char relationDir[100];
    sprintf(relationDir, "%s/fileYyNO.txt", argv[1]);
    ifstream frelation(relationDir);
    frelation >> splitNum;
    int *split = new int[splitNum];
    memset(split, 0, splitNum);
    splitLocation = new int[splitNum];
    // int *splitLocation = new int[splitNum];
    memset(splitLocation, 0, splitNum);
    int tmp;
    for (int i = 0; i < splitNum; i++) {
        frelation >> tmp >> split[i];
    }
    frelation.close();

    // 20170325

    char dictionaryDir[100];
    // char rowsDir[100];
    char inputDir[100];
    sprintf(dictionaryDir, "%s/dictionary.dic", argv[1]);
    // sprintf(rowsDir,"%s/rows.dic",argv[1]);
    // sprintf(colDir,"%s/rules_content.dic",argv[1]);
    ifstream fin(dictionaryDir);
    // ifstream fin("/home/fengzhang/zf/swift/input/19_NSR/dictionary.dic");
    ulong tem_num;
    string tem_word;
    while (!fin.eof()) {
        tem_num = -1;
        fin >> tem_num;

        fin.get();
        tem_word = getWordDictionary(fin);
        // dictionary_use[tem_num]=tem_word;
        dictionary_use.push_back(tem_word);
        wordLength.push_back(tem_word.length());
        if (tem_word == string(" ") || tem_word == string("\t") ||
            tem_word == string("\n")) {
            splitSet.insert(tem_num);
        }
    }
    fin.close();

    sprintf(inputDir, "%s/rowCol.dic", argv[1]);
    ifstream finput(inputDir);
    finput >> words >> rules;

    cout << "words:" << words << endl;
    cout << "rules:" << rules << endl;

    recordCurrent = 0;
    // recordCurrent=words+rules;

    /*20170325
    set<int> * word2ruleNUM;
    word2ruleNUM= new set<int> [words];
    */

    unordered_map<string, set<int>> word2rule;
    // rule2location.resize(rules+5);
    // rule2location=(vector<struct LOC> *)malloc(rules *sizeof(vector<struct
    // LOC>));
    rule2location = new vector<struct LOC>[rules];

    rule_full = new struct RULE[rules];
    rule_split_indexes = (int *)malloc(sizeof(int) * (rules + 1));
    rule_split_indexes[0] = 0;
    int cur = 0;
    int ruleSize;
    for (int i = 1; i <= rules; i++) {
        finput >> ruleSize;
        rule_split_indexes[i] = ruleSize + rule_split_indexes[i - 1];
        for (int j = 0; j < ruleSize; j++) {
            int tmp;
            finput >> tmp;
            rules_content.push_back(tmp);

            // 20170325
            if (tmp < words) {
                word2rule[dictionary_use[tmp]].insert(i - 1);
                // word2ruleNUM[ tmp].insert(i-1);
            }
        }
    }

    // int* rootOffset = (int*)malloc(sizeof(int)*(rule_split_indexes[1]));
    rootOffset = (int *)malloc(sizeof(int) * (rule_split_indexes[1]));
    memset(rootOffset, -1, sizeof(int) * (rule_split_indexes[1]));
    //  for(int i=0; i<20; i++)
    //   cout<<rootOffset[i]<<endl;
    // exit(0);

    double time2 = timestamp();

    seqCount = new map<string, int>[splitNum];

    // double time2_1=timestamp();
    //////////////////////////////////
    int start = 0, end = 0, splitCur = 0;
    for (int j = rule_split_indexes[0]; j < rule_split_indexes[1]; j++) {
        int tempt = rules_content[j];
        if (tempt == split[splitCur] && splitCur < splitNum) {
            splitLocation[splitCur] = j;
            splitCur++;
        }
    }

    // int curOffset = 0;
    for (int i = 0; i < splitNum; i++) {

        currOffset = 0;

        total = 0;
        if (i != 0)
            start = end;
        end = splitLocation[i];
        for (int j = start; j < end; j++) {
            int word = rules_content[j];

            if (word < words) {

                currOffset += wordLength[word];

                /*
                if(i==3){
                  cout<<dictionary_use[word]<<"("<<currOffset<<")";//zf
                }
                */

                total++;

            } else {

                LOC a;
                a.file = i;
                a.start = currOffset;
                dfs(word, i);
                a.end = currOffset;
                rule2location[word - words].push_back(a);
            }
            rootOffset[j] = currOffset;
        }
        LOC a;
        a.file = i;
        a.start = 0;
        a.end = currOffset;
        rule2location[0].push_back(a);
        // cout<<"file: "<<i<<" length: "<<currOffset<<endl;
        fileLength.push_back(currOffset);
    }

    /*for(int i=0; i<100; i++)
      cout<<rand()<<endl;
      exit(0);
      */

    // zhangfeng zf

    // printf("sizeofulong=%d bytes; colSize=%d\n", sizeof(unsigned long),
    //        rules_content.size());
    int colSize = rules_content.size();
    bitmap =
        (unsigned long *)malloc(sizeof(unsigned long) * (((colSize) >> 6) + 1));
    //  unsigned long *bitmap=(unsigned long*)malloc(sizeof(unsigned
    //  long)*(((colSize)>>6) + 1));
    memset(bitmap, 0, (sizeof(unsigned long) * (((colSize) >> 6) + 1)));

    double time3 = timestamp();

    printf("IO(s): %lf\nComputation(s): %lf\nTotal(s): %lf\n", time2 - time1,
           time3 - time2, time3 - time1);

    //////////////////////////////////

    int file_index = 2;
    int insert_offset = 33;

    string str(64, 'c');

    int totalJ = 3;
    // totalJ=2; splitNum=20000;//for nsfraa
    if (splitNum < 10)
        totalJ = 10000;
    // totalJ=100000;
    // int totalJ=1000;
    cout << "operations: " << totalJ * splitNum << endl;
    double time6 = timestamp();
    // splitNum=20000;
    int temCount = 0;
    for (int i = 0; i < splitNum; i++) {
        //    cout<<"i="<<i<<"VsplitNum="<<splitNum<<endl;
        for (int j = 0; j < totalJ; j++) {
            file_index = i;
            searchLen = 64;
            // searchLen=128;
            if (fileLength[i] < searchLen)
                continue;
            insert_offset = rand() % (fileLength[i] - searchLen);
            // cout<<"file_index: "<<file_index<<" insert_offset:
            // "<<insert_offset<<" searchLen: "<<searchLen<<" fileLen:
            // "<<fileLength[i]<<endl;
            //  double time8=timestamp();
            // insert(3, 1000, str);
            insert(file_index, insert_offset, str);
            //  double time9=timestamp();
            //  cout<<temCount++<<" Latency(us): "<<(time9-time8)*1000000<<endl;
        }
    }

    double time7 = timestamp();
    cout << "operations: " << totalJ * splitNum << endl;
    cout << "AVGLatency(s): " << (time7 - time6) / totalJ / splitNum << endl;
    cout << "AVGLatency(us): " << (time7 - time6) / totalJ / splitNum * 1000000
         << endl;
    cout << "totalTime(s): " << (time7 - time6) << endl;
    cout << "Throughput(op/s): " << (totalJ * splitNum) / (time7 - time6)
         << endl;

    cout << "recoredSets.size=" << recordSets.size() << endl;

    /*
    for(vector<struct Record> ::iterator it=recordSets.begin();
    it!=recordSets.end(); it++){ cout<<it->fileID<<" "<<it->fileOffset<<"
    "<<it->ruleID<<" "<<it->ruleLocation<<" "<<it->replaceWord<<endl;
    }
    */

    return 0;
}
