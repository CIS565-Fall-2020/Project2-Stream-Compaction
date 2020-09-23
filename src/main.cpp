/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 10; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

const int repeatTime = 100;
float *record = new float[repeatTime];

float getTimeAvg(float *src) {
  float t = 0.f;
  for (int i = 0; i < repeatTime; i++) {
    t += src[i];
  }
  return t / repeatTime;
}

void printTime(float *src) {
  std::cout << "    Time record is [";
  std::cout.precision(5);
  for (int i = 0; i < repeatTime; i++) {
    if (repeatTime > 16 && i == 13) {
          std::cout << "... ";
          i = repeatTime - 2;
          continue;
    }
    std::cout << src[i];
    if (i != repeatTime - 1) {
        std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);
    float res[13];


    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    printDesc("cpu scan, power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, b);
      StreamCompaction::CPU::scan(SIZE, b, a);
      record[i] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[0] = getTimeAvg(record);
    printElapsedTime(res[0], "(std::chrono Measured)");
    printArray(SIZE, b, true);

    
    printDesc("cpu scan, non-power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c);
      StreamCompaction::CPU::scan(NPOT, c, a);
      record[i] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[1] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    printDesc("naive scan, power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c); 
      StreamCompaction::Naive::scan(SIZE, c, a);
      record[i] = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[2] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(CUDA Measured)");
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */
    
    printDesc("naive scan, non-power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c);
      StreamCompaction::Naive::scan(NPOT, c, a);
      record[i] = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[3] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(CUDA Measured)");
    printCmpResult(NPOT, b, c);

    printDesc("work-efficient scan, power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c);
      StreamCompaction::Efficient::scan(SIZE, c, a);
      record[i] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[4] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    printDesc("work-efficient scan, non-power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c);
      StreamCompaction::Efficient::scan(NPOT, c, a);
      record[i] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[5] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(CUDA Measured)");
    printCmpResult(NPOT, b, c);

    printDesc("thrust scan, power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c);
      StreamCompaction::Thrust::scan(SIZE, c, a);
      record[i] = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[6] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    printDesc("thrust scan, non-power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c);
      StreamCompaction::Thrust::scan(NPOT, c, a);
      record[i] = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
    }    
    printTime(record);
    res[7] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    printDesc("cpu compact without scan, power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, b);
      count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
      record[i] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[8] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    printDesc("cpu compact without scan, non-power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c);
      count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
      record[i] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[9] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    printDesc("cpu compact with scan");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c);
      count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
      record[i] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[10] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    printDesc("work-efficient compact, power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c);
      count = StreamCompaction::Efficient::compact(SIZE, c, a);
      record[i] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[11] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    printDesc("work-efficient compact, non-power-of-two");
    for (int i = 0; i < repeatTime; i++) {
      zeroArray(SIZE, c);
      count = StreamCompaction::Efficient::compact(NPOT, c, a);
      record[i] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    }
    printTime(record);
    res[12] = getTimeAvg(record);
    printElapsedTime(getTimeAvg(record), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    std::cout << "Result for n = " << SIZE << " is :" << std::endl;
    printTime(res);

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
