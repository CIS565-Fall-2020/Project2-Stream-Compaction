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
#include <stream_compaction/radixSort.h>
#include "testing_helpers.hpp"
#include "csvfile.hpp"
#include <cassert>



const int power = 16;
const int SIZE = 1 << power; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];



int main(int argc, char* argv[]) {
    // Scan tests

    int sort_size_power = 5;
    assert(sort_size_power <= power);
    int sort_size = 1 << sort_size_power;

    int num_power = 5;

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    //onesArray(SIZE - 1, a);
    genArray(SIZE - 1, a, 2);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    std::ostringstream stringStream;
    stringStream << "Compact_" << power;
    stringStream << "_Sort_" << sort_size_power << "_" << num_power;
    stringStream << "_naiveblck_" << blocksize << "_effblck_" << efficient_blocksize;
    stringStream << ".csv";
    std::string file_name = stringStream.str();

    csvfile my_csv(file_name);
    my_csv << "Compact at power " << "Sort power" << "Sort num power" << endrow;
    my_csv << power << sort_size_power << num_power << endrow;
    my_csv << " " << endrow;

    my_csv << "Naive block size" << "Efficient block size" << endrow;
    my_csv << blocksize << efficient_blocksize << endrow;
    my_csv << endrow;

    my_csv << endrow;
    my_csv << "SCAN" << endrow;
    my_csv << endrow;

    float cur_time;
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    cur_time = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(std::chrono Measured)");
    printArray(SIZE, b, true);
    my_csv << "cpu scan p_2 " << cur_time << endrow;
    //
    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    cur_time = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);
    my_csv << "cpu scan n_p_2 " << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    cur_time = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    my_csv << "naive scan p_2" << cur_time << endrow;


    //// For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    ///*onesArray(SIZE, c);
    //printDesc("1s array for finding bugs");
    //StreamCompaction::Naive::scan(SIZE, c, a);
    //printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    cur_time = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);
    my_csv << "naive scan n_p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a, EFF_method::nonOptimization, true);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    my_csv << "efficient scan non-optimization p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a, EFF_method::nonOptimization, true);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
    my_csv << "efficient scan non-optimization n_p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient scan with shared memory, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a, EFF_method::sharedMemory, true);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    my_csv << "efficient scan shared p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient scan with shared memory, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a, EFF_method::sharedMemory, true);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
    my_csv << "efficient scan shared n_p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient scan with index scale, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a, EFF_method::idxMapping, true);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    my_csv << "efficient scan idx p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient scan with index scale, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a, EFF_method::idxMapping, true);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
    my_csv << "efficient scan idx n_p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    cur_time = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    my_csv << "thrust scan p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    cur_time = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
    my_csv << "thrust scan n_p_2" << cur_time << endrow;

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM SORT TESTS **\n");
    printf("*****************************\n");

    my_csv << endrow;
    my_csv << "SORT" << endrow;
    my_csv << endrow;

    
    genArray(sort_size - 1, a, 1 << num_power);  // Leave a 0 at the end to test that edge case
    a[sort_size - 1] = 0;
    printArray(sort_size, a, true);

    printf("The array to be sorted is : \n");
    printArray(sort_size, a, true);
    printDesc("Std sort");
    StreamCompaction::RadixSort::CpuStandardSort(sort_size, b, a);
    cur_time = StreamCompaction::RadixSort::timer().getCpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(std::chrono Measured)");
    printArray(sort_size, b, true);
    my_csv << "std sort" << cur_time << endrow;

    printDesc("Radix sort");
    zeroArray(sort_size, c);
    StreamCompaction::RadixSort::GpuRadixSort(sort_size, c, a, num_power);
    cur_time = StreamCompaction::RadixSort::timer().getCpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(std::chrono Measured)");
    printArray(sort_size, c, true);
    printCmpResult(sort_size, b, c);
    my_csv << "Radix sort" << cur_time << endrow;

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    my_csv << endrow;
    my_csv << "CAMPACTION" << endrow;
    my_csv << endrow;
    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    cur_time = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);
    my_csv << "cpu campact no scan p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    cur_time = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
    my_csv << "cpu campact no scan n_p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    cur_time = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);
    my_csv << "cpu campact scan n_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a, EFF_method::nonOptimization);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);
    my_csv << "eff compact non-opt p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a, EFF_method::nonOptimization);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
    my_csv << "eff compact non-opt n_p_2" << cur_time << endrow;
    
    zeroArray(SIZE, c);
    printDesc("work-efficient compact with idx mapping, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a, EFF_method::idxMapping);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);
    my_csv << "eff compact idx map p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient compact with idx mapping, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a, EFF_method::idxMapping);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
    my_csv << "eff compact idx map n_p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient compact with shared memory, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a, EFF_method::sharedMemory);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);
    my_csv << "eff compact shared p_2" << cur_time << endrow;

    zeroArray(SIZE, c);
    printDesc("work-efficient compact with shared memory, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a, EFF_method::sharedMemory);
    cur_time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    printElapsedTime(cur_time, "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
    my_csv << "eff compact shared n_p_2" << cur_time << endrow;


    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
    // save to csv
    try
    {
        //csvfile csv("MyTable.csv"); // throws exceptions!
        //// Hearer
        //csv << "X" << "VALUE" << endrow;
        //// Data
        //int i = 1;
        /*csv << i++ << "String value" << endrow;
        csv << i++ << 123 << endrow;
        csv << i++ << 1.f << endrow;
        csv << i++ << 1.2 << endrow;
        csv << i++ << "One more string" << endrow;*/
        /*csv << i++ << "\"Escaped\"" << endrow;
        csv << i++ << "=HYPERLINK(\"https://playkey.net\"; \"Playkey Service\")" << endrow;*/
    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
}
