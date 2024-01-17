#include <iostream>
#include <unistd.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <sys/statvfs.h>
#include <chrono>
#include <cstdint>
#include <stdio.h>

#include "analysis.h"

#define MIN_BYTES (1 << 14)
#define MAX_BYTES (1 << 27)
#define MAX_STRIDE 10   // stride 8 bytes
#define MAX_ELEMENTS MAX_BYTES / sizeof(int64_t)

int64_t data[MAX_ELEMENTS];  // The global array we'll be traversing

size_t get_cpu_cores();
std::string get_cpu_model_name();
double get_cpu_frequency();
size_t get_dram_size();
int64_t get_storage_size(const std::string& mount_point);
int64_t get_storage_cache();
void init_data(int64_t *data, size_t n);
int64_t sum_array(const int32_t n, const int32_t stride);
double measure_cache_bandwidth(int32_t size, int32_t stride, double cpu_mhz);
void write_file(const std::string& filename, long long fileSize);
void read_file(const std::string& filename);
double measure_storage_bandwidth(int64_t storage_cache, double cpu_mhz);

int main() {
  size_t n_cores = get_cpu_cores();
  size_t l1_isize = sysconf(_SC_LEVEL1_ICACHE_SIZE) / pow(2, 10);
  size_t l1_dsize = sysconf(_SC_LEVEL1_DCACHE_SIZE) / pow(2, 10);
  size_t l2_size = sysconf(_SC_LEVEL2_CACHE_SIZE) / pow(2, 10);
  size_t l3_size = sysconf(_SC_LEVEL3_CACHE_SIZE) / pow(2, 10);
  double dram_size = get_dram_size() / pow(2, 30);
  double storage_size = 0;
  int64_t l5_size = get_storage_size("/");
  if (l5_size != -1) {
    storage_size = l5_size / pow(2, 30);
  }
  else {
    std::cerr << "Can not find the storage device size\n";
  }
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "==========================================================================\n";
  std::cout << "                       Memory Hierarchy size                              \n";
  std::cout << "==========================================================================\n";
  std::cout << "Model name of CPU: " << get_cpu_model_name() << "\n";
  std::cout << "CPU cores: " << n_cores << "\n";
  std::cout << "Level 1 (L1) cache: \n";
  std::cout << "\tTotal instruction cache(i-cache) size: " << n_cores * l1_isize << " KB - " << l1_isize << " KB per instance\n";
  std::cout << "\tTotal data cache(d-cache) size: " << n_cores * l1_dsize << " KB - " << l1_dsize << " KB per instance\n";
  std::cout << "Level 2 (L2) cache size: " << n_cores * l2_size << " KB - " << l2_size << "KB per instance\n";
  std::cout << "Level 3 (L3) cache size: " << l3_size << " KB\n";
  std::cout << "DRAM size: " << dram_size << " GB\n";
  std::cout << "Storage device size: " << storage_size << " GB\n";
  std::cout << "==========================================================================\n";
  
  // Measure the bandwidth of L1/L2/L3 caches, DRAM and storage with varying access strides and sizes
  double cpu_freq = get_cpu_frequency();
  std::cout << "Clock frequency is approximate: " << cpu_freq << "\n";
  std::cout << "Memory mountain - L1, L2, L3, and RAM bandwidth (MB/s):\n";
  std::cout << "\t";
  int32_t size;
  int32_t stride;
  for (stride = 1; stride <= MAX_STRIDE; stride++)
    std::cout << "s" << stride << "\t";
  std::cout << "\n";

  for (size = MAX_BYTES; size >= MIN_BYTES; size >>= 1) {
    if (size >= (1 << 20))
      std::cout << size / (1 << 20) << " MB\t";
    else
      std::cout << size / 1024 << " KB\t";
    for (stride = 1; stride <= MAX_STRIDE; stride++) {
      std::cout << std::setw(6) << measure_cache_bandwidth(size, stride, cpu_freq) << "\t";
    }
    std::cout << "\n";
  }
  int64_t storage_cache = get_storage_cache();
  std::cout << "==========================================================================\n";
  std::cout << "The cache of storage in RAM is approximate " << storage_cache << " KB\n";
  std::cout << "Storage device bandwidth (MB/s): " << measure_storage_bandwidth(storage_cache, cpu_freq) << "\n";

  return 0;
}

size_t get_cpu_cores() {
  // Available only on Linux: Get total cpu core(s) of CPU by read the "/proc/cpuinfo" file
  std::ifstream file;
  size_t n_cores = 0;
  file.exceptions (std::ifstream::failbit | std::ifstream::badbit);
  
  try {
    std::string line;
    file.open("/proc/cpuinfo");
    while (std::getline(file, line)) {
      if (line.substr(0, 7) == "core id") {
        n_cores = std::max(n_cores, (uint64_t)std::stoi(line.substr(line.find(':') + 2)));
      }
    }
    file.close();
  }
  catch (std::ifstream::failure& e) {
     std::cerr << "Exception opening/reading/closing /proc/cpuinfo file\n";
  }

  return n_cores + 1;
}

std::string get_cpu_model_name() {
  // Available only on Linux: Read model name of CPU by read the "/proc/cpuinfo" file
  std::ifstream cpuinfo_file;
  std::string cpu_name = "";
  cpuinfo_file.exceptions (std::ifstream::failbit | std::ifstream::badbit);
  
  try {
    std::string line;
    cpuinfo_file.open("/proc/cpuinfo");
    while (std::getline(cpuinfo_file, line)) {
      if (line.substr(0, 10) == "model name") {
          cpu_name = line.substr(13);
          break;
      }
    }
    cpuinfo_file.close();
  }
  catch (std::ifstream::failure& e) {
    std::cerr << "Exception opening/reading/closing /proc/cpuinfo file\n";
  }

  return cpu_name;
}

double get_cpu_frequency() {
  std::ifstream file("/proc/cpuinfo");
  std::string line;
  double cpu_freq = -1.0;

  while (std::getline(file, line)) {
      if (line.substr(0, 7) == "cpu MHz") {
          cpu_freq = (double)std::stod(line.substr(line.find(':') + 2));
          break;
      }
  }

  return cpu_freq;  // Error
}

size_t get_dram_size() {
  // Available only on Linux: Get DRAM size by read the "/proc/meminfo" file
  std::ifstream file;
  size_t dram_size = 0;
  file.exceptions (std::ifstream::failbit | std::ifstream::badbit);
  
  try {
    std::string line;
    file.open("/proc/meminfo");
    while (std::getline(file, line)) {
      if (line.substr(0, 8) == "MemTotal") {
        std::istringstream iss(line);
        std::string value;
        iss >> value >> value;  // Skip "MemTotal:"
        dram_size = std::stoll(value) * 1024;  // B
        break;
      }
    }
    file.close();
  }
  catch (std::ifstream::failure& e) {
    std::cerr << "Exception opening/reading/closing: /proc/meminfo file\n";
  }

  return dram_size;
}

int64_t get_storage_size(const std::string& mount_point) {
    struct statvfs fsinfo;
    if (statvfs(mount_point.c_str(), &fsinfo) == 0) {
        return fsinfo.f_blocks * fsinfo.f_frsize;  // Total blocks * block size
    }
    return -1;  // Error
}

int64_t get_storage_cache() {
  std::ifstream file;
  int64_t storage_cache = 0;
  file.exceptions (std::ifstream::failbit | std::ifstream::badbit);
  
  try {
    std::string line;
    file.open("/proc/meminfo");
    while (std::getline(file, line)) {
      if (line.substr(0, 5) == "Shmem" || line.substr(0, 12) == "SReclaimable") {
        std::istringstream iss(line);
        std::string value;
        iss >> value >> value;  
        storage_cache += std::stoll(value);  // B
        break;
      }
    }
    file.close();
  }
  catch (std::ifstream::failure& e) {
    std::cerr << "Exception opening/reading/closing: /proc/meminfo file\n";
  }

  return storage_cache;
}

int64_t sum_array(const int32_t n, const int32_t stride) {
  int32_t i;
  int64_t sum = 0;
  volatile int64_t sink;
  for (i = 0; i < n; i += stride) {
    sum += data[i];
  }
  sink = sum;  // prevent the compiler optimize away the loop

  return sink;
}

double measure_cache_bandwidth(int32_t size, int32_t stride, double cpu_mhz) {
  int32_t n = size / sizeof(int64_t);
  Analysis eq_test(3, 500, 0.01, 0);
  double cycles = eq_test.get_execution_cpu_cycles(sum_array, n, stride);
  // return cycles;
  return (size / stride) / (cycles / cpu_mhz);
}

void write_file(const std::string& filename, long long fileSize) {
  std::ofstream file(filename, std::ofstream::binary);

  if (!file.is_open()) {
    std::cerr << "Error creating file: " << filename << "\n";
    return;
  }

  // Fill the file with data 
  char* buffer = new char[fileSize];
  std::fill_n(buffer, fileSize, 'a');  // Fill with any desired character

  file.write(buffer, fileSize);

  delete[] buffer;
  file.close();
}

void read_file(const std::string& filename) {  
  std::ifstream file;
  file.exceptions (std::ifstream::failbit | std::ifstream::badbit);
  
  try {
    file.open(filename, std::ofstream::binary);
    // get length of file:
    file.seekg (0, file.end);
    int64_t length = file.tellg();
    file.seekg (0, file.beg);

    char* buffer = new char[length];

    file.read (buffer,length);

    delete[] buffer;
    file.close();
  }
  catch (std::ifstream::failure& e) {
    std::cerr << "Exception opening/reading/closing: " << filename << " file\n";
  }
}

double measure_storage_bandwidth(int64_t storage_cache, double cpu_mhz) {
  // Test with file size = storage + 10 MB
  int64_t file_size = storage_cache * 1024 + 10 * (1 << 20);
  std::string file_name = "temporary_file.txt";
  write_file(file_name, file_size); 
  Analysis eq_test(3, 500, 0.01, 0);
  double cycles = eq_test.get_execution_cpu_cycles(read_file, file_name);
  // remove temporary file
  if (remove(file_name.c_str()))
    std::cerr << "\nError Occurred!";
  // return cycles;
  return file_size / (cycles / cpu_mhz);
}
