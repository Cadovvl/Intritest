#include <iostream>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>

using namespace std;

namespace testData {
  vector<uint64_t> missedNumber(uint64_t n, uint64_t t) {
    vector<uint64_t> res(n - 1);


    for (uint64_t i = 0; i < t; ++i) {
      res[i] = i;
    }

    for (uint64_t i = t + 1; i < n; ++i) {
      res[i - 1] = i;
    }

    std::random_device rng;
    std::mt19937 urng(rng());

    shuffle(begin(res), end(res), urng);
    return res;
  }
}



uint64_t findMissed(const vector<uint64_t>& v) {
  uint64_t res = 0ull;

  // Todo: do in compile time, if we know size
  for (uint64_t i = 0; i < v.size() + 1; ++i) {
    res ^= i;
  }

  for (const auto& i : v) {
    res ^= i;
  }

  return res;
}


uint64_t getInintialConstant(uint64_t n) {
  uint64_t res = 0ull;
  auto lz = __lzcnt64(n);

  for (uint64_t i = 1ull << (63 - lz); i < n; ++i) {
    res ^= i;
  }

  return res;
}

uint64_t findMissedOptimized(const vector<uint64_t>& v) {
  uint64_t res = getInintialConstant(v.size() + 1);


  for (const auto& i : v) {
    res ^= i;
  }

  return res;
}


uint64_t findMissedOptimized2(const vector<uint64_t>& v) {
  auto size = v.size();

  if (size < 100) {
    return findMissed(v);
  }

  uint64_t res = getInintialConstant(size + 1);

  const uint64_t* it = &(*begin(v));
  const uint64_t* e = &(*end(v));


  auto cummul = _mm512_load_si512(it);
  it += 8;

  while (it + 8 < e) {
    auto tmp = _mm512_load_si512(it);
    cummul = _mm512_xor_si512(cummul, tmp);
    it += 8;
  }

  while (it != e) {
    res ^= *it++;
  }

  res ^= cummul.m512i_u64[0];
  res ^= cummul.m512i_u64[1];
  res ^= cummul.m512i_u64[2];
  res ^= cummul.m512i_u64[3];
  res ^= cummul.m512i_u64[4];
  res ^= cummul.m512i_u64[5];
  res ^= cummul.m512i_u64[6];
  res ^= cummul.m512i_u64[7];

  return res;
}


static constexpr uint64_t N = 50000000;


int main_()
{

  std::random_device rd; 
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> distrib(1, N);

  uint64_t d = distrib(gen);

  cout << "Missed number: " << d << endl;

  auto data = testData::missedNumber(N, d);

  cout << "Data generated" << endl;


  auto t1 = chrono::high_resolution_clock::now();
  auto missed = findMissed(data);
  auto t2 = chrono::high_resolution_clock::now();

  cout << "Execution time: " 
    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
    << endl;

  cout << "Missed found: " << missed << endl;


  t1 = chrono::high_resolution_clock::now();
  missed = findMissedOptimized(data);
  t2 = chrono::high_resolution_clock::now();

  cout << "Execution time: "
    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
    << endl;

  cout << "Missed found: " << missed << endl;


  t1 = chrono::high_resolution_clock::now();
  missed = findMissedOptimized2(data);
  t2 = chrono::high_resolution_clock::now();

  cout << "Execution time: "
    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
    << endl;

  cout << "Missed found: " << missed << endl;

  return 0;
}
