#include <iostream>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <map>
#include <iomanip>
#include <limits>
#include <memory>

#include <snappy.h>

constexpr bool PRINT_STATS = false;
constexpr bool PRINT_COMPRESSION_STATS = false;
constexpr bool PRINT_PATTERNS_FREQ = false;

template <typename Dist>
std::vector<uint32_t> generateTestData(size_t n, Dist d) {
  std::vector<uint32_t> res;
  res.reserve(n);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::map<uint32_t, size_t> freq{};

  uint32_t min = std::numeric_limits<uint32_t>::max();
  uint32_t max = std::numeric_limits<uint32_t>::min();
  uint32_t aboves = 0;

  for (int i = 0; i < n; ++i) {
    res.push_back((uint32_t)std::abs(d(gen)));

    if constexpr (PRINT_STATS) {
      ++freq[res.back()];
      if (res.back() > 65'536) {
        aboves++;
      }

      min = std::min(min, res.back());
      max = std::max(max, res.back());
    }
  }

  if constexpr (PRINT_STATS) {

    std::cout << "Generated data: " << std::endl;


    std::cout << "Total: " << n << std::endl;
    std::cout << "Values above 65'536: " << aboves << std::endl;
    std::cout << "[min, max]: [" << min << ", " << max << "]" << std::endl;

    std::cout << "Most frequent values: " << std::endl;

    std::vector<std::pair<uint32_t, size_t>> mf(std::min(10ull, n));

    std::partial_sort_copy(freq.begin(), freq.end(),
      mf.begin(), mf.end(),
      [](const auto& l, const auto& r) {
        return l.second > r.second || (l.second == r.second && l.first < r.first);
      });

    for (const auto& p : mf) {
      std::cout << p.first << ":\t" << p.second << " \ttimes" << std::endl;
    }

  }

  return res;
}


void testEqual(const std::vector<uint32_t>& l, const std::vector<uint32_t>& r) {
  if (l.size() != r.size()) {
    std::cout << "Different size" << std::endl;
    return;
  }

  auto il = std::begin(l);
  auto ir = std::begin(r);
  auto el = std::end(l);

  for (; il != el; ++il, ++ir) {
    if (*il != *ir) {
      std::cout << "Different values: " << *il << " != " << *ir << std::endl;
      return;
    }
  }

  std::cout << " * | Everything is correct | * " << std::endl;
}

namespace {

  constexpr uint64_t lz_to_code[33] = { 
    0x3,
    0x3,
    0x3,
    0x3,
    0x3,
    0x3,
    0x3,
    0x3,
    0x2,
    0x2,
    0x2,
    0x2,
    0x2,
    0x2,
    0x2,
    0x2,
    0x2,
    0x2,
    0x2,
    0x2,
    0x2,
    0x1,
    0x1,
    0x1,
    0x1,
    0x1,
    0x1,
    0x1,
    0x0,
    0x0,
    0x0,
    0x0,
    0x0,
  };

  constexpr uint8_t lz_to_length[33] = {
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    4,
    4,
    4,
    4,
    4,
  };

  constexpr uint8_t mask_to_length[4] = {
    4, 11, 24, 32
  };
}


namespace {

  const auto AND_MASK = _mm512_setr_epi32(
    //       ||      ||      ||      ||
    0b11000011000011000011000000000000,
    0b11000011000011000011000000000000,
    0b11000011000011000000000000000000,
    0b11000011000000000001100000000000,
    //       ||      ||      ||      ||
    0b11000000000001100001100000000000,
    0b11000000000001100000000000110000,
    0b11000000000001100001100000000000,
    0b11000011000000000001100000000000, 
    //       ||      ||      ||      ||
    0b11000011000011000000000000000000, 
    0b11000000000001100000000000000000,
    0b11000011000000000000000000000000, 
    0b11000011000000000000000000000000,
    //       ||      ||      ||      ||
    0b11000000000001100000000000000000,
    0b11000000000000000000000000000000, // single values
    0b11000000000000000000000000000000,
    0b11000000000000000000000000000000
  );


  const auto CMP_MASK = _mm512_setr_epi32(
    //       ||      ||      ||      ||
    0b00000000000000000000000000000000,
    0b00000000000000000001000000000000,
    0b00000000000001000000000000000000,
    0b00000001000000000000000000000000,
    //       ||      ||      ||      ||
    0b01000000000000000000000000000000,
    0b01000000000000100000000000000000,
    0b01000000000000000000100000000000,
    0b00000001000000000000100000000000, // 8
    //       ||      ||      ||      ||
    0b00000000000000000000000000000000,
    0b01000000000000100000000000000000,
    0b00000000000000000000000000000000, // replication
    0b00000001000000000000000000000000,
    //       ||      ||      ||      ||
    0b01000000000000000000000000000000,
    0b00000000000000000000000000000000, // single values
    0b01000000000000000000000000000000,
    0b10000000000000000000000000000000
  );


  const __m512i DIGIT_MASKS[16] = {
    _mm512_setr_epi32(
      0b00111100000000000000000000000000,
      0b00000000111100000000000000000000,
      0b00000000000000111100000000000000,
      0b00000000000000000000111100000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111100000000000000000000000000,
      0b00000000111100000000000000000000,
      0b00000000000000111100000000000000,
      0b00000000000000000000111111111110,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111100000000000000000000000000,
      0b00000000111100000000000000000000,
      0b00000000000000111111111110000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111100000000000000000000000000,
      0b00000000111111111110000000000000,
      0b00000000000000000000011110000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    //
    _mm512_setr_epi32(
      0b00111111111110000000000000000000,
      0b00000000000000011110000000000000,
      0b00000000000000000000011110000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111111111110000000000000000000,
      0b00000000000000011111111111000000,
      0b00000000000000000000000000001111,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111111111110000000000000000000,
      0b00000000000000011110000000000000,
      0b00000000000000000000011111111111,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111100000000000000000000000000,
      0b00000000111111111110000000000000,
      0b00000000000000000000011111111111,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    // 
    _mm512_setr_epi32(
      0b00111100000000000000000000000000,
      0b00000000111100000000000000000000,
      0b00000000000000111100000000000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111111111110000000000000000000,
      0b00000000000000011111111111000000,
      0b00000000000000000000000000000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111100000000000000000000000000,
      0b00000000111100000000000000000000,
      0b00000000000000000000000000000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111100000000000000000000000000,
      0b00000000111111111110000000000000,
      0b00000000000000000000000000000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    //
    _mm512_setr_epi32(
      0b00111111111110000000000000000000,
      0b00000000000000011110000000000000,
      0b00000000000000000000000000000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111100000000000000000000000000,
      0b00000000000000000000000000000000,
      0b00000000000000000000000000000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111111111110000000000000000000,
      0b00000000000000000000000000000000,
      0b00000000000000000000000000000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      0b00111111111111111111111111000000,
      0b00000000000000000000000000000000,
      0b00000000000000000000000000000000,
      0b00000000000000000000000000000000,
      0,0,0,0,0,0,0,0,0,0,0,0),
  };


  const __m512i SHIFT_MASKS[16] = {
    _mm512_setr_epi32(
      26,
      20,
      14,
      8,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      26,
      20,
      14,
      1,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      26,
      20,
      7,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      26,
      13,
      7,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    //
    _mm512_setr_epi32(
      19,
      13,
      7,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      19,
      6,
      0,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      19,
      13,
      0,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      26,
      13,
      0,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    // 
    _mm512_setr_epi32(
      26,
      20,
      14,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      19,
      6,
      0,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      26,
      20,
      0,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      26,
      13,
      0,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    //
    _mm512_setr_epi32(
      19,
      13,
      0,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      26,
      0,
      0,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      19,
      0,
      0,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
    _mm512_setr_epi32(
      6,
      0,
      0,
      0,
      0,0,0,0,0,0,0,0,0,0,0,0),
  };
  const uint16_t DIGITS_DECOMPRESSED[16] = {
    4,4,3,3,
    3,3,3,3,
    3,2,2,2,
    2,1,1,1
  };
  const uint16_t BITS_SPENT[16] = {
    24, 31, 25, 25,
    25, 32, 32, 32,
    18, 26, 12, 19, 
    19, 6, 13, 26
  };
}

class Compressed {
  uint64_t _bits;
  uint64_t _digits;
  std::vector<uint32_t> _data;

  void flush(
    uint64_t& buf,
    uint8_t& bbits) {
    if (bbits >= 32) {
      _data.push_back((uint32_t)(buf >> 32));
      buf <<= 32;
      bbits -= 32;
      _bits += 32;
    }
  }

public:
  explicit Compressed(const std::vector<uint32_t>& v) :
    _bits(0),
    _digits(v.size()),
    _data{}
  {
    uint64_t buf = 0;
    uint8_t bbits = 0;

    std::map<uint64_t, size_t> freq{};
    std::map<uint8_t, size_t> t4_freq{};
    std::map<uint8_t, size_t> t3_freq{};
    std::map<uint8_t, size_t> t2_freq{};

    uint8_t t4 = 0;
    uint8_t t3 = 0;
    uint8_t t2 = 0;
    

    for (uint32_t num : v) {
      auto lz = __lzcnt(num);
      
      /**
      * Append 2-bit mask
      */

      if constexpr (PRINT_PATTERNS_FREQ) {
        t4 <<= 2;
        t4 |= lz_to_code[lz];
        ++t4_freq[t4];

        t3 <<= 2;
        t3 |= lz_to_code[lz];
        t3 &= 077;
        ++t3_freq[t3];

        t2 <<= 2;
        t2 |= lz_to_code[lz];
        t2 &= 017;
        ++t2_freq[t2];
      }

      if constexpr (PRINT_COMPRESSION_STATS) {
        ++freq[lz_to_code[lz]];
      }

      // 2 bits (64-2) shifted by allocated space
      buf |= lz_to_code[lz] << (62 - bbits);
      bbits += 2;
      flush(buf, bbits);

      /**
      * Append digit
      */

      // Cast to 64-bit for shift
      buf |= ((uint64_t)num) << (64 - lz_to_length[lz] - bbits);
      bbits += lz_to_length[lz];
      flush(buf, bbits);
    }

    if (bbits > 0) {
      _data.push_back((uint32_t)(buf >> 32));
      _bits += bbits;
    }

    if constexpr (PRINT_COMPRESSION_STATS) {
      for (const auto& p : freq) {
        std::cout << p.first << ": " << p.second << " times" << std::endl;
      }
    }


    if constexpr (PRINT_PATTERNS_FREQ) {


      std::vector<std::pair<uint8_t, size_t>> mf(10);

      std::partial_sort_copy(t4_freq.begin(), t4_freq.end(),
        mf.begin(), mf.end(),
        [](const auto& l, const auto& r) {
          return l.second > r.second || (l.second == r.second && l.first < r.first);
        });

      std::cout << "T4" << std::endl;
      for (const auto& p : mf) {
        std::cout << (int)p.first << ": " << p.second << " times" << std::endl;
      }

      std::partial_sort_copy(t3_freq.begin(), t3_freq.end(),
        mf.begin(), mf.end(),
        [](const auto& l, const auto& r) {
          return l.second > r.second || (l.second == r.second && l.first < r.first);
        });

      std::cout << "T3" << std::endl;
      for (const auto& p : mf) {
        std::cout << (int)p.first << ": " << p.second << " times" << std::endl;
      }

      std::partial_sort_copy(t2_freq.begin(), t2_freq.end(),
        mf.begin(), mf.end(),
        [](const auto& l, const auto& r) {
          return l.second > r.second || (l.second == r.second && l.first < r.first);
        });

      std::cout << "T2" << std::endl;
      for (const auto& p : mf) {
        std::cout << (int)p.first << ": " << p.second << " times" << std::endl;
      }
    }
  }

  std::vector<uint32_t> decompress() const {
    std::vector<uint32_t> res;
    res.reserve(_digits);


    auto it = std::begin(_data);

    // todo: try to use 128 bit buffer and use one 'if' per circle
    uint64_t buf = 0;
    uint8_t bbits = 0;



    for (size_t i = 0; i < _digits; ++i) {
      if (bbits < 2) {
        buf |= ((uint64_t)*it) << (32 - bbits);
        bbits += 32;
        ++it;
      }

      auto len = mask_to_length[buf >> 62];
      buf <<= 2;
      bbits -= 2;

      if (bbits < len) {
        buf |= ((uint64_t)*it) << (32 - bbits);
        bbits += 32;
        ++it;
      }

      res.push_back(buf >> (64 - len));
      buf <<= len;
      bbits -= len;
    }

    return res;
  }

  std::vector<uint32_t> decompress_optimized() {
    std::vector<uint32_t> res;
    // NB resize here, cause we will store mm512 in allocated memory 
    // todo: how to do it without memset(0) ?
    res.resize(_digits);

    auto it = std::begin(_data);
    auto rit = std::begin(res);

    size_t i = 0;

    uint64_t buf = 0;
    uint8_t bbits = 0;

    // decode optimized
    // left 16 digits to garantee no memory access violation 
    for (; i + 16 < _digits;) {
      if (bbits < 32) {
        buf |= ((uint64_t)*it) << (32 - bbits);
        bbits += 32;
        ++it;
      }

      int sb = buf >> 32;

      auto buf_copy = _mm512_setr_epi32(
        sb, sb, sb, sb,
        sb, sb, sb, sb,
        sb, sb, sb, sb,
        sb, sb, sb, sb);

      auto masked = _mm512_and_epi32(buf_copy, AND_MASK);
      auto compared = _mm512_cmpeq_epu32_mask(masked, CMP_MASK); 

      // if zero - we have 32 bit digit next
      if (compared) [[likely]]
      {
        auto tz = _tzcnt_u32(compared); // number of pattern

        auto digits = _mm512_and_epi32(buf_copy, DIGIT_MASKS[tz]);
        auto shifted = _mm512_srav_epi32(digits, SHIFT_MASKS[tz]);

        _mm512_store_epi32(&(*rit), shifted);
        std::advance(rit, DIGITS_DECOMPRESSED[tz]);
        i += DIGITS_DECOMPRESSED[tz];

        buf <<= BITS_SPENT[tz];
        bbits -= BITS_SPENT[tz];
      }
      else {
        buf <<= 2;
        bbits -= 2;
        if (bbits < 32) {
          buf |= ((uint64_t)*it) << (32 - bbits);
          bbits += 32;
          ++it;
        }

        *rit = buf >> 32;
        ++rit;
        buf <<= 32;
        bbits -= 32;
        ++i;
      }
    }

    // decode rest
    for (; i < _digits; ++i) {
      if (bbits < 2) {
        buf |= ((uint64_t)*it) << (32 - bbits);
        bbits += 32;
        ++it;
      }

      auto len = mask_to_length[buf >> 62];
      buf <<= 2;
      bbits -= 2;

      if (bbits < len) {
        buf |= ((uint64_t)*it) << (32 - bbits);
        bbits += 32;
        ++it;
      }

      *rit = buf >> (64 - len);
      ++rit;
      buf <<= len;
      bbits -= len;
    }

    return res;
  }

  uint64_t bits() const {
    return _bits;
  }

};



int main()
{
  auto td1 = generateTestData(10'000'000,  
    std::cauchy_distribution<>{4.0, 2.0});
  auto td2 = generateTestData(10'000'000,
    std::cauchy_distribution<>{8.0, 8.0});
  auto td3 = generateTestData(10'000'000,
    std::cauchy_distribution<>{8.0, 16.0});
  auto td4 = generateTestData(10'000'000,
    std::cauchy_distribution<>{16.0, 16.0});
  auto td5 = generateTestData(10'000'000,
    std::cauchy_distribution<>{1000.0, 1000.0});

  std::cout << " _______ " << std::endl;
  std::cout << std::endl;
  std::cout << " ** Compressing ** " << std::endl;
  std::cout << " _______ " << std::endl;
  std::cout << std::endl;


  for (const auto* const td : 
    std::vector<const decltype(td1)*>{ 
            &td1, &td2, &td3, &td4, &td5 
    }) {

    auto t1 = std::chrono::high_resolution_clock::now();
    Compressed cd(*td);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ct = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "* Original:\t";
    std::cout << "[\t" 
      << td1.size() * 32 << " bit,\t"
      << td1.size() * 4 << " b,\t"
      << td1.size() / 256 << " Kb]"
      << std::endl;


    std::cout << "* Compressed:\t";
    std::cout << "[\t"
      << cd.bits() << " bit,\t"
      << cd.bits() / 8 << " b,\t"
      << cd.bits() / 8'192 << " Kb] "
      << std::endl;

    /*

    snappy compression

    */
    std::string snappy_compressed;


    t1 = std::chrono::high_resolution_clock::now();
    auto csize = snappy::Compress(reinterpret_cast<const char*>(&(*td->begin())), td->size() * 4, &snappy_compressed);
    t2 = std::chrono::high_resolution_clock::now();

    auto sct = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "* Snappy:\t";
    std::cout << "[\t"
      << csize * 8 << " bit,\t"
      << csize << " b,\t"
      << csize / 1024 << " Kb]"
      << std::endl;

    std::cout << std::endl;

    std::cout << "       Compression time:\t\t" << ct << std::endl;
    std::cout << "Snappy compression time:\t\t" << sct << std::endl;
    std::cout << std::endl;



    t1 = std::chrono::high_resolution_clock::now();
    auto dd = cd.decompress();
    t2 = std::chrono::high_resolution_clock::now();

    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();



    t1 = std::chrono::high_resolution_clock::now();
    auto ddo = cd.decompress_optimized();
    t2 = std::chrono::high_resolution_clock::now();

    auto odt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();


    std::vector<uint32_t> sd(td->size());
    std::string uncompressed;

    t1 = std::chrono::high_resolution_clock::now();
    snappy::Uncompress(
      snappy_compressed.c_str(), 
      csize, 
      &uncompressed);
    t2 = std::chrono::high_resolution_clock::now();

    auto sdt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    memcpy(&(*sd.begin()), uncompressed.c_str(), td->size() * 4);




    std::cout << "    Decompression time:\t\t" << dt << std::endl;
    std::cout << "   Opt decompress time:\t\t" << odt << std::endl;
    std::cout << "Snappy decompress time:\t\t" << sdt << std::endl;


    std::cout << "    Optimization boost:\t\t" << (dt - odt) * 100.0 / dt 
      << " % " << std::endl;

    std::cout << std::endl;


    std::cout << std::endl;

    std::cout << " ** CHECKING CORRECTNESS ** " << std::endl;
    testEqual(*td, dd);

    std::cout << " ** CHECKING OPTIMIZED CORRECTNESS ** " << std::endl;
    testEqual(*td, ddo);

    std::cout << " ** CHECKING SNAPPY CORRECTNESS ** " << std::endl;
    testEqual(*td, sd);

    std::cout << " _______ " << std::endl;
    std::cout << std::endl;
  }



  return 0;
}

