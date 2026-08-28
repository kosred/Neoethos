#include <Random123/philox.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct DirectCase {
  const char* id;
  std::array<std::uint32_t, 4> counter;
  std::array<std::uint32_t, 2> key;
};

struct AddressCase {
  const char* id;
  std::uint64_t search_seed;
  std::array<std::uint8_t, 32> run_identity;
  std::uint32_t generation;
  std::uint64_t candidate;
  std::uint32_t operator_identity;
  std::uint32_t decision_slot;
  std::uint32_t rejection_attempt;
};

struct Address {
  std::array<std::uint32_t, 4> counter;
  std::array<std::uint32_t, 2> key;
};

std::array<std::uint32_t, 4> official_random123_philox4x32_10(
    const std::array<std::uint32_t, 4>& counter,
    const std::array<std::uint32_t, 2>& key) {
  philox4x32_ctr_t c = {{counter[0], counter[1], counter[2], counter[3]}};
  philox4x32_key_t k = {{key[0], key[1]}};
  const philox4x32_ctr_t output = philox4x32_R(10, c, k);
  return {output.v[0], output.v[1], output.v[2], output.v[3]};
}

Address candidate_address_mapping_v1(const AddressCase& input) {
  const std::uint32_t run_word_0 =
      static_cast<std::uint32_t>(input.run_identity[0]) |
      (static_cast<std::uint32_t>(input.run_identity[1]) << 8) |
      (static_cast<std::uint32_t>(input.run_identity[2]) << 16) |
      (static_cast<std::uint32_t>(input.run_identity[3]) << 24);
  const std::uint32_t run_word_1 =
      static_cast<std::uint32_t>(input.run_identity[4]) |
      (static_cast<std::uint32_t>(input.run_identity[5]) << 8) |
      (static_cast<std::uint32_t>(input.run_identity[6]) << 16) |
      (static_cast<std::uint32_t>(input.run_identity[7]) << 24);
  return Address{
      {static_cast<std::uint32_t>(input.candidate),
       static_cast<std::uint32_t>(input.candidate >> 32), input.generation,
       input.rejection_attempt},
      {static_cast<std::uint32_t>(input.search_seed) ^ run_word_0 ^
           input.operator_identity,
       static_cast<std::uint32_t>(input.search_seed >> 32) ^ run_word_1 ^
           input.decision_slot},
  };
}

void print_words(const std::array<std::uint32_t, 4>& words) {
  std::printf("\t%08x\t%08x\t%08x\t%08x", words[0], words[1], words[2],
              words[3]);
}

void print_run_identity(const std::array<std::uint8_t, 32>& identity) {
  for (const std::uint8_t byte : identity) {
    std::printf("%02x", static_cast<unsigned>(byte));
  }
}

constexpr std::array<std::uint8_t, 32> sequential_run_identity() {
  return {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
          0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
          0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
          0x98, 0xa9, 0xba, 0xcb, 0xdc, 0xed, 0xfe, 0x0f};
}

constexpr std::array<std::uint8_t, 32> edge_run_identity() {
  return {0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01,
          0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
          0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88,
          0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00};
}

constexpr std::array<std::uint8_t, 32> repeated(std::uint8_t value) {
  std::array<std::uint8_t, 32> result{};
  for (auto& byte : result) {
    byte = value;
  }
  return result;
}

constexpr std::array<std::uint8_t, 32> tail_variant(std::uint8_t tail) {
  std::array<std::uint8_t, 32> result{};
  result[0] = 0x01;
  result[1] = 0x23;
  result[2] = 0x45;
  result[3] = 0x67;
  result[4] = 0x89;
  result[5] = 0xab;
  result[6] = 0xcd;
  result[7] = 0xef;
  for (std::size_t index = 8; index < result.size(); ++index) {
    result[index] = tail;
  }
  return result;
}

void print_direct() {
  constexpr DirectCase cases[] = {
      {"counter_low_one", {0x00000001, 0, 0, 0}, {0, 0}},
      {"counter_and_key_high_bits",
       {0x80000000, 0x00000000, 0x80000000, 0xffffffff},
       {0x00000000, 0x80000000}},
      {"key_edges",
       {0x00000000, 0xffffffff, 0x00000001, 0x80000000},
       {0xffffffff, 0x00000000}},
      {"alternating_and_nonzero",
       {0xaaaaaaaa, 0x55555555, 0xdeadbeef, 0x01020304},
       {0x13579bdf, 0x2468ace0}},
      {"carry_edges",
       {0xffffffff, 0x00000000, 0xffffffff, 0x00000000},
       {0x00000001, 0xffffffff}},
  };
  std::puts("# case_id\tcounter0\tcounter1\tcounter2\tcounter3\tkey0\tkey1\texpected0\texpected1\texpected2\texpected3");
  for (const auto& test_case : cases) {
    const auto output =
        official_random123_philox4x32_10(test_case.counter, test_case.key);
    std::printf("%s", test_case.id);
    print_words(test_case.counter);
    std::printf("\t%08x\t%08x", test_case.key[0], test_case.key[1]);
    print_words(output);
    std::putchar('\n');
  }
}

void print_address() {
  constexpr AddressCase cases[] = {
      {"nonzero_parent_a", 0x0123456789abcdefULL, sequential_run_identity(),
       0x00000001, 0x0000000000000001ULL, 8, 0, 0},
      {"candidate_high_parent_b", 0x0123456789abcdefULL,
       sequential_run_identity(), 0x7fffffff, 0x0000000100000000ULL, 9,
       1, 0},
      {"decision_slot_high_retry_one", 0x8000000000000001ULL,
       edge_run_identity(), 0x80000000, 0x8000000000000001ULL, 12,
       0x80000000, 1},
      {"all_max_survivor", 0xffffffffffffffffULL, repeated(0xff),
       0xffffffff, 0xffffffffffffffffULL, 14, 0xffffffff, 0xffffffff},
      {"operator_parent_a", 0x0f1e2d3c4b5a6978ULL, edge_run_identity(),
       9, 42, 8, 7, 3},
      {"operator_parent_b", 0x0f1e2d3c4b5a6978ULL, edge_run_identity(),
       9, 42, 9, 7, 3},
      {"retry_zero", 0x0f1e2d3c4b5a6978ULL, edge_run_identity(), 9, 42,
       12, 7, 0},
      {"retry_max", 0x0f1e2d3c4b5a6978ULL, edge_run_identity(), 9, 42,
       12, 7, 0xffffffff},
      {"tail_unbound_zero", 0x1111222233334444ULL, tail_variant(0x00),
       5, 0x100000002ULL, 5, 6, 7},
      {"tail_unbound_ff", 0x1111222233334444ULL, tail_variant(0xff), 5,
       0x100000002ULL, 5, 6, 7},
  };
  std::puts("# case_id\tsearch_seed\trun_identity_sha256\tgeneration\tcandidate_identity\toperator_identity\tdecision_slot\trejection_attempt\tcounter0\tcounter1\tcounter2\tcounter3\tkey0\tkey1\texpected0\texpected1\texpected2\texpected3");
  for (const auto& test_case : cases) {
    const Address address = candidate_address_mapping_v1(test_case);
    const auto output =
        official_random123_philox4x32_10(address.counter, address.key);
    std::printf("%s\t%016llx\t", test_case.id,
                static_cast<unsigned long long>(test_case.search_seed));
    print_run_identity(test_case.run_identity);
    std::printf("\t%08x\t%016llx\t%08x\t%08x\t%08x", test_case.generation,
                static_cast<unsigned long long>(test_case.candidate),
                test_case.operator_identity, test_case.decision_slot,
                test_case.rejection_attempt);
    print_words(address.counter);
    std::printf("\t%08x\t%08x", address.key[0], address.key[1]);
    print_words(output);
    std::putchar('\n');
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s direct|address\n", argv[0]);
    return 2;
  }
  if (std::strcmp(argv[1], "direct") == 0) {
    print_direct();
    return 0;
  }
  if (std::strcmp(argv[1], "address") == 0) {
    print_address();
    return 0;
  }
  std::fprintf(stderr, "unknown mode: %s\n", argv[1]);
  return 2;
}
