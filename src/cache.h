#pragma once

// Minimal JS5 cache reader: just enough of the RuneScape cache container
// stack to pull map, config, and texture data out of a locally provided
// cache dump (main_file_cache.dat2 + .idx files, e.g. from an OpenRS2
// archive). Ported from the TypeScript implementation in XRSPS.
//
// The layering, bottom to top:
//   DiskStore        - sector-chained raw group storage (.dat2 + .idx)
//   container        - per-group compression envelope (none/bzip2/gzip),
//                      optionally XTEA-encrypted
//   ReferenceTable   - index 255 metadata: which groups exist, their names
//                      (as djb2 hashes), and the files inside each group
//   Js5Cache         - ties the above together: "give me file F of group G
//                      in index I"
//
// No game assets ship with this repository - the cache/ folder is
// .gitignored and must be provided by the user.

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace js5 {

// Big-endian cursor over a byte span (the cache is Java-heritage, so all
// multi-byte values are big-endian).
class ByteReader {
public:
    ByteReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    explicit ByteReader(const std::vector<uint8_t>& data)
        : data_(data.data()), size_(data.size()) {}

    size_t offset = 0;

    size_t size() const { return size_; }
    size_t remaining() const { return offset < size_ ? size_ - offset : 0; }

    uint8_t u8() {
        check(1);
        return data_[offset++];
    }
    int8_t i8() { return static_cast<int8_t>(u8()); }
    uint16_t u16() {
        check(2);
        const uint16_t v = static_cast<uint16_t>((data_[offset] << 8) | data_[offset + 1]);
        offset += 2;
        return v;
    }
    int16_t i16() { return static_cast<int16_t>(u16()); }
    uint32_t u24() {
        check(3);
        const uint32_t v = (static_cast<uint32_t>(data_[offset]) << 16) |
                           (static_cast<uint32_t>(data_[offset + 1]) << 8) |
                           data_[offset + 2];
        offset += 3;
        return v;
    }
    int32_t i32() {
        check(4);
        const uint32_t v = (static_cast<uint32_t>(data_[offset]) << 24) |
                           (static_cast<uint32_t>(data_[offset + 1]) << 16) |
                           (static_cast<uint32_t>(data_[offset + 2]) << 8) |
                           data_[offset + 3];
        offset += 4;
        return static_cast<int32_t>(v);
    }
    // "big smart": u16 when the top bit is clear, i32 & 0x7fffffff otherwise.
    int32_t bigSmart() {
        check(1);
        if ((data_[offset] & 0x80) != 0) return i32() & 0x7fffffff;
        return u16();
    }
    void skip(size_t n) {
        check(n);
        offset += n;
    }
    std::vector<uint8_t> bytes(size_t n) {
        check(n);
        std::vector<uint8_t> out(data_ + offset, data_ + offset + n);
        offset += n;
        return out;
    }

private:
    void check(size_t n) const {
        if (offset + n > size_) throw std::runtime_error("cache: buffer underflow");
    }
    const uint8_t* data_;
    size_t size_;
};

// Java String.hashCode(); group names ("m48_54") are stored as this hash.
int32_t hashDjb2(const std::string& s);

// Decompresses a container envelope: [u8 compression][i32 size][payload].
// key: optional 4-word XTEA key (nullptr or all-zero = not encrypted).
std::vector<uint8_t> decompressContainer(const std::vector<uint8_t>& data,
                                         const uint32_t* xteaKey = nullptr);

// One group's entry in a reference table.
struct GroupRef {
    int32_t id = -1;
    int32_t nameHash = 0;
    int32_t fileCount = 0;
    std::vector<int32_t> fileIds; // actual (possibly sparse) file ids
};

// Decoded index-255 metadata for one index.
struct ReferenceTable {
    bool named = false;
    std::vector<GroupRef> groups;
    std::unordered_map<int32_t, size_t> idToIndex;
    std::unordered_map<int32_t, int32_t> nameHashToId;

    static ReferenceTable decode(ByteReader& r);

    const GroupRef* group(int32_t id) const {
        auto it = idToIndex.find(id);
        return it == idToIndex.end() ? nullptr : &groups[it->second];
    }
    int32_t groupIdForName(const std::string& name) const {
        auto it = nameHashToId.find(hashDjb2(name));
        return it == nameHashToId.end() ? -1 : it->second;
    }
};

// Splits a decompressed group blob into its files (chunk-interleaved layout
// described by a trailing chunk count). Keyed by actual file id.
std::unordered_map<int32_t, std::vector<uint8_t>> splitGroupFiles(
    const std::vector<uint8_t>& data, const GroupRef& ref);

// The .dat2 + .idx sector store. Index files are held in memory (they are
// small); group payloads are read from the dat2 file on demand by walking
// its 520-byte sector chain.
class DiskStore {
public:
    void open(const std::string& directory);
    bool hasIndex(int indexId) const;
    std::vector<uint8_t> read(int indexId, int groupId);

private:
    mutable std::ifstream dat_;
    std::vector<std::vector<uint8_t>> indexFiles_; // [0..254]
    std::vector<uint8_t> metaIndex_;               // idx255
};

class Js5Cache {
public:
    void open(const std::string& directory);

    const ReferenceTable& table(int indexId);

    // Raw decompressed group payload.
    std::vector<uint8_t> readGroup(int indexId, int groupId,
                                   const uint32_t* xteaKey = nullptr);

    // All files of a group, split per the reference table.
    std::unordered_map<int32_t, std::vector<uint8_t>> readGroupFiles(
        int indexId, int groupId, const uint32_t* xteaKey = nullptr);

    int32_t groupIdForName(int indexId, const std::string& name) {
        return table(indexId).groupIdForName(name);
    }

private:
    DiskStore store_;
    std::vector<std::unique_ptr<ReferenceTable>> tables_;
};

// Parses an OpenRS2-style keys.json ({"<groupId>": [k0,k1,k2,k3], ...}).
// Modern OSRS caches ship no keys (map files stopped being encrypted), so an
// empty file/map is normal.
std::unordered_map<int32_t, std::array<uint32_t, 4>> loadXteaKeys(
    const std::string& path);

} // namespace js5
