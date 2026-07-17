#include "cache.h"

// bzlib.h includes <windows.h> on Windows for its DLL linkage macros; stop
// the min/max macros there from clobbering std::min/std::max below.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif

#include <bzlib.h>
#include <zlib.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace js5 {

namespace {

// --- sector store constants -------------------------------------------------
// The dat2 file is a chain of fixed 520-byte sectors. Groups whose id fits
// in 16 bits use an 8-byte header + 512 data bytes; larger ids use a 10-byte
// header + 510 data bytes. Each .idx file is an array of 6-byte records
// (u24 total size, u24 first sector) indexed by group id.
constexpr size_t kSectorSize = 520;
constexpr size_t kHeaderSize = 8;
constexpr size_t kDataSize = 512;
constexpr size_t kExtHeaderSize = 10;
constexpr size_t kExtDataSize = 510;
constexpr size_t kClusterSize = 6;
constexpr int kMetaIndexId = 255;

// --- XTEA --------------------------------------------------------------------
constexpr uint32_t kXteaGoldenRatio = 0x9e3779b9u;
constexpr int kXteaRounds = 32;

bool xteaKeyValid(const uint32_t* key) {
    return key != nullptr && (key[0] != 0 || key[1] != 0 || key[2] != 0 || key[3] != 0);
}

// In-place XTEA block decryption over [start, end) of `data` (whole 8-byte
// blocks only, matching the client).
void xteaDecrypt(std::vector<uint8_t>& data, size_t start, size_t end,
                 const uint32_t* key) {
    if (end > data.size()) end = data.size();
    if (end <= start) return;
    const size_t blocks = (end - start) / 8;
    for (size_t i = 0; i < blocks; ++i) {
        uint8_t* p = data.data() + start + i * 8;
        uint32_t v0 = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
                      (uint32_t(p[2]) << 8) | p[3];
        uint32_t v1 = (uint32_t(p[4]) << 24) | (uint32_t(p[5]) << 16) |
                      (uint32_t(p[6]) << 8) | p[7];
        uint32_t sum = kXteaGoldenRatio * uint32_t(kXteaRounds);
        for (int r = 0; r < kXteaRounds; ++r) {
            v1 -= (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum + key[(sum >> 11) & 3]);
            sum -= kXteaGoldenRatio;
            v0 -= (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum + key[sum & 3]);
        }
        p[0] = uint8_t(v0 >> 24); p[1] = uint8_t(v0 >> 16);
        p[2] = uint8_t(v0 >> 8);  p[3] = uint8_t(v0);
        p[4] = uint8_t(v1 >> 24); p[5] = uint8_t(v1 >> 16);
        p[6] = uint8_t(v1 >> 8);  p[7] = uint8_t(v1);
    }
}

std::vector<uint8_t> gzipDecompress(const uint8_t* data, size_t size,
                                    size_t expectedSize) {
    std::vector<uint8_t> out(expectedSize);
    z_stream stream{};
    // 15 = max window; +16 selects the gzip wrapper (the cache stores full
    // gzip streams, header and all).
    if (inflateInit2(&stream, 15 + 16) != Z_OK) {
        throw std::runtime_error("cache: inflateInit2 failed");
    }
    stream.next_in = const_cast<Bytef*>(data);
    stream.avail_in = static_cast<uInt>(size);
    stream.next_out = out.data();
    stream.avail_out = static_cast<uInt>(out.size());
    const int result = inflate(&stream, Z_FINISH);
    inflateEnd(&stream);
    if (result != Z_STREAM_END) {
        throw std::runtime_error("cache: gzip stream did not decompress cleanly");
    }
    out.resize(out.size() - stream.avail_out);
    return out;
}

std::vector<uint8_t> bzip2Decompress(const uint8_t* data, size_t size,
                                     size_t expectedSize) {
    // The cache strips the 4-byte "BZh1" magic to save space; restore it.
    std::vector<uint8_t> withHeader(size + 4);
    withHeader[0] = 'B'; withHeader[1] = 'Z'; withHeader[2] = 'h'; withHeader[3] = '1';
    std::memcpy(withHeader.data() + 4, data, size);

    std::vector<uint8_t> out(expectedSize);
    unsigned int outLen = static_cast<unsigned int>(out.size());
    const int result = BZ2_bzBuffToBuffDecompress(
        reinterpret_cast<char*>(out.data()), &outLen,
        reinterpret_cast<char*>(withHeader.data()),
        static_cast<unsigned int>(withHeader.size()), 0, 0);
    if (result != BZ_OK) {
        throw std::runtime_error("cache: bzip2 decompress failed (" +
                                 std::to_string(result) + ")");
    }
    out.resize(outLen);
    return out;
}

} // namespace

int32_t hashDjb2(const std::string& s) {
    // Java String.hashCode with 32-bit wrapping: h = h*31 + c.
    uint32_t hash = 0;
    for (const char c : s) {
        hash = (hash << 5) - hash + static_cast<uint8_t>(c);
    }
    return static_cast<int32_t>(hash);
}

std::vector<uint8_t> decompressContainer(const std::vector<uint8_t>& input,
                                         const uint32_t* xteaKey) {
    if (input.size() < 5) throw std::runtime_error("cache: container too small");
    std::vector<uint8_t> data = input; // mutable copy (XTEA decrypts in place)

    ByteReader r(data);
    const uint8_t compression = r.u8();
    const int32_t size = r.i32();
    if (size < 0 || 5 + static_cast<size_t>(size) > data.size()) {
        throw std::runtime_error("cache: bad container size");
    }
    if (xteaKeyValid(xteaKey)) {
        // The encrypted span covers the (optional) decompressed-size word
        // plus the payload.
        xteaDecrypt(data, r.offset, r.offset + 4 + static_cast<size_t>(size), xteaKey);
    }

    switch (compression) {
        case 0: // none
            return r.bytes(static_cast<size_t>(size));
        case 1: { // bzip2
            const uint32_t actualSize = static_cast<uint32_t>(r.i32());
            return bzip2Decompress(data.data() + r.offset, static_cast<size_t>(size),
                                   actualSize);
        }
        case 2: { // gzip
            const uint32_t actualSize = static_cast<uint32_t>(r.i32());
            return gzipDecompress(data.data() + r.offset, static_cast<size_t>(size),
                                  actualSize);
        }
        default:
            throw std::runtime_error("cache: unsupported compression type " +
                                     std::to_string(compression));
    }
}

ReferenceTable ReferenceTable::decode(ByteReader& r) {
    ReferenceTable table;

    const uint8_t protocol = r.u8();
    if (protocol < 5 || protocol > 7) {
        throw std::runtime_error("cache: invalid reference table protocol");
    }
    if (protocol > 5) r.i32(); // revision, unused here
    const uint8_t flags = r.u8();
    const bool hasNames = (flags & 0x1) != 0;
    const bool hasWhirlpool = (flags & 0x2) != 0;
    const bool hasSizes = (flags & 0x4) != 0;
    // flag 0x8 (uncompressed CRCs) carries no extra table data we read past.
    table.named = hasNames;

    auto readCount = [&]() -> int32_t {
        return protocol == 7 ? r.bigSmart() : static_cast<int32_t>(r.u16());
    };

    const int32_t groupCount = readCount();
    table.groups.resize(static_cast<size_t>(groupCount));

    int32_t lastId = 0;
    for (int32_t i = 0; i < groupCount; ++i) {
        lastId += readCount(); // delta-encoded ids
        table.groups[i].id = lastId;
        table.idToIndex.emplace(lastId, i);
    }
    if (hasNames) {
        for (int32_t i = 0; i < groupCount; ++i) {
            table.groups[i].nameHash = r.i32();
            table.nameHashToId.emplace(table.groups[i].nameHash, table.groups[i].id);
        }
    }
    if (hasWhirlpool) r.skip(static_cast<size_t>(groupCount) * 64);
    r.skip(static_cast<size_t>(groupCount) * 4); // crcs
    if (hasSizes) r.skip(static_cast<size_t>(groupCount) * 8);
    r.skip(static_cast<size_t>(groupCount) * 4); // revisions

    for (int32_t i = 0; i < groupCount; ++i) {
        table.groups[i].fileCount = readCount();
    }
    for (int32_t i = 0; i < groupCount; ++i) {
        GroupRef& group = table.groups[i];
        group.fileIds.resize(static_cast<size_t>(group.fileCount));
        int32_t lastFileId = 0;
        for (int32_t f = 0; f < group.fileCount; ++f) {
            lastFileId += readCount();
            group.fileIds[f] = lastFileId;
        }
    }
    if (hasNames) {
        // Per-file name hashes - present in the stream, unused by us.
        for (int32_t i = 0; i < groupCount; ++i) {
            r.skip(static_cast<size_t>(table.groups[i].fileCount) * 4);
        }
    }
    return table;
}

std::unordered_map<int32_t, std::vector<uint8_t>> splitGroupFiles(
    const std::vector<uint8_t>& data, const GroupRef& ref) {
    std::unordered_map<int32_t, std::vector<uint8_t>> files;
    if (ref.fileCount <= 0) return files;

    if (ref.fileCount == 1) {
        files.emplace(ref.fileIds[0], data);
        return files;
    }

    // Multi-file groups are stored as N interleaved chunks; a trailing byte
    // gives the chunk count and a footer table gives delta-encoded sizes.
    if (data.empty()) throw std::runtime_error("cache: empty multi-file group");
    const size_t fileCount = static_cast<size_t>(ref.fileCount);
    const uint8_t chunks = data[data.size() - 1];
    const size_t footer = static_cast<size_t>(chunks) * fileCount * 4 + 1;
    if (footer > data.size()) throw std::runtime_error("cache: bad group footer");

    ByteReader sizes(data.data() + (data.size() - footer), footer - 1);
    std::vector<int32_t> chunkSizes(static_cast<size_t>(chunks) * fileCount);
    std::vector<size_t> fileSizes(fileCount, 0);
    for (uint8_t chunk = 0; chunk < chunks; ++chunk) {
        int32_t running = 0;
        for (size_t f = 0; f < fileCount; ++f) {
            running += sizes.i32();
            chunkSizes[chunk * fileCount + f] = running;
            fileSizes[f] += static_cast<size_t>(running);
        }
    }

    std::vector<std::vector<uint8_t>> fileData(fileCount);
    for (size_t f = 0; f < fileCount; ++f) fileData[f].reserve(fileSizes[f]);

    size_t offset = 0;
    for (uint8_t chunk = 0; chunk < chunks; ++chunk) {
        for (size_t f = 0; f < fileCount; ++f) {
            const size_t chunkSize =
                static_cast<size_t>(chunkSizes[chunk * fileCount + f]);
            if (offset + chunkSize > data.size()) {
                throw std::runtime_error("cache: group chunk out of range");
            }
            fileData[f].insert(fileData[f].end(), data.begin() + offset,
                               data.begin() + offset + chunkSize);
            offset += chunkSize;
        }
    }
    for (size_t f = 0; f < fileCount; ++f) {
        files.emplace(ref.fileIds[f], std::move(fileData[f]));
    }
    return files;
}

void DiskStore::open(const std::string& directory) {
    namespace fs = std::filesystem;
    const fs::path dir(directory);

    dat_.open(dir / "main_file_cache.dat2", std::ios::binary);
    if (!dat_) {
        throw std::runtime_error("cache: main_file_cache.dat2 not found in " + directory);
    }

    auto loadFile = [](const fs::path& path) -> std::vector<uint8_t> {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) return {};
        const std::streamsize size = file.tellg();
        std::vector<uint8_t> data(static_cast<size_t>(size));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(data.data()), size);
        return data;
    };

    indexFiles_.resize(kMetaIndexId);
    for (int i = 0; i < kMetaIndexId; ++i) {
        indexFiles_[i] =
            loadFile(dir / ("main_file_cache.idx" + std::to_string(i)));
    }
    metaIndex_ = loadFile(dir / ("main_file_cache.idx" + std::to_string(kMetaIndexId)));
    if (metaIndex_.empty()) {
        throw std::runtime_error("cache: main_file_cache.idx255 not found in " +
                                 directory);
    }
}

bool DiskStore::hasIndex(int indexId) const {
    if (indexId == kMetaIndexId) return !metaIndex_.empty();
    return indexId >= 0 && indexId < static_cast<int>(indexFiles_.size()) &&
           !indexFiles_[indexId].empty();
}

std::vector<uint8_t> DiskStore::read(int indexId, int groupId) {
    const std::vector<uint8_t>& indexFile =
        indexId == kMetaIndexId ? metaIndex_ : indexFiles_.at(static_cast<size_t>(indexId));
    if (indexFile.empty()) {
        throw std::runtime_error("cache: index " + std::to_string(indexId) + " missing");
    }

    const size_t clusterPtr = static_cast<size_t>(groupId) * kClusterSize;
    if (clusterPtr + kClusterSize > indexFile.size()) {
        throw std::runtime_error("cache: group " + std::to_string(groupId) +
                                 " out of range for index " + std::to_string(indexId));
    }
    ByteReader cluster(indexFile.data() + clusterPtr, kClusterSize);
    const uint32_t totalSize = cluster.u24();
    uint32_t sector = cluster.u24();
    if (totalSize == 0 || sector == 0) {
        throw std::runtime_error("cache: group " + std::to_string(groupId) + " is empty");
    }

    const bool extended = groupId > 0xffff;
    const size_t headerSize = extended ? kExtHeaderSize : kHeaderSize;
    const size_t dataSize = extended ? kExtDataSize : kDataSize;

    std::vector<uint8_t> out(totalSize);
    std::vector<uint8_t> sectorBuf(kSectorSize);
    size_t written = 0;
    uint32_t expectedChunk = 0;

    while (written < totalSize) {
        const size_t want = std::min(dataSize, totalSize - written);

        dat_.seekg(static_cast<std::streamoff>(sector) *
                   static_cast<std::streamoff>(kSectorSize));
        dat_.read(reinterpret_cast<char*>(sectorBuf.data()),
                  static_cast<std::streamsize>(headerSize + want));
        if (!dat_) throw std::runtime_error("cache: dat2 read failed (truncated?)");

        ByteReader r(sectorBuf.data(), headerSize + want);
        const int32_t sectorGroup =
            extended ? r.i32() : static_cast<int32_t>(r.u16());
        const uint32_t chunk = r.u16();
        const uint32_t nextSector = r.u24();
        r.u8(); // sector's index id; validated implicitly by the chain

        if (sectorGroup != groupId || chunk != expectedChunk) {
            throw std::runtime_error("cache: sector chain mismatch for group " +
                                     std::to_string(groupId));
        }
        std::memcpy(out.data() + written, sectorBuf.data() + headerSize, want);
        written += want;
        sector = nextSector;
        ++expectedChunk;
    }
    return out;
}

void Js5Cache::open(const std::string& directory) {
    store_.open(directory);
    tables_.resize(kMetaIndexId);
}

const ReferenceTable& Js5Cache::table(int indexId) {
    auto& slot = tables_.at(static_cast<size_t>(indexId));
    if (!slot) {
        const std::vector<uint8_t> raw = store_.read(kMetaIndexId, indexId);
        const std::vector<uint8_t> decoded = decompressContainer(raw);
        ByteReader r(decoded);
        slot = std::make_unique<ReferenceTable>(ReferenceTable::decode(r));
    }
    return *slot;
}

std::vector<uint8_t> Js5Cache::readGroup(int indexId, int groupId,
                                         const uint32_t* xteaKey) {
    return decompressContainer(store_.read(indexId, groupId), xteaKey);
}

std::unordered_map<int32_t, std::vector<uint8_t>> Js5Cache::readGroupFiles(
    int indexId, int groupId, const uint32_t* xteaKey) {
    const GroupRef* ref = table(indexId).group(groupId);
    if (ref == nullptr) {
        throw std::runtime_error("cache: no reference for group " +
                                 std::to_string(groupId));
    }
    return splitGroupFiles(readGroup(indexId, groupId, xteaKey), *ref);
}

std::unordered_map<int32_t, std::array<uint32_t, 4>> loadXteaKeys(
    const std::string& path) {
    // Purpose-built scan for OpenRS2's {"groupId": [k0,k1,k2,k3], ...}
    // shape - not a general JSON parser, but robust for that format.
    std::unordered_map<int32_t, std::array<uint32_t, 4>> keys;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return keys;
    std::string text(static_cast<size_t>(file.tellg()), '\0');
    file.seekg(0);
    file.read(text.data(), static_cast<std::streamsize>(text.size()));

    size_t pos = 0;
    auto skipWs = [&]() { while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos; };
    skipWs();
    if (pos >= text.size() || text[pos] != '{') return keys;
    ++pos;
    while (true) {
        skipWs();
        if (pos >= text.size() || text[pos] == '}') break;
        if (text[pos] != '"') break;
        const size_t keyStart = ++pos;
        while (pos < text.size() && text[pos] != '"') ++pos;
        const int32_t groupId = std::atoi(text.substr(keyStart, pos - keyStart).c_str());
        ++pos; // closing quote
        skipWs();
        if (pos >= text.size() || text[pos] != ':') break;
        ++pos;
        skipWs();
        if (pos >= text.size() || text[pos] != '[') break;
        ++pos;
        std::array<uint32_t, 4> key{};
        for (int i = 0; i < 4; ++i) {
            skipWs();
            const size_t numStart = pos;
            while (pos < text.size() && text[pos] != ',' && text[pos] != ']') ++pos;
            key[static_cast<size_t>(i)] = static_cast<uint32_t>(
                std::atoll(text.substr(numStart, pos - numStart).c_str()));
            if (pos < text.size() && text[pos] == ',') ++pos;
        }
        while (pos < text.size() && text[pos] != ']') ++pos;
        if (pos < text.size()) ++pos; // ']'
        keys.emplace(groupId, key);
        skipWs();
        if (pos < text.size() && text[pos] == ',') ++pos;
    }
    return keys;
}

} // namespace js5
