#include "filter_data_io.hpp"
#include "sextant/error.hpp"

#include <cstring>

namespace sextant::tree {

void write_filter_data(const std::string& path,
                        const Schema& schema,
                        const std::vector<ColumnData>& cols,
                        const std::vector<uint8_t>& payload_data,
                        const std::vector<uint32_t>& payload_offsets) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw Error(ErrorCode::IoError, "Cannot open " + path + " for writing");

    const uint64_t n_rows = schema.columns.empty() ? 0 :
        (cols[0].type == ColumnType::Int32 || cols[0].type == ColumnType::Int64 ||
         cols[0].type == ColumnType::Float || cols[0].type == ColumnType::Bool)
        ? cols[0].fixed_data.size() / column_type_width(cols[0].type)
        : cols[0].str_offsets.size();

    const uint32_t n_cols = static_cast<uint32_t>(schema.columns.size());

    // Header.
    f.write(reinterpret_cast<const char*>(&kFilterDataMagic), 4);
    f.write(reinterpret_cast<const char*>(&n_rows), 8);
    f.write(reinterpret_cast<const char*>(&n_cols), 4);
    for (const auto& col : schema.columns) {
        const uint8_t t = static_cast<uint8_t>(col.type);
        f.write(reinterpret_cast<const char*>(&t), 1);
    }

    // Schema column names (length-prefixed).
    for (const auto& col : schema.columns) {
        const uint16_t len = static_cast<uint16_t>(col.name.size());
        f.write(reinterpret_cast<const char*>(&len), 2);
        f.write(col.name.data(), len);
    }

    // Payload.
    const uint8_t has_payload = !payload_data.empty() ? 1 : 0;
    f.write(reinterpret_cast<const char*>(&has_payload), 1);
    if (has_payload) {
        // Write offsets (n_rows+1 entries) + data.
        f.write(reinterpret_cast<const char*>(payload_offsets.data()),
                payload_offsets.size() * 4);
        f.write(reinterpret_cast<const char*>(payload_data.data()),
                payload_data.size());
    }

    // Per-column data.
    for (uint32_t c = 0; c < n_cols; ++c) {
        const auto& col = cols[c];
        switch (col.type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float:
            case ColumnType::Bool:
                f.write(reinterpret_cast<const char*>(col.fixed_data.data()),
                        col.fixed_data.size());
                break;
            case ColumnType::String:
                f.write(reinterpret_cast<const char*>(col.str_offsets.data()),
                        col.str_offsets.size() * 4);
                f.write(reinterpret_cast<const char*>(col.str_lengths.data()),
                        col.str_lengths.size() * 2);
                f.write(col.str_data.data(), col.str_data.size());
                break;
            case ColumnType::Set:
                f.write(reinterpret_cast<const char*>(col.set_counts.data()),
                        col.set_counts.size());
                f.write(reinterpret_cast<const char*>(col.set_offsets.data()),
                        col.set_offsets.size() * 4);
                f.write(reinterpret_cast<const char*>(col.set_elem_lengths.data()),
                        col.set_elem_lengths.size() * 2);
                f.write(col.set_elem_data.data(), col.set_elem_data.size());
                break;
        }
    }
}

FilterDataFile read_filter_data(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error(ErrorCode::IoError, "Cannot open " + path + " for reading");

    FilterDataFile result;

    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != kFilterDataMagic)
        throw Error(ErrorCode::CorruptIndex, "Bad filter data magic in " + path);

    f.read(reinterpret_cast<char*>(&result.n_rows), 8);
    uint32_t n_cols;
    f.read(reinterpret_cast<char*>(&n_cols), 4);

    // Column types.
    result.schema.columns.resize(n_cols);
    for (uint32_t c = 0; c < n_cols; ++c) {
        uint8_t t;
        f.read(reinterpret_cast<char*>(&t), 1);
        result.schema.columns[c].type = static_cast<ColumnType>(t);
    }

    // Column names.
    for (uint32_t c = 0; c < n_cols; ++c) {
        uint16_t len;
        f.read(reinterpret_cast<char*>(&len), 2);
        result.schema.columns[c].name.resize(len);
        f.read(result.schema.columns[c].name.data(), len);
    }

    // Payload.
    f.read(reinterpret_cast<char*>(&result.has_payload), 1);
    if (result.has_payload) {
        result.payload_offsets.resize(result.n_rows + 1);
        f.read(reinterpret_cast<char*>(result.payload_offsets.data()),
               result.payload_offsets.size() * 4);
        // Data size = last offset.
        const uint32_t data_size = result.payload_offsets.back();
        result.payload_data.resize(data_size);
        f.read(reinterpret_cast<char*>(result.payload_data.data()), data_size);
    }

    // Per-column data.
    result.cols.resize(n_cols);
    for (uint32_t c = 0; c < n_cols; ++c) {
        auto& col = result.cols[c];
        col.type = result.schema.columns[c].type;
        switch (col.type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float:
            case ColumnType::Bool: {
                const uint8_t w = column_type_width(col.type);
                col.fixed_data.resize(result.n_rows * w);
                f.read(reinterpret_cast<char*>(col.fixed_data.data()),
                       col.fixed_data.size());
                break;
            }
            case ColumnType::String: {
                col.str_offsets.resize(result.n_rows);
                f.read(reinterpret_cast<char*>(col.str_offsets.data()),
                       result.n_rows * 4);
                col.str_lengths.resize(result.n_rows);
                f.read(reinterpret_cast<char*>(col.str_lengths.data()),
                       result.n_rows * 2);
                // Data size = last offset + last length.
                const uint32_t data_size = (result.n_rows > 0)
                    ? col.str_offsets.back() + col.str_lengths.back() : 0;
                col.str_data.resize(data_size);
                f.read(col.str_data.data(), data_size);
                break;
            }
            case ColumnType::Set: {
                col.set_counts.resize(result.n_rows);
                f.read(reinterpret_cast<char*>(col.set_counts.data()),
                       result.n_rows);
                col.set_offsets.resize(result.n_rows);
                f.read(reinterpret_cast<char*>(col.set_offsets.data()),
                       result.n_rows * 4);
                // Total elements = sum of counts.
                uint32_t total_elems = 0;
                for (uint32_t i = 0; i < result.n_rows; ++i)
                    total_elems += col.set_counts[i];
                col.set_elem_lengths.resize(total_elems);
                f.read(reinterpret_cast<char*>(col.set_elem_lengths.data()),
                       total_elems * 2);
                // Data size = sum of element lengths.
                uint32_t data_size = 0;
                for (uint32_t i = 0; i < total_elems; ++i)
                    data_size += col.set_elem_lengths[i];
                col.set_elem_data.resize(data_size);
                f.read(col.set_elem_data.data(), data_size);
                break;
            }
        }
    }

    return result;
}

}  // namespace sextant::tree
