#include <sextant/engine_trace.hpp>

namespace sextant {

std::unique_ptr<EngineTrace> EngineTrace::create(const std::string& path,
                                                 std::string_view run_label) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return nullptr;
    }
    auto t = std::unique_ptr<EngineTrace>(new EngineTrace());
    t->out_ = std::move(out);
    // Substitute any newlines in the label — the header is one line.
    std::string label(run_label);
    for (char& c : label) {
        if (c == '\n' || c == '\r') c = ' ';
    }
    t->out_ << "# sextant-trace v1 label=\"" << label << "\"\n";
    return t;
}

}  // namespace sextant
