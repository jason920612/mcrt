#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace mcrt {

// Defers destruction of GPU objects until the timeline value of the last frame that could
// reference them has completed.
class DeletionQueue {
public:
    void push(uint64_t retireValue, std::function<void()> destroy) {
        entries_.emplace_back(retireValue, std::move(destroy));
    }

    void collect(uint64_t completedValue) {
        size_t kept = 0;
        for (auto& entry : entries_) {
            if (entry.first <= completedValue)
                entry.second();
            else
                entries_[kept++] = std::move(entry);
        }
        entries_.resize(kept);
    }

    void flushAll() {
        for (auto& entry : entries_)
            entry.second();
        entries_.clear();
    }

private:
    std::vector<std::pair<uint64_t, std::function<void()>>> entries_;
};

} // namespace mcrt
