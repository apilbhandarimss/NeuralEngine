#pragma once

namespace neural {

enum class DeviceType {
    CPU
    // Future: CUDA, Metal, Vulkan, etc.
};

struct Device {
    DeviceType type = DeviceType::CPU;
};

} // namespace neural
