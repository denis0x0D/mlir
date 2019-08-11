//===- mlir-vulkan-runtime.h - MLIR Vulkan Execution Driver---------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_VULKAN_RUNTIME_H
#define MLIR_VULKAN_RUNTIME_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"

#include <string>
#include <vulkan/vulkan.h>

using Descriptor = int32_t;
struct VulkanDeviceMemoryBuffer {
  VkBuffer buffer;
  VkDeviceMemory deviceMemory;
  int32_t descriptor{0};
};

struct VulkanBufferContent {
  void *ptr{nullptr};
  int64_t size{0};
};

struct VulkanMemoryContext {
  uint32_t queueFamilyIndex{0};
  uint32_t memoryTypeIndex{VK_MAX_MEMORY_TYPES};
  VkDeviceSize memorySize{0};
};

struct VulkanExecutionContext {
  struct LocalSize {
    int32_t x{1};
    int32_t y{1};
    int32_t z{1};
  } localSize;
  // TODO; Is it better to use SmallString?
  std::string entryPoint;
};

extern mlir::LogicalResult
runOnShader(llvm::SmallVectorImpl<char> &binaryShader,
            llvm::DenseMap<Descriptor, VulkanBufferContent> &data,
            const VulkanExecutionContext &);
extern mlir::LogicalResult
runOnModule(mlir::ModuleOp, llvm::DenseMap<Descriptor, VulkanBufferContent> &);

#endif // MLIR_VULKAN_RUNTIME_H
