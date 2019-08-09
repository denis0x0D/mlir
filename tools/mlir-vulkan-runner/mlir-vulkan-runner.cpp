//===- mlir-vulkan-runner.cpp - MLIR Vulkan Execution Driver---------------===//
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

#include "mlir/Analysis/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ToolOutputFile.h"

#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>
#include <iostream>
#include <fstream>

using namespace mlir;
using namespace llvm;
using Descriptor = int32_t;

inline void emit_vulkan_error(const llvm::Twine &message, VkResult error) {
  llvm::errs()
      << message.concat(" failed with error code ").concat(llvm::Twine{error});
}

#define RETURN_ON_VULKAN_ERROR(result, msg)                                    \
  if ((result) != VK_SUCCESS) {                                                \
    emit_vulkan_error(msg, (result));                                          \
    return failure();                                                          \
  }

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static void processVariable(spirv::VariableOp varOp) {
  auto descriptorSetName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::DescriptorSet));
  auto bindingName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::Binding));
  auto descriptorSet = varOp.getAttrOfType<IntegerAttr>(descriptorSetName);
  auto binding = varOp.getAttrOfType<IntegerAttr>(bindingName);
  if (descriptorSet && binding) {
    std::cout << "( " << descriptorSet.getInt() << " , " << binding.getInt()
              << " )" << std::endl;
  }
}

static void Print(int32_t *result, int size) {
  std::cout << "buffer started with size" << size << std::endl;
  for (int i = 0; i < size; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << "buffer ended" << std::endl;
}

static uint32_t *ReadFromFile(size_t *size_out, const char *filename_to_read) {
  if (!filename_to_read)
    return nullptr;

  char *shader = nullptr;
  std::ifstream stream(filename_to_read, std::ios::ate);
  if (stream.is_open()) {
    size_t size = stream.tellg();
    *size_out = size;
    stream.seekg(0, std::ios::beg);
    shader = (char *)malloc(size);
    stream.read(shader, size);
    stream.close();
  }
  return reinterpret_cast<uint32_t *>(shader);
}

struct VulkanDeviceMemoryBuffer {
  VkBuffer buffer;
  VkDeviceMemory deviceMemory;
  int descriptor;
};

struct VulkanBufferContent {
  void *ptr;
  int64_t size;
};

struct VulkanMemoryContext {
  uint32_t queueFamilyIndex;
  uint32_t memoryTypeIndex;
  VkDeviceSize memorySize;
};

static LogicalResult
vkGetBestComputeQueueNPH(const VkPhysicalDevice &physicalDevice,
                         uint32_t &queueFamilyIndex) {
  uint32_t queueFamilyPropertiesCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice,
                                           &queueFamilyPropertiesCount, 0);
  std::vector<VkQueueFamilyProperties> queueFamilyProperties(
      queueFamilyPropertiesCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice,
                                           &queueFamilyPropertiesCount,
                                           queueFamilyProperties.data());

  for (uint32_t i = 0; i < queueFamilyPropertiesCount; ++i) {
    const VkQueueFlags maskedFlags =
        (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
         queueFamilyProperties[i].queueFlags);

    if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags) &&
        (VK_QUEUE_COMPUTE_BIT & maskedFlags)) {
      queueFamilyIndex = i;
      return success();
    }
  }

  // Try to find other queue.
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
    const VkQueueFlags maskedFlags =
        (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
         queueFamilyProperties[i].queueFlags);

    if (VK_QUEUE_COMPUTE_BIT & maskedFlags) {
      queueFamilyIndex = i;
      return success();
    }
  }
  return failure();
}

static LogicalResult vulkanCreateInstance(VkInstance &instance) {
  VkApplicationInfo applicationInfo;
  applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  applicationInfo.pNext = nullptr;
  applicationInfo.pApplicationName = "Vulkan MLIR runtime";
  applicationInfo.applicationVersion = 0;
  applicationInfo.pEngineName = "mlir";
  applicationInfo.engineVersion = 0;
  applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 9);

  VkInstanceCreateInfo instanceCreateInfo;
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pNext = nullptr;
  instanceCreateInfo.flags =  0;
  instanceCreateInfo.pApplicationInfo = &applicationInfo; 
  instanceCreateInfo.enabledLayerCount = 0;
  instanceCreateInfo.ppEnabledLayerNames = 0;
  instanceCreateInfo.enabledExtensionCount = 0;
  instanceCreateInfo.ppEnabledExtensionNames = 0;

  RETURN_ON_VULKAN_ERROR(vkCreateInstance(&instanceCreateInfo, 0, &instance),
                         "vkCreateInstance");
  return success();
}

static LogicalResult
createMemoryBuffer(const VkDevice &device,
                   std::pair<Descriptor, VulkanBufferContent> var,
                   const VulkanMemoryContext &memoryContext,
                   VulkanDeviceMemoryBuffer &memoryBuffer) {
  memoryBuffer.descriptor = var.first;
  // TODO: Check that the size is not 0, because it will fail.
  const int64_t bufferSize = var.second.size;

  VkMemoryAllocateInfo memoryAllocateInfo;
  memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memoryAllocateInfo.pNext = nullptr;
  memoryAllocateInfo.allocationSize = bufferSize;
  memoryAllocateInfo.memoryTypeIndex = memoryContext.memoryTypeIndex;
  // Allocate the device memory.
  RETURN_ON_VULKAN_ERROR(vkAllocateMemory(device, &memoryAllocateInfo, 0,
                                          &memoryBuffer.deviceMemory),
                         "vkAllocateMemory");
  void *payload;
  RETURN_ON_VULKAN_ERROR(vkMapMemory(device, memoryBuffer.deviceMemory, 0,
                                     bufferSize, 0, (void **)&payload),
                         "vkMapMemory");
  // TODO: eliminate memcpy?
  memcpy(payload, var.second.ptr, var.second.size);
  vkUnmapMemory(device, memoryBuffer.deviceMemory);

  VkBufferCreateInfo bufferCreateInfo;
  bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.pNext = nullptr;
  bufferCreateInfo.flags = 0;
  bufferCreateInfo.size = bufferSize;
  bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  bufferCreateInfo.queueFamilyIndexCount = 1;
  bufferCreateInfo.pQueueFamilyIndices = &memoryContext.queueFamilyIndex;
  RETURN_ON_VULKAN_ERROR(
      vkCreateBuffer(device, &bufferCreateInfo, 0, &memoryBuffer.buffer),
      "vkCreateBuffer");
  RETURN_ON_VULKAN_ERROR(vkBindBufferMemory(device, memoryBuffer.buffer,
                                            memoryBuffer.deviceMemory, 0),
                         "vkBindBufferMemory");
  return success();
}

static void createDescriptorBufferInfoAndUpdateDesriptorSets(
    const VkDevice &device,
    std::vector<VulkanDeviceMemoryBuffer> &memoryBuffers,
    VkDescriptorSet &descriptorSet) {

  for (auto memoryBuffer : memoryBuffers) {
    VkDescriptorBufferInfo descriptorBufferInfo;
    descriptorBufferInfo.buffer = memoryBuffer.buffer;
    descriptorBufferInfo.offset = 0;
    descriptorBufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet wSet;
    wSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wSet.pNext = nullptr;
    wSet.dstSet = descriptorSet;
    wSet.dstBinding = memoryBuffer.descriptor;
    wSet.dstArrayElement = 0;
    wSet.descriptorCount = 1;
    wSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wSet.pImageInfo = nullptr;
    wSet.pBufferInfo = &descriptorBufferInfo;
    wSet.pTexelBufferView = nullptr;
    vkUpdateDescriptorSets(device, 1, &wSet, 0, nullptr);
  }
}

static LogicalResult vulkanCreateDevice(const VkInstance &instance,
                                        VulkanMemoryContext &memoryContext,
                                        VkDevice &device) {
  uint32_t physicalDeviceCount = 0;
  RETURN_ON_VULKAN_ERROR(
      vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, 0),
      "vkEnumeratePhysicalDevices");

  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  RETURN_ON_VULKAN_ERROR(vkEnumeratePhysicalDevices(instance,
                                                    &physicalDeviceCount,
                                                    physicalDevices.data()),
                         "vkEnumeratePhysicalDevices");

  if (!physicalDeviceCount) {
    // TOOD: generate an error message.
    return failure();
  }

  // TODO: find the best device.
  vkGetBestComputeQueueNPH(physicalDevices[0], memoryContext.queueFamilyIndex);

  const float queuePrioritory = 1.0f;
  VkDeviceQueueCreateInfo deviceQueueCreateInfo;
  deviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  deviceQueueCreateInfo.pNext = nullptr;
  deviceQueueCreateInfo.flags = 0;
  deviceQueueCreateInfo.queueFamilyIndex = memoryContext.queueFamilyIndex;
  deviceQueueCreateInfo.queueCount = 1;
  deviceQueueCreateInfo.pQueuePriorities = &queuePrioritory;

  // Structure specifying parameters of a newly created device
  VkDeviceCreateInfo deviceCreateInfo;
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.pNext = nullptr;
  deviceCreateInfo.flags = 0;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
  deviceCreateInfo.enabledLayerCount = 0;
  deviceCreateInfo.ppEnabledLayerNames = nullptr;
  deviceCreateInfo.enabledExtensionCount = 0;
  deviceCreateInfo.ppEnabledExtensionNames = nullptr;
  deviceCreateInfo.pEnabledFeatures = nullptr;

  RETURN_ON_VULKAN_ERROR(
      vkCreateDevice(physicalDevices[0], &deviceCreateInfo, 0, &device),
      "vkCreateDevice");

  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevices[0], &properties);

  for (uint32_t memoryTypeIndex = 0;
       memoryTypeIndex < properties.memoryTypeCount; ++memoryTypeIndex) {
    if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT &
         properties.memoryTypes[memoryTypeIndex].propertyFlags) &&
        (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT &
         properties.memoryTypes[memoryTypeIndex].propertyFlags) &&
        (memoryContext.memorySize <
         properties
             .memoryHeaps[properties.memoryTypes[memoryTypeIndex].heapIndex]
             .size)) {
      memoryContext.memoryTypeIndex = memoryTypeIndex;
      break;
    }
  }

  RETURN_ON_VULKAN_ERROR(memoryContext.memoryTypeIndex == VK_MAX_MEMORY_TYPES
                             ? VK_ERROR_OUT_OF_HOST_MEMORY
                             : VK_SUCCESS,
                         "memoryTypeIndex");
  return success();
}

static LogicalResult createShaderModule(const VkDevice &device,
                                        VkShaderModule &shaderModule) {
  size_t size = 0;
  SmallVector<uint32_t, 0> binary;
  uint32_t *shader =
      ReadFromFile(&size, "/home/khalikov/llvm-project/llvm/projects/mlir/"
                          "test/mlir-vulkan-runner/kernel.spv");
  if (!shader) {
    exit(0);
  }
  /*
  if (failed(spirv::serialize(module, binary))) {
    llvm::errs() << "can not serialize module" << '\n';
    return failure();
  }
  */

  VkShaderModuleCreateInfo shaderModuleCreateInfo;
  shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCreateInfo.pNext = nullptr;
  shaderModuleCreateInfo.flags = 0;
  shaderModuleCreateInfo.codeSize = size;
  shaderModuleCreateInfo.pCode = shader;
  RETURN_ON_VULKAN_ERROR(
      vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &shaderModule),
      "vkCreateShaderModule");
  return success();
}

static LogicalResult vulkanCreateDescriptorSetLayoutInfo(
    const VkDevice &device,
    std::vector<VkDescriptorSetLayoutBinding> &descriptorSetLayoutBindings,
    VkDescriptorSetLayout &descriptorSetLayout) {
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
  descriptorSetLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.pNext = nullptr;
  descriptorSetLayoutCreateInfo.flags = 0;
  descriptorSetLayoutCreateInfo.bindingCount =
      descriptorSetLayoutBindings.size();
  descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();
  RETURN_ON_VULKAN_ERROR(
      vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, 0,
                                  &descriptorSetLayout),
      "vkCreateDescriptorSetLayout");
  return success();
}

static LogicalResult
vulkanCreatePipelineLayout(const VkDevice &device,
                           const VkDescriptorSetLayout &descriptorSetLayout,
                           VkPipelineLayout &pipelineLayout) {
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.pNext = nullptr;
  pipelineLayoutCreateInfo.flags = 0;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
  pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
  pipelineLayoutCreateInfo.pPushConstantRanges = 0;
  RETURN_ON_VULKAN_ERROR(vkCreatePipelineLayout(device,
                                                &pipelineLayoutCreateInfo, 0,
                                                &pipelineLayout),
                         "vkCreatePipelineLayout");
  return success();
}

static LogicalResult
vulkanCreatePipeline(const VkDevice &device,
                     const VkPipelineLayout &pipelineLayout,
                     const VkShaderModule &shaderModule, VkPipeline &pipeline) {
  // TODO: actual kernel name
  const char *kernel_name = "compute_kernel";
  VkPipelineShaderStageCreateInfo stageInfo;
  stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageInfo.pNext = nullptr;
  stageInfo.flags = 0;
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = shaderModule;
  stageInfo.pName = kernel_name;
  stageInfo.pSpecializationInfo = 0;

  VkComputePipelineCreateInfo computePipelineCreateInfo;
  computePipelineCreateInfo.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.pNext = nullptr;
  computePipelineCreateInfo.flags = 0;
  computePipelineCreateInfo.stage = stageInfo;
  computePipelineCreateInfo.layout = pipelineLayout;
  computePipelineCreateInfo.basePipelineHandle = 0;
  computePipelineCreateInfo.basePipelineIndex = 0;
  RETURN_ON_VULKAN_ERROR(vkCreateComputePipelines(device, 0, 1,
                                                  &computePipelineCreateInfo, 0,
                                                  &pipeline),
                         "vkCreateComputePipelines");
  return success();
}

static LogicalResult
vulkanCreateDescriptorPool(const VkDevice &device, uint32_t queueFamilyIndex,
                           uint32_t descriptorCount,
                           VkCommandPoolCreateInfo &commandPoolCreateInfo,
                           VkDescriptorPool &descriptorPool) {
  commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCreateInfo.pNext = nullptr;
  commandPoolCreateInfo.flags = 0;
  commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
  VkDescriptorPoolSize descriptorPoolSize;
  descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptorPoolSize.descriptorCount = descriptorCount;
  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
  descriptorPoolCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolCreateInfo.pNext = nullptr;
  descriptorPoolCreateInfo.flags = 0;
  descriptorPoolCreateInfo.maxSets = 1;
  descriptorPoolCreateInfo.poolSizeCount = 1;
  descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
  RETURN_ON_VULKAN_ERROR(vkCreateDescriptorPool(device,
                                                &descriptorPoolCreateInfo, 0,
                                                &descriptorPool),
                         "vkCreateDescriptorPool");
  return success();
}

static LogicalResult vulkanAllocateDescriptorSets(
    const VkDevice &device, const VkDescriptorSetLayout &descriptorSetLayout,
    const VkDescriptorPool &descriptorPool, VkDescriptorSet &descriptorSet) {
  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
  descriptorSetAllocateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.pNext = nullptr;
  descriptorSetAllocateInfo.descriptorPool = descriptorPool;
  descriptorSetAllocateInfo.descriptorSetCount = 1;
  descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

  RETURN_ON_VULKAN_ERROR(vkAllocateDescriptorSets(device,
                                                  &descriptorSetAllocateInfo,
                                                  &descriptorSet),
                         "vkAllocateDescriptorSets");
  return success();
}

static LogicalResult vulkanCreateAndDispatchCommandBuffer(
    const VkDevice &device,
    const VkCommandPoolCreateInfo &commandPoolCreateInfo,
    const VkDescriptorSet &descriptorSet, const VkPipeline &pipeline,
    const VkPipelineLayout &pipelineLayout, VkCommandBuffer &commandBuffer) {
  VkCommandPool commandPool;
  RETURN_ON_VULKAN_ERROR(
      vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &commandPool),
      "vkCreateCommandPool");

  VkCommandBufferAllocateInfo commandBufferAllocateInfo;
  commandBufferAllocateInfo.sType =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  commandBufferAllocateInfo.pNext = nullptr;
  commandBufferAllocateInfo.commandPool = commandPool;
  commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandBufferAllocateInfo.commandBufferCount = 1;
  RETURN_ON_VULKAN_ERROR(vkAllocateCommandBuffers(device,
                                                  &commandBufferAllocateInfo,
                                                  &commandBuffer),
                         "vkAllocateCommandBuffers");

  VkCommandBufferBeginInfo commandBufferBeginInfo;
  commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  commandBufferBeginInfo.pNext = nullptr;
  commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  commandBufferBeginInfo.pInheritanceInfo = 0;

  RETURN_ON_VULKAN_ERROR(
      vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo),
      "vkBeginCommandBuffer");
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout, 0, 1, &descriptorSet, 0, 0);
  // Local global pool size.
  vkCmdDispatch(commandBuffer, 1, 1, 1);
  RETURN_ON_VULKAN_ERROR(vkEndCommandBuffer(commandBuffer),
                         "vkEndCommandBuffer");
  return success();
}

static LogicalResult
vulkanSubmitDeviceQueue(const VkDevice &device,
                        const VkCommandBuffer &commandBuffer,
                        uint32_t queueFamilyIndex) {
  VkQueue queue;
  vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
  VkSubmitInfo submitInfo;
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.pNext = nullptr;
  submitInfo.waitSemaphoreCount = 0;
  submitInfo.pWaitSemaphores = 0;
  submitInfo.pWaitDstStageMask = 0;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  submitInfo.signalSemaphoreCount = 0;
  submitInfo.pSignalSemaphores = nullptr;
  RETURN_ON_VULKAN_ERROR(vkQueueSubmit(queue, 1, &submitInfo, 0),
                         "vkQueueSubmit");
  RETURN_ON_VULKAN_ERROR(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
  return success();
}

static LogicalResult
checkResults(const VkDevice &device,
             const std::vector<VulkanDeviceMemoryBuffer> &memoryBuffers,
             std::unordered_map<Descriptor, VulkanBufferContent> &vars) {
  for (auto memBuf : memoryBuffers) {
    int32_t *payload;
    size_t size = vars[memBuf.descriptor].size;
    RETURN_ON_VULKAN_ERROR(
        vkMapMemory(device, memBuf.deviceMemory, 0, size, 0, (void **)&payload),
        "map memory");
    Print(payload, size);
  }
}

static LogicalResult vulkanCreateMemoryBuffers(
    const VkDevice &device,
    std::unordered_map<Descriptor, VulkanBufferContent> &vars,
    const VulkanMemoryContext &memoryContext,
    std::vector<VulkanDeviceMemoryBuffer> &memoryBuffers) {
  for (auto &var : vars) {
    VulkanDeviceMemoryBuffer memoryBuffer;
    createMemoryBuffer(device, var, memoryContext, memoryBuffer);
    memoryBuffers.push_back(memoryBuffer);
  }
  return success();
}

static void initDescriptorSetLayoutBindings(
    const std::vector<VulkanDeviceMemoryBuffer> &memBuffers,
    std::vector<VkDescriptorSetLayoutBinding> &descriptorSetLayoutBindings) {
  for (auto memBuffer : memBuffers) {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding;
    // Actual descriptor.
    descriptorSetLayoutBinding.binding = memBuffer.descriptor;
    descriptorSetLayoutBinding.descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding.descriptorCount = 1;
    descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    descriptorSetLayoutBinding.pImmutableSamplers = 0;
    descriptorSetLayoutBindings.push_back(descriptorSetLayoutBinding);
  }
}

static LogicalResult initMemoryContext(
    const std::unordered_map<Descriptor, VulkanBufferContent> &vars,
    VulkanMemoryContext &memoryContext) {
  memoryContext.memorySize = 0;
  for (auto var : vars) {
    if (var.second.size) {
      memoryContext.memorySize += var.second.size;
    } else {
      return failure();
    }
  }
  memoryContext.queueFamilyIndex = 0;
  memoryContext.memoryTypeIndex = VK_MAX_MEMORY_TYPES;
  return success();
}

static LogicalResult
processModule(spirv::ModuleOp module,
              std::unordered_map<Descriptor, VulkanBufferContent> &vars) {
  /*
  // TODO: deduce needed info from vars.
  for (auto &op : module.getBlock()) {
    if (isa<spirv::VariableOp>(op)) {
      processVariable(dyn_cast<spirv::VariableOp>(op));
    }
  }
  */

  VulkanMemoryContext memoryContext;
  initMemoryContext(vars, memoryContext);

  VkInstance instance;
  vulkanCreateInstance(instance);

  VkDevice device;
  vulkanCreateDevice(instance, memoryContext, device);

  std::vector<VulkanDeviceMemoryBuffer> memoryBuffers;
  vulkanCreateMemoryBuffers(device, vars, memoryContext, memoryBuffers);

  VkShaderModule shaderModule;
  createShaderModule(device, shaderModule);

  std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
  initDescriptorSetLayoutBindings(memoryBuffers, descriptorSetLayoutBindings);

  VkDescriptorSetLayout descriptorSetLayout;
  vulkanCreateDescriptorSetLayoutInfo(device, descriptorSetLayoutBindings,
                                      descriptorSetLayout);
  VkPipelineLayout pipelineLayout;
  vulkanCreatePipelineLayout(device, descriptorSetLayout, pipelineLayout);

  VkPipeline pipeline;
  vulkanCreatePipeline(device, pipelineLayout, shaderModule, pipeline);

  VkCommandPoolCreateInfo commandPoolCreateInfo;
  VkDescriptorPool descriptorPool;
  vulkanCreateDescriptorPool(device, memoryContext.queueFamilyIndex,
                             memoryBuffers.size(), commandPoolCreateInfo,
                             descriptorPool);

  VkDescriptorSet descriptorSet;
  vulkanAllocateDescriptorSets(device, descriptorSetLayout, descriptorPool,
                               descriptorSet);
  
  createDescriptorBufferInfoAndUpdateDesriptorSets(device, memoryBuffers,
                                                   descriptorSet);

  VkCommandBuffer commandBuffer;
  vulkanCreateAndDispatchCommandBuffer(device, commandPoolCreateInfo,
                                       descriptorSet, pipeline, pipelineLayout,
                                       commandBuffer);

  vulkanSubmitDeviceQueue(device, commandBuffer,
                          memoryContext.queueFamilyIndex);

  // TODO: Fix this.
  checkResults(device, memoryBuffers, vars);
  return success();
}

static LogicalResult
runOnModule(raw_ostream &os, ModuleOp module,
            std::unordered_map<Descriptor, VulkanBufferContent> &vars) {

  if (failed(module.verify())) {
    return failure();
  }

  auto result = failure();
  bool done = false;
  for (auto fn : module.getOps<FuncOp>()) {
    fn.walk<spirv::ModuleOp>([&](spirv::ModuleOp spirvModule) {
      if (done) {
        spirvModule.emitError("found more than one module");
      }
      done = true;
      processModule(spirvModule, vars);
    });
  }

  // out the result
  module.print(os);
  return success();
}

static void
PopulateData(std::unordered_map<Descriptor, VulkanBufferContent> &vars,
             int count) {
  for (int i = 0; i < count; ++i) {
    int *ptr = new int[4];
    for (int j = 0; j < 4; ++j) {
      ptr[j] = j;
    }
    vars.insert({i, {ptr, sizeof(int) * 4}});
  }
}

int main(int argc, char **argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR Vulkan execution driver\n");
  std::string errorMessage;
  auto inputFile = openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto outputFile = openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), SMLoc());
  std::unordered_map<Descriptor, VulkanBufferContent> variables;
  PopulateData(variables, 3);

  MLIRContext context;
  OwningModuleRef moduleRef(parseSourceFile(sourceMgr, &context));
  if (!moduleRef) {
    llvm::errs() << "can not open the file" << '\n';
    return 1;
  }

  if (failed(runOnModule(outputFile->os(), moduleRef.get(), variables))) {
    llvm::errs() << "can't run on module" << '\n';
    return 1;
  }

  return 0;
}
