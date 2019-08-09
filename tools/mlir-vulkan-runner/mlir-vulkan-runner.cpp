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

#define BAIL_ON_BAD_RESULT(result)                                             \
  if (VK_SUCCESS != (result)) {                                                \
    fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__);                 \
    fprintf(stderr, "Exit code %d\n", result);                                 \
    exit(-1);                                                                  \
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
  // Pointer to the host memory
  void *ptr;
  // Size in bytes
  int64_t size;
};

static VkResult vkGetBestComputeQueueNPH(const VkPhysicalDevice &physicalDevice,
                                         uint32_t *queueFamilyIndex) {
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
      *queueFamilyIndex = i;
      return VK_SUCCESS;
    }
  }

  // Try to find other queue.
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
    const VkQueueFlags maskedFlags =
        (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
         queueFamilyProperties[i].queueFlags);

    if (VK_QUEUE_COMPUTE_BIT & maskedFlags) {
      *queueFamilyIndex = i;
      return VK_SUCCESS;
    }
  }
  // TODO: must other error;
  return VK_ERROR_INITIALIZATION_FAILED;
}

static VkInstance vulkanCreateInstance() {
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

  VkInstance instance;
  BAIL_ON_BAD_RESULT(vkCreateInstance(&instanceCreateInfo, 0, &instance));
  return instance;
}

static size_t
getMemorySize(std::unordered_map<Descriptor, VulkanBufferContent> &vars) {
  size_t count = 0;
  for (auto var : vars) {
    // TODO: Here should be a type.
    count += var.second.size;
  }
  return count;
}

static VulkanDeviceMemoryBuffer
createMemoryBuffer(const VkDevice &device,
                   std::pair<Descriptor, VulkanBufferContent> var,
                   uint32_t memoryTypeIndex, uint32_t queueFamilyIndex) {
  VulkanDeviceMemoryBuffer memoryBuffer;
  memoryBuffer.descriptor = var.first;
  // TODO: Check that the size is not 0, because it will fail.
  const int64_t bufferSize = var.second.size;

  VkMemoryAllocateInfo memoryAllocateInfo;
  memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memoryAllocateInfo.pNext = nullptr;
  memoryAllocateInfo.allocationSize = bufferSize;
  memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;
  // Allocate the device memory.
  BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, 0,
                                      &memoryBuffer.deviceMemory));

  void *payload;
  BAIL_ON_BAD_RESULT(vkMapMemory(device, memoryBuffer.deviceMemory, 0,
                                 bufferSize, 0, (void **)&payload));

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
  bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;
  BAIL_ON_BAD_RESULT(
      vkCreateBuffer(device, &bufferCreateInfo, 0, &memoryBuffer.buffer));
  BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, memoryBuffer.buffer,
                                        memoryBuffer.deviceMemory, 0));
  return memoryBuffer;
}

static void createDescriptorBufferInfoAndUpdateDesriptorSet(
    const VkDevice &device, VulkanDeviceMemoryBuffer &memoryBuffer,
    VkDescriptorSet &descriptorSet) {
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

static VkDevice vulkanCreateDevice(const VkInstance &instance,
                                   uint32_t &memoryTypeIndex,
                                   uint32_t &queueFamilyIndex,
                                   const VkDeviceSize memorySize) {
  uint32_t physicalDeviceCount = 0;
  VkDevice device;
  BAIL_ON_BAD_RESULT(
      vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, 0));
  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);

  BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
                                                physicalDevices.data()));
  // TODO: return error;
  if (physicalDeviceCount) {
    queueFamilyIndex = 0;
    BAIL_ON_BAD_RESULT(
        vkGetBestComputeQueueNPH(physicalDevices[0], &queueFamilyIndex));

    const float queuePrioritory = 1.0f;
    VkDeviceQueueCreateInfo deviceQueueCreateInfo;
    deviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    deviceQueueCreateInfo.pNext = nullptr;
    deviceQueueCreateInfo.flags = 0;
    deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;
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

    BAIL_ON_BAD_RESULT(
        vkCreateDevice(physicalDevices[0], &deviceCreateInfo, 0, &device));
    VkPhysicalDeviceMemoryProperties properties;
    // TODO: better way to take device.
    vkGetPhysicalDeviceMemoryProperties(physicalDevices[0], &properties);
    memoryTypeIndex = VK_MAX_MEMORY_TYPES;

    // Find valid memory types.
    // TODO: Update it be indexing.
    for (uint32_t k = 0; k < properties.memoryTypeCount; k++) {
      if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT &
           properties.memoryTypes[k].propertyFlags) &&
          (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT &
           properties.memoryTypes[k].propertyFlags) &&
          (memorySize <
           properties.memoryHeaps[properties.memoryTypes[k].heapIndex].size)) {
        memoryTypeIndex = k;
        break;
      }
    }

    BAIL_ON_BAD_RESULT(memoryTypeIndex == VK_MAX_MEMORY_TYPES
                           ? VK_ERROR_OUT_OF_HOST_MEMORY
                           : VK_SUCCESS);
  }
  return device;
}

static void Print(int32_t *result, int size) {
  std::cout << "buffer started with size" << size << std::endl;
  for (int i = 0; i < size; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << "buffer ended" << std::endl;
}

static VkDescriptorSetLayoutBinding
createDescriptorSetLayoutBinding(Descriptor descriptor) {
  VkDescriptorSetLayoutBinding descriptorSetLayoutBindings;
  descriptorSetLayoutBindings.binding = descriptor;
  descriptorSetLayoutBindings.descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptorSetLayoutBindings.descriptorCount = 1;
  descriptorSetLayoutBindings.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  descriptorSetLayoutBindings.pImmutableSamplers = 0;
  return descriptorSetLayoutBindings;
}

static VkShaderModule createShaderModule(const VkDevice &device) {
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

  uint64_t codeSize = size;
  VkShaderModuleCreateInfo shaderModuleCreateInfo;
  shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCreateInfo.pNext = nullptr;
  shaderModuleCreateInfo.flags = 0;
  shaderModuleCreateInfo.codeSize = codeSize;
  shaderModuleCreateInfo.pCode = shader;
  VkShaderModule shader_module;
  BAIL_ON_BAD_RESULT(
      vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &shader_module));
  // TODO: return LogicalResult.
  return shader_module;
}

static VkDescriptorSetLayout vulkanCreateDescriptorSetLayoutInfo(
    const VkDevice &device,
    std::vector<VkDescriptorSetLayoutBinding> &descriptorSetLayoutBindings) {
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
  descriptorSetLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.pNext = nullptr;
  descriptorSetLayoutCreateInfo.flags = 0;
  descriptorSetLayoutCreateInfo.bindingCount =
      descriptorSetLayoutBindings.size();
  descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();
  VkDescriptorSetLayout descriptorSetLayout;
  BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(
      device, &descriptorSetLayoutCreateInfo, 0, &descriptorSetLayout));
  return descriptorSetLayout;
}

static VkPipelineLayout
vulkanCreatePipelineLayout(const VkDevice &device,
                           VkDescriptorSetLayout &descriptorSetLayout) {
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.pNext = nullptr;
  pipelineLayoutCreateInfo.flags = 0;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
  pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
  pipelineLayoutCreateInfo.pPushConstantRanges = 0;
  VkPipelineLayout pipelineLayout;
  BAIL_ON_BAD_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                            0, &pipelineLayout));
  return pipelineLayout;
}

static VkPipeline vulkanCreatePipeline(const VkDevice &device,
                                       VkPipelineLayout &pipelineLayout,
                                       VkShaderModule &shaderModule) {
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
  VkPipeline pipeline;
  BAIL_ON_BAD_RESULT(vkCreateComputePipelines(
      device, 0, 1, &computePipelineCreateInfo, 0, &pipeline));
  return pipeline;
}

static VkDescriptorPool
vulkanCreateDescriptorPool(const VkDevice &device, uint32_t queueFamilyIndex,
                           uint32_t descriptorCount,
                           VkCommandPoolCreateInfo &commandPoolCreateInfo) {
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
  VkDescriptorPool descriptorPool;
  BAIL_ON_BAD_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo,
                                            0, &descriptorPool));
  return descriptorPool;
}

static VkDescriptorSet
vulkanAllocateDescriptorSets(const VkDevice &device,
                             const VkDescriptorSetLayout &descriptorSetLayout,
                             const VkDescriptorPool &descriptorPool) {
  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
  descriptorSetAllocateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.pNext = nullptr;
  descriptorSetAllocateInfo.descriptorPool = descriptorPool;
  descriptorSetAllocateInfo.descriptorSetCount = 1;
  descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

  VkDescriptorSet descriptorSet;
  BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(
      device, &descriptorSetAllocateInfo, &descriptorSet));
  return descriptorSet;
}

static VkCommandBuffer vulkanCreateAndDispatchCommandBuffer(
    const VkDevice &device,
    const VkCommandPoolCreateInfo &commandPoolCreateInfo,
    const VkDescriptorSet &descriptorSet, const VkPipeline &pipeline,
    const VkPipelineLayout &pipelineLayout) {
  VkCommandPool commandPool;
  BAIL_ON_BAD_RESULT(
      vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &commandPool));
  VkCommandBufferAllocateInfo commandBufferAllocateInfo;
  commandBufferAllocateInfo.sType =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  commandBufferAllocateInfo.pNext = nullptr;
  commandBufferAllocateInfo.commandPool = commandPool;
  commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandBufferAllocateInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  BAIL_ON_BAD_RESULT(vkAllocateCommandBuffers(
      device, &commandBufferAllocateInfo, &commandBuffer));

  VkCommandBufferBeginInfo commandBufferBeginInfo;
  commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  commandBufferBeginInfo.pNext = nullptr;
  commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  commandBufferBeginInfo.pInheritanceInfo = 0;

  BAIL_ON_BAD_RESULT(
      vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout, 0, 1, &descriptorSet, 0, 0);
  // Local global pool size.
  vkCmdDispatch(commandBuffer, 1, 1, 1);
  BAIL_ON_BAD_RESULT(vkEndCommandBuffer(commandBuffer));
  return commandBuffer;
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
  BAIL_ON_BAD_RESULT(vkQueueSubmit(queue, 1, &submitInfo, 0));
  BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue));
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

  uint32_t memoryTypeIndex, queueFamilyIndex;
  auto instance = vulkanCreateInstance();
  const VkDeviceSize memorySize = getMemorySize(vars);
  if (!memorySize) {
    // TODO: Update for better formating.
    return failure();
  }
  // TODO: return type is LogicalResult
  auto device = vulkanCreateDevice(instance, memoryTypeIndex, queueFamilyIndex,
                                   memorySize);
  // TODO: Refactor to single function.
  std::vector<VulkanDeviceMemoryBuffer> memoryBuffers;
  for (auto &var : vars) {
    auto memoryBuffer =
        createMemoryBuffer(device, var, memoryTypeIndex, queueFamilyIndex);
    memoryBuffers.push_back(memoryBuffer);
  }
  auto shader_module = createShaderModule(device);
  std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
  for (auto var : vars) {
    descriptorSetLayoutBindings.push_back(
        createDescriptorSetLayoutBinding(var.first));
  }
  auto descriptorSetLayout =
      vulkanCreateDescriptorSetLayoutInfo(device, descriptorSetLayoutBindings);
  auto pipelineLayout = vulkanCreatePipelineLayout(device, descriptorSetLayout);
  auto pipeline = vulkanCreatePipeline(device, pipelineLayout, shader_module);
  VkCommandPoolCreateInfo commandPoolCreateInfo;
  auto descriptorPool = vulkanCreateDescriptorPool(
      device, queueFamilyIndex, memoryBuffers.size(), commandPoolCreateInfo);
  auto descriptorSet =
      vulkanAllocateDescriptorSets(device, descriptorSetLayout, descriptorPool);

  // TODO: Move to function.
  for (auto memoryBuffer : memoryBuffers) {
    createDescriptorBufferInfoAndUpdateDesriptorSet(device, memoryBuffer,
                                                    descriptorSet);
  }

  auto commandBuffer = vulkanCreateAndDispatchCommandBuffer(
      device, commandPoolCreateInfo, descriptorSet, pipeline, pipelineLayout);

  vulkanSubmitDeviceQueue(device, commandBuffer);

  for (auto memBuf : memoryBuffers) {
    int32_t *payload;
    size_t size = vars[memBuf.descriptor].size;
    BAIL_ON_BAD_RESULT(vkMapMemory(device, memBuf.deviceMemory, 0, size, 0,
                                   (void **)&payload));
    Print(payload, size);
  }
  std::cout << "End of pipeline" << std::endl;
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
