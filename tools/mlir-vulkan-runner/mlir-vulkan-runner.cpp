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

using namespace mlir;
using namespace llvm;

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

static void *_alloca(size_t size) { return malloc(size); }

static VkResult vkGetBestComputeQueueNPH(VkPhysicalDevice physicalDevice,
                                         uint32_t *queueFamilyIndex) {
  uint32_t queueFamilyPropertiesCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice,
                                           &queueFamilyPropertiesCount, 0);

  VkQueueFamilyProperties *const queueFamilyProperties =
      (VkQueueFamilyProperties *)_alloca(sizeof(VkQueueFamilyProperties) *
                                         queueFamilyPropertiesCount);

  vkGetPhysicalDeviceQueueFamilyProperties(
      physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties);

  // first try and find a queue that has just the compute bit set
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
    // mask out the sparse binding bit that we aren't caring about (yet!) and
    // the transfer bit
    const VkQueueFlags maskedFlags =
        (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
         queueFamilyProperties[i].queueFlags);

    if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags) &&
        (VK_QUEUE_COMPUTE_BIT & maskedFlags)) {
      *queueFamilyIndex = i;
      return VK_SUCCESS;
    }
  }

  // lastly get any queue that'll work for us
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
    // mask out the sparse binding bit that we aren't caring about (yet!) and
    // the transfer bit
    const VkQueueFlags maskedFlags =
        (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
         queueFamilyProperties[i].queueFlags);

    if (VK_QUEUE_COMPUTE_BIT & maskedFlags) {
      *queueFamilyIndex = i;
      return VK_SUCCESS;
    }
  }

  return VK_ERROR_INITIALIZATION_FAILED;
}

static VkInstance vulkanCreateInstance() {
  const VkApplicationInfo applicationInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                             0,
                                             "VKComputeSample",
                                             0,
                                             "",
                                             0,
                                             VK_MAKE_VERSION(1, 0, 9)};

  const VkInstanceCreateInfo instanceCreateInfo = {
      VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      0,
      0,
      &applicationInfo,
      0,
      0,
      0,
      0};

  VkInstance instance;
  BAIL_ON_BAD_RESULT(vkCreateInstance(&instanceCreateInfo, 0, &instance));
  return instance;
}

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

static void processModule(spirv::ModuleOp module,
                          std::unordered_map<int, std::vector<int32_t>> &vars) {

  for (auto &op : module.getBlock()) {
    if (isa<spirv::VariableOp>(op)) {
      processVariable(dyn_cast<spirv::VariableOp>(op));
    }
  }

  const VkInstance instance = vulkanCreateInstance();
  uint32_t physicalDeviceCount = 0;
  BAIL_ON_BAD_RESULT(
      vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, 0));
  VkPhysicalDevice *const physicalDevices = (VkPhysicalDevice *)malloc(
      sizeof(VkPhysicalDevice) * physicalDeviceCount);
  BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
                                                physicalDevices));
  if (physicalDeviceCount) {
    uint32_t queueFamilyIndex = 0;
    BAIL_ON_BAD_RESULT(
        vkGetBestComputeQueueNPH(physicalDevices[0], &queueFamilyIndex));

    const float queuePrioritory = 1.0f;
    const VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        0,
        0,
        queueFamilyIndex,
        1,
        &queuePrioritory};

    const VkDeviceCreateInfo deviceCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        0,
        0,
        1,
        &deviceQueueCreateInfo,
        0,
        0,
        0,
        0,
        0};

    VkDevice device;
    BAIL_ON_BAD_RESULT(
        vkCreateDevice(physicalDevices[0], &deviceCreateInfo, 0, &device));

    VkPhysicalDeviceMemoryProperties properties;

    vkGetPhysicalDeviceMemoryProperties(physicalDevices[0], &properties);

    const size_t K = 128;
    const int32_t bufferLength = K;
    const uint32_t bufferSize = sizeof(int32_t) * bufferLength;

    // we are going to need two buffers from this one memory
    const VkDeviceSize memorySize = bufferSize;

    // set memoryTypeIndex to an invalid entry in the properties.memoryTypes
    // array
    uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

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

    const VkMemoryAllocateInfo memoryAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, memorySize, memoryTypeIndex};

    // Allocate the device memory.
    VkDeviceMemory memory1;
    BAIL_ON_BAD_RESULT(
        vkAllocateMemory(device, &memoryAllocateInfo, 0, &memory1));
    VkDeviceMemory memory2;
    BAIL_ON_BAD_RESULT(
        vkAllocateMemory(device, &memoryAllocateInfo, 0, &memory2));
    VkDeviceMemory memory3;
    BAIL_ON_BAD_RESULT(
        vkAllocateMemory(device, &memoryAllocateInfo, 0, &memory3));

    // Map the device memory to host memory
    int32_t *payload1, *payload2, *payload3;
    BAIL_ON_BAD_RESULT(
        vkMapMemory(device, memory1, 0, memorySize, 0, (void **)&payload1));
    BAIL_ON_BAD_RESULT(
        vkMapMemory(device, memory2, 0, memorySize, 0, (void **)&payload2));
    BAIL_ON_BAD_RESULT(
        vkMapMemory(device, memory3, 0, memorySize, 0, (void **)&payload3));

    // Init 2D tensors.
    for (int i = 0; i < K; ++i) {
      payload2[i] = 1;
      payload3[i] = 3;
      payload1[i] = 0;
    }
    vkUnmapMemory(device, memory1);
    vkUnmapMemory(device, memory2);
    vkUnmapMemory(device, memory3);

    const VkBufferCreateInfo bufferCreateInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        0,
        0,
        bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        1,
        &queueFamilyIndex};

    // Create buffers and bind them to the device memory.
    VkBuffer buffer1, buffer2, buffer3;
    BAIL_ON_BAD_RESULT(
        vkCreateBuffer(device, &bufferCreateInfo, 0, &buffer1));
    BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, buffer1, memory1, 0));
    BAIL_ON_BAD_RESULT(
        vkCreateBuffer(device, &bufferCreateInfo, 0, &buffer2));
    BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, buffer2, memory2, 0));
    BAIL_ON_BAD_RESULT(
        vkCreateBuffer(device, &bufferCreateInfo, 0, &buffer3));
    BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, buffer3, memory3, 0));

    // Read the shader from file.
    size_t size = 0;
    // Hardcoded path to binary shader.
    uint32_t *shader_ptr = nullptr;

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, size, shader_ptr};
    VkShaderModule shader_module;
    // Create Shader Module.
    BAIL_ON_BAD_RESULT(vkCreateShaderModule(device, &shaderModuleCreateInfo, 0,
                                            &shader_module));

    VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
         0},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
         0},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
         0}};

    // 3 buffers - 3 bindings
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 3,
        descriptorSetLayoutBindings};

    VkDescriptorSetLayout descriptorSetLayout;
    BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(
        device, &descriptorSetLayoutCreateInfo, 0, &descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        0,
        0,
        1,
        &descriptorSetLayout,
        0,
        0};

    VkPipelineLayout pipelineLayout;
    BAIL_ON_BAD_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                              0, &pipelineLayout));

    const char *kernel_name = "compute_kernel";

    VkComputePipelineCreateInfo computePipelineCreateInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        0,
        0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0,
         VK_SHADER_STAGE_COMPUTE_BIT, shader_module, kernel_name, 0},
        pipelineLayout,
        0,
        0};

    VkPipeline pipeline;

    BAIL_ON_BAD_RESULT(vkCreateComputePipelines(
        device, 0, 1, &computePipelineCreateInfo, 0, &pipeline));

    VkCommandPoolCreateInfo commandPoolCreateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, 0, queueFamilyIndex};

    VkDescriptorPoolSize descriptorPoolSize = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        0,
        0,
        1,
        1,
        &descriptorPoolSize};

    VkDescriptorPool descriptorPool;
    BAIL_ON_BAD_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo,
                                              0, &descriptorPool));

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, descriptorPool, 1,
        &descriptorSetLayout};

    VkDescriptorSet descriptorSet;
    BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(
        device, &descriptorSetAllocateInfo, &descriptorSet));

    VkDescriptorBufferInfo in1_descriptorBufferInfo = {buffer1, 0,
                                                       VK_WHOLE_SIZE};
    VkDescriptorBufferInfo in2_descriptorBufferInfo = {buffer2, 0,
                                                       VK_WHOLE_SIZE};
    VkDescriptorBufferInfo in3_descriptorBufferInfo = {buffer3, 0,
                                                       VK_WHOLE_SIZE};

    const int descriptors_count = 3;

    VkWriteDescriptorSet writeDescriptorSet[descriptors_count] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, descriptorSet, 0, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &in1_descriptorBufferInfo, 0},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, descriptorSet, 1, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &in2_descriptorBufferInfo, 0},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, descriptorSet, 2, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &in3_descriptorBufferInfo, 0}};

    vkUpdateDescriptorSets(device, descriptors_count, writeDescriptorSet, 0, 0);

    VkCommandPool commandPool;
    BAIL_ON_BAD_RESULT(
        vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &commandPool));

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, commandPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};

    VkCommandBuffer commandBuffer;
    BAIL_ON_BAD_RESULT(vkAllocateCommandBuffers(
        device, &commandBufferAllocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0};

    BAIL_ON_BAD_RESULT(
        vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &descriptorSet, 0, 0);

    vkCmdDispatch(commandBuffer, 1, 1, 1);

    BAIL_ON_BAD_RESULT(vkEndCommandBuffer(commandBuffer));

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    VkSubmitInfo submitInfo = {
        VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &commandBuffer, 0, 0};

    BAIL_ON_BAD_RESULT(vkQueueSubmit(queue, 1, &submitInfo, 0));

    BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue));

    // Check the result.
    BAIL_ON_BAD_RESULT(
        vkMapMemory(device, memory1, 0, memorySize, 0, (void **)&payload1));
    BAIL_ON_BAD_RESULT(
        vkMapMemory(device, memory2, 0, memorySize, 0, (void **)&payload2));
    BAIL_ON_BAD_RESULT(
        vkMapMemory(device, memory3, 0, memorySize, 0, (void **)&payload3));
  }
}

static LogicalResult
runOnModule(raw_ostream &os, ModuleOp module,
            std::unordered_map<int, std::vector<int32_t>> &vars) {

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

static void PopulateData(std::unordered_map<int, std::vector<int32_t>> &vars) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  for (int i = 0; i < 3; ++i) {
    vars.insert({i, move(data)});
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
  std::unordered_map<int, std::vector<int32_t>> variables;
  PopulateData(variables);

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
