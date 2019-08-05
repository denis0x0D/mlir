func @spirvmodule() -> () {
  spv.module "Logical" "VulkanKHR" {
    %0 = spv.Variable bind(0, 0) : !spv.ptr<f32, StorageBuffer>
    %1 = spv.Variable bind(0, 1) : !spv.ptr<!spv.array<12 x f32>, StorageBuffer>
    func @kernel_1(%arg0: !spv.ptr<f32, StorageBuffer>, %arg1: !spv.ptr<!spv.array<12 x f32>, StorageBuffer>) {
      spv.Return
    }
    spv.EntryPoint "GLCompute" @kernel_1, %0, %1 : !spv.ptr<f32, StorageBuffer>, !spv.ptr<!spv.array<12 x f32>, StorageBuffer>
  }
  func @kernel_1(%arg0: f32, %arg1: memref<12xf32, 1>)
  attributes  {gpu.kernel} {
    return
  }
  return
}
