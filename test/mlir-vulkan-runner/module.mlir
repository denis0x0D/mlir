func @spirvmodule() -> () {
  spv.module "Logical" "VulkanKHR" {
    %0 = spv.Variable bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<4 x f32>>, StorageBuffer>
    func @kernel_1(%arg0: !spv.ptr<!spv.struct<!spv.array<4 x f32>>, StorageBuffer>) {
      %c0 = spv.constant 0: i32
      %c1 = spv.constant 1: i32
      %1 = spv.AccessChain %arg0[%c0, %c1] : !spv.ptr<!spv.struct<!spv.array<4 x f32>>, StorageBuffer>
      %2 = spv.Load "StorageBuffer" %1 : f32
      %3 = spv.FAdd %2, %2 : f32
      spv.Store "StorageBuffer" %1, %3 : f32
      spv.Return
    }
    spv.EntryPoint "GLCompute" @kernel_1, %0 : !spv.ptr<!spv.struct<!spv.array<4 x f32>>, StorageBuffer>
  }
  return
}
