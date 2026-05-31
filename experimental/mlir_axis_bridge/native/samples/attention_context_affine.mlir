module {
  func.func @attention_context(%prob: memref<1x2x4x4xf32>, %value: memref<1x2x4x8xf32>, %context: memref<1x2x4x8xf32>) {
    affine.for %b = 0 to 1 {
      affine.for %head = 0 to 2 {
        affine.for %q = 0 to 4 {
          affine.for %d = 0 to 8 {
            affine.for %k = 0 to 4 {
              %lhs = affine.load %prob[%b, %head, %q, %k] : memref<1x2x4x4xf32>
              %rhs = affine.load %value[%b, %head, %k, %d] : memref<1x2x4x8xf32>
              %product = arith.mulf %lhs, %rhs : f32
              affine.store %product, %context[%b, %head, %q, %d] : memref<1x2x4x8xf32>
            }
          }
        }
      }
    }
    return
  }
}
