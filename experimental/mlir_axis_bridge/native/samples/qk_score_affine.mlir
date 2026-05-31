module {
  func.func @qk_score(%query: memref<1x2x4x8xf32>, %key: memref<1x2x8x4xf32>, %score: memref<1x2x4x4xf32>) {
    affine.for %b = 0 to 1 {
      affine.for %head = 0 to 2 {
        affine.for %q = 0 to 4 {
          affine.for %k = 0 to 4 {
            affine.for %d = 0 to 8 {
              %lhs = affine.load %query[%b, %head, %q, %d] : memref<1x2x4x8xf32>
              %rhs = affine.load %key[%b, %head, %d, %k] : memref<1x2x8x4xf32>
              %value = arith.mulf %lhs, %rhs : f32
              affine.store %value, %score[%b, %head, %q, %k] : memref<1x2x4x4xf32>
            }
          }
        }
      }
    }
    return
  }
}
