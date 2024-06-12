/**
 * Tensor, as more felxible method of working with Matrix-like objects.
 * 
 * Tensor :: class containing TensorElement
 *            All other TensorLike are to inherit from this class
 * 
 * TensorElement :: private struct or a 'lighter weight' object that contains contents
 *                    of tensor only.
 *                  All operations are done ONTO TensorElement through its wrapper Tensor class.
 *                  Intention of this is for other 'accessor' or 'reference objects' to simply point to 
 *                    a given TensorElement without needing to reallocate potentially massive data.
 *                  ** CAREFUL: This of cource may lead to ownership issue. ie) incorrect destruction time
 *                    This may be resolved at object initialization by a simple flag of ownership
 * 
 * Referencer :: We may want to use this to reference smallest subtensor of TensorElement
 *                ie) only first columns, or [:][2:] like with numpy
 *              This should be very carefully implemented especailly due to ownership.
 *              We may like to implement 'export' function which returns separate Tensor
 *                that copies over the selected subtensor
 * 
 * Matrix :: May or maynot need to be derived from Tensor
 *            Most operations and maths will be defined in terms of Matrix, but 
 *              would ideally allow tensors to be valid input to these matrices.
 *            Asserts that order is 2
 * 
 * Vector :: Similar to above, may or may not derive from Matrix
 *            Asserts order is 1           
 *            However, when computed with matrix, it is treated as nx1 matrix
 * 
 * 
 * Notes on Operation:
 * Matrix Multiplication is ideally implemted on tensor level
 * - Tensor will perform internal checks on order and decide multicability.
 * - - [2, 3, 4] * [4, 5] is to be understood as 'array of 2 matrices mutlplied each to latter matrix"
 * - -   so result will be [2, 3, 5]
 * - -
 * - - [2, 4] * [4, 4, 5] likewise
 * - -
 * - - [4, 2, 3] * [2, 3, 4] may be understood as computing every combination of matrices
 * - - will result in [4, 2, 2, 4], where the outer [4,2] specifies which combination of matrices are
 * - -   chosen to result in [2,4] matrix
 * - - 
 * - Vectors are naturally of order 1,
 * - - [2, 4] * [4] are understood as 4d vector input to [2x4] matrix
 * - -   in effect [4] is treated then as [4, 1],
 * - -   Such may be called 'internal reshaping'
 * - 
 * - Covectors are then treated as [1, 4]
 * - 
 * - Special flags may be added to multiplication to indicate that certain inputs are to be treated as
 * -   collection of vectors or matrices
 * - - For example, [4, 2] * [4, 2] may be flagged with 'asVectors' to indicate the latter is to be treated as 
 * - -   4-array of 2d vectors
 * - - As such result is to be [4, 4] a 4 array of 4d vector
 * - - This can be explicitly forced behavior by reshaping latter as [4, 2, 1]
 * 
 * As above suggests, we mainly focus on matrix mult only 
 *   with tensor serving as container for such matrices. 
 * Perhaps inplication of this is that Matrices should remain lowest object.?
 * 
 * TODO more multiplcation behaviours should be consdiered. 
 *   else throw dimension error?
 * So far only up to 3-orders are allowed in such.
 */