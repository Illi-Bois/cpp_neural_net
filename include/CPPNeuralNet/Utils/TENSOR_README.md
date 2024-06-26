
Tensor Reimplement Idea.

As tensor is the core of this project, correct and efficient implementation of it is vital. As it is defined now, it is unorganized collection of ideas that are forcibly out together. 

We will ideate and orgaize Tensor's structure and behaviours here, then by referencing it, implement Tensor in easy to read and use way. 


Tensor : Multi-dimensional array. It also serves a 'generalization' of matrices, in the sense that 2order tensor represent a matrix, and 1 order tensor represent vector. 

Interpretation: given tebnsor of dimension [dim... r, c], it can be read as dim-dimensional multiarray of r x c matrices.
[r, c] is 0dim array of matrix, or simply just a matrix.
[d ] is a vectror of d-dim, but for practrical purposes VECTORS SHOULD PREFERABLY REPRESENTED AS [d x 1] matrix


TENSOR SIGNITURE MEHTODS:
- Constructors 
  - (init_list, init_val)
  - (vector, init_val)
  - copy(other&)
  - move(other&&)

- Getters
  - getDimension(axis)
  - getOrder()
  - & getElement(vector indicies) / const
  - & getElementByAddress(int addres) / const
    ??: MAYBE THIS SHOULD NOT BE EXPOSED AS SO

- Modifiers
  - Transpose(axis1, axis2)
  - Reshape(vector newDim)


- Operations
- Scalar Operations
  - +(T val) 
    : addes val to all
  - *(T val)
    : mult val to all
- Tensor Operations
  - opeeration+(Tensor& other)
  - operation*(Tensor& other)


- Reduction and Concatonation
  : Will need to think these ones through more...





ON OPERATIONS AND TEMPLATE METAPROGRAMMING
- following advice from 
https://conradsanderson.id.au/misc/sanderson_templates_lecture_uqcomp7305.pdf
it may be vauable to consider implementing OperatorAssistant struct for Tensor Operations
- This of course will require us to make additional constructors and operators 

IE)
Glue
|- SumGlue
|- ProdGlue
|- TransposeGlue

where glue has virtual function or member of 'apply' 
and Tensor can move= from these Glue

The glues may need to be either protected or private.