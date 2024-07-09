
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






ON OPERATIONS AND OEPRATION WRAPPERS


Tranpose should not modify tensor on its own, but rather,

must be used like:

tensor = tensor.Tranpose(a1, a2)

With the idea of static polymorphism, we might let tranpose return some tempHolder object which can then be moved to Contrcutor of Tensor




Using CRTP, 

We may have TensorOperationHolder class which are returned by each operations

and TensorOperationHolder may have method of retrieving each index as per:
https://conradsanderson.id.au/pdfs/sanderson_curtin_armadillo_pasc_2017.pdf


Im not usre how further inheritence will happen, but the idea is to have TransposeOperationHolde, SumOperationHolder, MultiplyOperationHolder
which will all in all allow efficient uses?



OperationHolder {
  // or by CRTP ??
  virtual T getAt(idx {...})
}

TranposeHolder : OperatioHolder { 
  const inst axis1, axis2;
  const& tensor
  virtual T getAt(idx {...})
}

SumHolder : OperationHolder {
  const& tensor a, b;
  T getAt(idx {...})
}



Maybe follow idea of
Tensor(operaton holder) {
  // get all references as array first,
  // get all operation types
}


WITH CRTP

template<typename Derived>
class Base {
  const Derived& getRef() const {
    return static_cast<const Derived>(*this);
  }
}

Tensor : BAse<Tensor> {

}

OperationHolder : Base<OperationHolder> {

}






Maybe with CRTP we do not need dynamic polymorphism?



template<typename Derived>
class Base {
  const Derived& getRef() const {
    return static_cast<const Derived>(*this);
  }
}

Tensor : BAse<Tensor> {

}

TransposeOperationHolder : Base<OperationHolder> {
  const Tensor& primary;
  const int axis1, axis2.
}

SumOperationHolder : Base<OperationHolder> {
  const Tensor& primary;
  const Tensor& secondary;
}

MultOperationHolder : Base<OperationHolder> {
  const Tensor& primary;
  const Tensor& secondary;
}








!!!!!!

Brief intermediate summary:

Tensor Operation must be done efficently, minimizing movement of data.
It may be unlikely that tensor multiplication does not have in-place optimization, and Strassen algorithm may be prefered over other optimization tactics done thorugh CRTP. 
However, element-wise operation with broadcasting, such as summation, may benefit from static polymorphism and OperationHolder optimzation. Same may be said for Tranpose. 
However, implementing for summatuion only may necessitate that multiplication be also implemented. This later is at worst as bad as regular multiplcation (as we may simply perform multiplication within the operationHandler), but again obscures the operation which may lead to compiler failing to optimze our code.

Furthermore, we would idealy not want expose the OperationHolder, therefore require it to be private; however by the virtue of it facilitating public Tensor operations, it is doomed to be exposed and capturable. 

From a small test run, such design:
```
class A {
 private:
  struct B {
    int b;
  };

  int a_;
 public:
  A(int a) {
    a_ = a;
  }

  A(B b) {
    // std::cout << "Making from " << quote(b) << std::endl;
    a_ = b.b;
  }

  int get() {
    return a_;
  }

  B getB() {
    return B{2 * a_};
  }
};


A makeA(A a) {
  return a;
}
```
will allow opertaionHolder to be private while still allowing OperationHolder to be used inbwteen Tensor definition.

Or, we can use unnamed namesapce for similar effect.
```
namespace Anon {

namespace {

struct B {
  int b;
};

}

class A {
 private:
  int a_;
 public:
  A(int a) {
    a_ = a;
  }

  A(B b) {
    a_ = b.b;
  }

  B getB() {
    return B{2 * a_};
  }
};

}
```

However, we are stilll vulnerable to auto-catching. This may not be such a terrible issue, however. Under practical circumstances, we would never expect auto to be used to catch Tensor operation results.

However, we should call into question of above steps even being necessary. In wanting to reduce data-movement, we have sacrificed readablity. 
And not to mention, the correct and fast implementation of move operator may eleiminate this need completely. 
Only tranpose, which, with our current design of each tensor being a complete object (meaning no in-place tranposing), only tranpose require data moving. This may suggest above OperationHolder may only be needed for TranposeHolder. 

The above will require discussion with my partner to lay down the final decision. 









--- Iterator ---

On Tensor, we can easily do with address
- pointer to Tensor,
- current address

Iterator needs to handle +int, -int





Iterators divide into two cases
- Iterator
- ConstIterator

Each TensorLike will have ...::Iterator and ...::ConstIterator as inner class
Each TensorLike will have beign and end to return these
== Therefore TensorLike needs to have these as inner class
  = maybe each insatnces have separate inner which inherit from this TensorLike::Iterator,
  = which has virtual?

