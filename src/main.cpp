#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/Utils/utils.h"


std::vector<int> CutBroadcast(std::vector<int> idx, std::vector<int> shape) {
  std::vector<int> res(shape.size(), 0);
  for (int i = 1; i <= res.size(); ++i ){
    res[res.size() - i] = (shape[shape.size() - i] == 1) ? 0 : idx[idx.size() - i];
  }
  return res;
}

// TODO: method for computing broadcasted shapes and its prev chunk sizes
/*
 GET REF TO RES PREV AND RES CHUNK SIZE AS REF
 AND BRAODCAST SHAPE AND ORI SHAPE

 0. IF SHAPES EQUAL RETURN EMPTY
- reserve both res to broadcast length
 1. COMPUTE BROADCAST CHUNKSIZE AND CAP    O(B)
 2. for over each axis, in broadcasted shape,
  a. if axis beyond right-align, or is broadcasestd
    b. add to res (** NEED TO HANDLE axis 0 specially, because it will need to store CAPACITY not idx-1)    O(B)
- shrink to fit on them to remove reserved
 */


// need only broadcast dim and broadcast chuk at the broadcasted axis
// broadcast only found for axis that has corresponding real axis on original shape
void FindBroadcastAxes(const std::vector<int>& broadcast_dim,
                       const std::vector<int>& original_dim,
                       const std::vector<int>& broadcast_chunk,
                       std::vector<int>& res_broadcast_dim,
                       std::vector<int>& res_broadcast_chunk) {
  // assumes the res vectors are empty
  res_broadcast_dim.reserve(original_dim.size());
  res_broadcast_chunk.reserve(original_dim.size());

  int off_set = broadcast_dim.size() - original_dim.size();
  for (int i = 0; i < original_dim.size(); ++i) {
    if (broadcast_dim[off_set + i] != original_dim[i]) {
      // broadtcast at this axis
      res_broadcast_dim.push_back(broadcast_dim[off_set + i]);
      res_broadcast_chunk.push_back(broadcast_chunk[off_set + i]);
    }
  }

  res_broadcast_dim.shrink_to_fit();
  res_broadcast_chunk.shrink_to_fit();
  return;
}

// BROADCAST DIM AND CHUNK ARE DATA OF ONLY THE AXES THAT HAS BROADCAST APPLIED
size_t ComputeUnbroadcastAddress(const std::vector<int>& broadcast_dim,
                                 const std::vector<int>& broadcast_chunk,
                                 size_t original_capacity,
                                 size_t broadcast_address) {
  for (int axis = 0; axis < broadcast_dim.size(); ++axis) {
    // see how many times we have repeated the chunksize
    const size_t repetition = broadcast_address / broadcast_chunk[axis];
    // every final index in dimension gets a 'pass'
    const size_t repetition_off_setter = repetition / broadcast_dim[axis]; 

    // need to remove address by this count
    const size_t reset_count = repetition - repetition_off_setter;
    broadcast_address -= broadcast_chunk[axis] * reset_count;
  }
  // Underset, ensure is in bound
  //   needed because broadcast does not consider front-broadcasting
  broadcast_address %= original_capacity;

  return broadcast_address;
}

int main() {
  using namespace cpp_nn::util;

  // Tensor A = Tensor({2, 3, 4}, 0);
  // int val = 0; 
  // auto ait = A.begin();
  // auto aend = A.end();

  // while (ait != aend) {
  //   *ait = ++val;
  //   ++ait;
  // }

  // PrintTensor(A);

  // // std::cout << "This is what it shoudl look like" << std::endl;
  // // Tensor C = A.Transpose(0, 1).Reshape({3, 2, 4}).Transpose(0, 2);

  // std::cout << "On multiple lines" << std::endl;
  // Tensor C = A.Transpose(0, 1).Reshape({3, 2, 4});
  // C = C.Transpose(0, 2);
  // PrintTensor(C);

  // std::cout << "On single lines" << std::endl;
  // C = A.Transpose(0, 1).Reshape({3, 2, 4}).Transpose(0, 2);
  // PrintTensor(C);


  // std::cout << "without reshape in the middle" << std::endl;
  // C = A.Transpose(0, 1).Transpose(0, 2);
  // PrintTensor(C);

  // // std::cout << "With multi" << std::endl;
  // // Tensor D = A.Transpose(0, 1).Transpose(0, 2);
  // // PrintTensor(D);


  {
    std::cout << "Broadcast iteator order testground" << std::endl;

    // multiple broadcast in the middle in a row
    std::vector<int> broad({2, 3, 2, 2, 2, 5, 1});
    std::vector<int> small({2, 1, 2, 1, 2, 5, 1});

    // // multiple broadcast in the middle in a row
    // std::vector<int> broad({2, 3, 2, 2, 1});
    // std::vector<int> small({2, 1, 1, 2, 1});

    // // broadcasting and front-broadcasting
    // std::vector<int> broad({2, 3, 2, 2, 2});
    // std::vector<int> small(   {3, 2, 1, 2});

    // // broadcasting and  multiple front-broadcasting
    // std::vector<int> broad({2, 3, 2, 3, 2, 2, 2});
    // std::vector<int> small(         {3, 2, 1, 2});

    // // onyl front broadcasting
    // std::vector<int> broad({2, 2, 3, 2, 3, 2, 2, 2, 7});
    // std::vector<int> small(                        {7});

    // // all broadcasting
    // std::vector<int> broad({2, 2, 3, 2, 3, 2, 2, 2, 7});
    // std::vector<int> small(                        {1});

    // // single order broadcasting
    // std::vector<int> broad({4});
    // std::vector<int> small({1});

    // // multiple all order broadcasting
    // std::vector<int> broad({4, 3, 4});
    // std::vector<int> small({1, 1, 1});

    // // no broadcasting
    // std::vector<int> broad({2, 3, 2});
    // std::vector<int> small({2, 3, 2});

    std::vector<int> broad_chunk(broad.size(), 1);
    std::vector<int> small_chunk(small.size(), 1);

    size_t cap = 0;
    size_t cap2 = 0;
    ComputeCapacityAndChunkSizes(broad, broad_chunk, cap);
    ComputeCapacityAndChunkSizes(small, small_chunk, cap2);

    std::cout << "Chunk size are ";
    for (auto i : broad_chunk) {
      std::cout << i << ", ";
    }
    std::cout << std::endl << std::endl;
    std::cout << "broad size are ";
    for (auto i : broad) {
      std::cout << i << ", ";
    }
    std::cout << std::endl << std::endl;


    std::vector<int> idx(broad.size(), 0);


    // TO BE USED WHEN RECOMPUTING ADDRESS
    std::vector<int> res_broadcast_dim;
    std::vector<int> res_broadcast_chunk;

    FindBroadcastAxes(broad,
                      small, 
                      broad_chunk,
                      res_broadcast_dim,res_broadcast_chunk);


    /**
     * TRY AGAIN MAKE THINGS EASIER TO READ
     */
    {
      std::cout << "Trial ground 2" << std::endl;

      do {
        size_t broadcast_address = IndicesToAddress(broad, broad_chunk, idx);

        auto cut_idx = CutBroadcast(idx, small);
        size_t cut_address = IndicesToAddress(small, small_chunk, cut_idx);

        int diff = static_cast<int>(cut_address) - static_cast<int>(broadcast_address);

        std::cout << broadcast_address << "\t->\t" << cut_address;
        std::cout << "\t diff is \t" << diff;


        int recomp = ComputeUnbroadcastAddress(res_broadcast_dim, res_broadcast_chunk, cap2, broadcast_address);

        std::cout << "\t\t\t fin is " << recomp;
        std::cout << ((recomp == cut_address) ? " Corr" : " WRONG");

        if (recomp != cut_address)
         return -1;

        std::cout << std::endl;

      } while (IncrementIndicesByShape(broad.begin(), broad.end(), idx.begin(), idx.end()));
    }
    /**
     * TODO:!!!!
     *  COMMENT, for broadcast iterator, if we keep running broadcast-iterator incremented count, that is, instead of having a vector iterator to compute address
     *  we simply sum increment and decrement, we can compute original-shape address in simply O(broadcasted_axes_count).
     */
  }
  
}