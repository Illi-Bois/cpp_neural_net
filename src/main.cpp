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

    std::vector<int> broad({3, 2, 2, 3});
    std::vector<int> small({1, 2, 1, 1});

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


    std::vector<int> idx(broad.size(), 0);


    // TO BE USED WHEN RECOMPUTING ADDRESS
    std::vector<int> broad_cast_dim;
    std::vector<int> broad_cast_dim_prev;

    int IDX = 0; // over broad
    int diff_dim = broad.size() - small.size();

    if (diff_dim) {
      // need to fill up from front 
        std::cout << "BORADCAST AT " << IDX << std::endl;
      broad_cast_dim.push_back(broad_chunk[0]);
      broad_cast_dim_prev.push_back(cap);

      --diff_dim;
      ++IDX;

      while (diff_dim)
      {
        std::cout << "BORADCAST AT " << IDX << std::endl;
        broad_cast_dim.push_back(broad_chunk[IDX]);
        broad_cast_dim_prev.push_back(broad_chunk[IDX - 1]);
        --diff_dim;
      }
      
    }

    if (IDX == 0) {
      if (broad[0] != small[0]) {
        // broadcasted at first dim
        std::cout << "BORADCAST AT " << IDX << std::endl;
        broad_cast_dim.push_back(broad_chunk[0]);
        broad_cast_dim_prev.push_back(cap);
        ++IDX;
      }
    }

    //!  we actually need that difference
    diff_dim = broad.size() - small.size();

    while (IDX < broad.size()) {
      if (broad[IDX] != small[IDX - diff_dim]) {
        // broadcasted at first dim
        std::cout << "BORADCAST AT " << IDX << std::endl;
        broad_cast_dim.push_back(broad_chunk[IDX]);
        broad_cast_dim_prev.push_back(broad_chunk[IDX - 1]);
      }
      ++IDX;
    }

    // THIS IS HOW YOU COMPUTE ALL THE BROADCAST DIM AND ITS PREV
    for (int i = 0; i < broad_cast_dim.size(); ++i) {
      std::cout << broad_cast_dim[i] << ", " << broad_cast_dim_prev[i] << "\t";
    }
    std::cout << std::endl;



    do {
      // for (auto i : idx) {
      //   std::cout << i << ",\t";
      // }
      size_t al;
      std::cout << ":: " << (al = IndicesToAddress(broad, broad_chunk, idx) ) << "\t";

      auto cut = CutBroadcast(idx, small);
      size_t OTHER;
      std::cout << " AND " << (OTHER = IndicesToAddress(small, small_chunk, cut)) << "\t::";
      // for (auto i : cut) {
      //   std::cout << i << ",\t";
      // }

      size_t res = 0;
      size_t comp = al;

      // same thing down to lower broadcast sizes do this but upwards?
      // comp %= broad_chunk[0];

      // if (comp / broad_chunk[2] != broad[2]) {
      //   res += broad_chunk[2];
      // }
      // comp %= broad_chunk[2];




      // FOR EACH BROADCAST SHAPE,
      /*
        starting from top
        if broadcast occured at i, if divisible by chunk[i - 1], add chunk[i]
        irrgardless, address /= chunk[i]

        repeat for all broadcast dim

        at the end, add remaining from address
      */
      // if (comp / cap) {
      //   res += broad_chunk[0];
      // }
      // comp %= broad_chunk[0];
      // // when it is divisible by 24, add 8
      // if (comp / broad_chunk[1]) {
      //   res += broad_chunk[2];
      // }
      // comp %= broad_chunk[2];

      // FOREACH BROADCAST DIMS
      for (int i = 0; i < broad_cast_dim.size(); ++i) {
        if (comp / broad_cast_dim_prev[i]) {
          res += broad_cast_dim[i];
        }
        comp %= broad_cast_dim[i];
      }
      res += comp;
      
      std::cout << " ummm " << (res) << " DIFF " << (static_cast<int>(OTHER) - static_cast<int>(res))  << ((OTHER != res) ? " WRONG" : " " )<< std::endl;

      std::cout << std::endl;

    } while (IncrementIndicesByShape(broad.begin(), broad.end(), idx.begin(), idx.end()));

    /**
     * TODO:!!!!
     *  COMMENT, for broadcast iterator, if we keep running broadcast-iterator incremented count, that is, instead of having a vector iterator to compute address
     *  we simply sum increment and decrement, we can compute original-shape address in simply O(broadcasted_axes_count).
     */
  }
  
}