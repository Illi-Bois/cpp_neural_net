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

    std::vector<int> broad({2, 3, 3, 2});
    std::vector<int> small(   {3, 1, 2});

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
    std::vector<int> broad_cast_dim_count;
    std::vector<int> broad_cast_dim_prev;
    std::vector<int> add_dim;

    int IDX = 0; // over broad
    int diff_dim = broad.size() - small.size();

    if (diff_dim) {
      // need to fill up from front 
        std::cout << "BORADCAST AT " << IDX << std::endl;
      broad_cast_dim.push_back(broad_chunk[0]);
      broad_cast_dim_count.push_back(broad[0]);
      broad_cast_dim_prev.push_back(cap);
      add_dim.push_back(0); // for no match, add 0??
      // add_dim.push_back(cap2); // for no match, or cap2
      

      --diff_dim;
      ++IDX;

      while (diff_dim)
      {
        std::cout << "BORADCAST AT " << IDX << std::endl;
        broad_cast_dim.push_back(broad_chunk[IDX]);
        broad_cast_dim_count.push_back(broad[IDX]);
        broad_cast_dim_prev.push_back(broad_chunk[IDX - 1]);
        add_dim.push_back(0); // for no match, add 0??
        // add_dim.push_back(cap2); // for no match, or cap2
        --diff_dim;
      }
      
    }

    if (IDX == 0) {
      if (broad[0] != small[0]) {
        // broadcasted at first dim
        std::cout << "BORADCAST AT " << IDX << std::endl;
        broad_cast_dim.push_back(broad_chunk[0]);
        broad_cast_dim_count.push_back(broad[0]);
        broad_cast_dim_prev.push_back(cap);
        add_dim.push_back(cap2); // for no match, add 0??

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
        broad_cast_dim_count.push_back(broad[IDX]);
        broad_cast_dim_prev.push_back(broad_chunk[IDX - 1]);
        add_dim.push_back(small_chunk[IDX]); // for no match, add 0??

      }
      ++IDX;
    }

    // THIS IS HOW YOU COMPUTE ALL THE BROADCAST DIM AND ITS PREV
    for (int i = 0; i < broad_cast_dim.size(); ++i) {
      std::cout << broad_cast_dim[i] << ", " << broad_cast_dim_prev[i] << ".." << broad_cast_dim_count[i] << "\t";
    }
    std::cout << std::endl;



    // do {
    //   // for (auto i : idx) {
    //   //   std::cout << i << ",\t";
    //   // }
    //   size_t al;
    //   std::cout << ":: " << (al = IndicesToAddress(broad, broad_chunk, idx) ) << "\t";

    //   auto cut = CutBroadcast(idx, small);
    //   size_t OTHER;
    //   std::cout << " AND " << (OTHER = IndicesToAddress(small, small_chunk, cut)) << "\t::";
    //   std::cout << (al - OTHER) << " is diff";
    //   // for (auto i : cut) {
    //   //   std::cout << i << ",\t";
    //   // }

    //   size_t res = 0;
    //   size_t comp = al;

    //   // same thing down to lower broadcast sizes do this but upwards?
    //   // comp %= broad_chunk[0];

    //   // if (comp / broad_chunk[2] != broad[2]) {
    //   //   res += broad_chunk[2];
    //   // }
    //   // comp %= broad_chunk[2];




    //   // FOR EACH BROADCAST SHAPE,
    //   /*
    //     starting from top
    //     if broadcast occured at i, if divisible by chunk[i - 1], add chunk[i]
    //     irrgardless, address /= chunk[i]

    //     repeat for all broadcast dim

    //     at the end, add remaining from address
    //   */
    //   // if (comp / cap) {
    //   //   res += broad_chunk[0];
    //   // }
    //   // comp %= broad_chunk[0];
    //   // // when it is divisible by 24, add 8
    //   // if (comp / broad_chunk[1]) {
    //   //   res += broad_chunk[2];
    //   // }
    //   // comp %= broad_chunk[2];

    //   // FOREACH BROADCAST DIMS
    //   // for (int i = 0; i < broad_cast_dim.size(); ++i) {
    //   //   if (comp / broad_cast_dim_prev[i]) {
    //   //     // res += broad_cast_dim[i];
    //   //     res += add_dim[i];
    //   //   }
    //   //   comp %= broad_cast_dim[i];
    //   // }
    //   // int size = broad_cast_dim_prev.size();
    //   // for (int i = 0; i < size; ++i) {
    //   // // for (int i = size - 1; i >= 0; --i) {
    //   //   int temp = comp % broad_cast_dim_prev[i];
    //   //   temp = comp / broad_cast_dim[i];
    //   //   if (temp) {
    //   //     std::cout << "GOD ";
    //   //     res += broad_cast_dim[i];
    //   //   }
    //   //   comp %= broad_cast_dim[i];
    //   //   // clearly some repeated division is fucking this up
    //   // }
    //   // res += comp;



      
    //   std::cout << " ummm " << (res) << " DIFF " << (static_cast<int>(OTHER) - static_cast<int>(res))  << ((OTHER != res) ? " WRONG" : " " )<< std::endl;

    //   std::cout << std::endl;

    // } while (IncrementIndicesByShape(broad.begin(), broad.end(), idx.begin(), idx.end()));


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


        int recomp = broadcast_address;

        for (int i = 0; i < broad_cast_dim.size(); ++i) {

          int computed_difference = 0;
          // try catch last skip
          //  should be add,add,skip, add,add,skip  because broadasted tp 3 with intervals of 2
          int temp = recomp / broad_cast_dim[i];
          // std::cout << "Rep is " << temp;
          // NEED TO TAKE AWAY EVERY THIRD
          int temptemp = temp / broad_cast_dim_count[i];
          // std::cout << "ev is " << temptemp;

          int diff_by_this = temp - temptemp;
          // std::cout << "\t\t" << "diff by that is " << diff_by_this;

          
          computed_difference += broad_cast_dim[i] * diff_by_this;


          // std::cout << "\t\t\t Computed is " << computed_difference;

          recomp -= computed_difference;
        }

        std::cout << "\t\t\t fin is " << recomp;
        std::cout << ((recomp == cut_address) ? " Corr" : "WRONG");

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