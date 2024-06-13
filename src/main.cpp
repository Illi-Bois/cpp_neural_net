// #include <iostream>
// #include <initializer_list>
// #include <vector>


// // TESTZONE, modify as you want to test out ideas.
// template<typename T=int>
// class A {
//  public:
//   A mult(A other);

//   template<class Derived>
//   Derived genMult(Derived d);
// };


// template<typename T=int>
// class B : public A<T> {
// };

// template<typename T>
// A<T> A<T>::mult(A<T> other) {
//   std::cout << "okay" << std::endl;
// }

// template<typename T>
// template<class D>
// D A<T>::genMult(D d) {
//   static_assert(std::is_base_of<A, D>::value, "Error: not a A");
//   std::cout << "weird" << std::endl;
//   return d;
// }


// void test(int arr...) {
//   std::cout << arr << std::endl;
//   std::cout << *(&arr + sizeof(int)*2) << std::endl;
// }

// // void a1(std::initializer_list<int> arr) {
// //   std::cout << "This is init list" << std::endl;
// //   std::cout << arr[0] << std::endl;
// // }
// void a1(std::vector<int> arr) {
//   std::cout << "This is vec" << std::endl;
//   std::cout << arr.size() << std::endl;
// }

// int main() {
//   std::cout << "Hello World" << std::endl;
//   // Model model;

//   // model.addLayer(new Layer(255, 25))
//   //      .addLayer(new Sogmoid())
//   //      .addLayer(new Layer(255, 25))
//   //      .addLayer(new Sogmoid())
//   //      .addLayer(new Layer(255, 25));

//   // A<> a;
//   // B<> b;

//   // A<> c = b.mult(b);
//   // A<> d = a.mult(b);
//   // B<> e = a.genMult(b);
//   // A<> f = a.genMult(a);

//   // int k = 0;
//   // int g = a.genMult(k);


//   // test(1, 2, 3);

//   // a1({1,2,3});


//   std::vector<int> arr(-1);
//   // arr.resize(10);

//   arr[0] = 1;
//   arr[1] = 2;

//   std::cout << arr[0] << std::endl;
//   std::cout << arr[1] << std::endl;
//   std::cout << arr[2] << std::endl;
//   std::cout << arr.size() << std::endl;
// }

// C++ code to demonstrate copy of vector 
// by assign() 
#include<iostream> 
#include<vector> // for vector 
#include<algorithm> // for copy() and assign() 
#include<iterator> // for back_inserter 
using namespace std; 

int main() 
{ 
	// Initializing vector with values 
	vector<int> vect1{1, 2, 3, 4}; 

	// Declaring another vector 
	vector<int> vect2{1,2,3,4,5,6,7,8,9}; 

	// Copying vector by assign function 
  vect2.insert(vect2.begin(), vect1.begin(), vect1.end() - 2); 

	cout << "Old vector elements are : "; 
	for (int i=0; i<vect1.size(); i++) 
		cout << vect1[i] << " "; 
	cout << endl; 

	cout << "New vector elements are : "; 
	for (int i=0; i<vect2.size(); i++) 
		cout << vect2[i] << " "; 
	cout<< endl; 

	// Changing value of vector to show that a new 
	// copy is created. 
	vect1[0] = 2; 

	cout << "The first element of old vector is :"; 
	cout << vect1[0] << endl; 
	cout << "The first element of new vector is :"; 
	cout << vect2[0] <<endl; 

	return 0; 
} 
