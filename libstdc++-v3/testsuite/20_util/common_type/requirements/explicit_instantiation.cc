// { dg-do compile { target c++11 } }

// Copyright (C) 2008-2025 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along
// with this library; see the file COPYING3.  If not see
// <http://www.gnu.org/licenses/>.

// NB: This file is for testing type_traits with NO OTHER INCLUDES.

#include <type_traits>

namespace std
{
  typedef int test_type1;
  typedef int& test_type2;
  typedef double test_type3;
  typedef float test_type4;
  typedef void test_type5;
  typedef const void test_type6;  

  template struct common_type<>;
  template struct common_type<test_type1>;
  template struct common_type<test_type1, test_type2>;
  template struct common_type<test_type1, test_type2, test_type3>;
  template struct common_type<test_type1, test_type2, test_type3, test_type4>;

  template struct common_type<test_type5>;
  template struct common_type<test_type5, test_type6>;
}
