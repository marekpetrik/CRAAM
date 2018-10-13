// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "craam/definitions.hpp"
#include "craam/Transition.hpp"
#include <functional>


namespace craam::algorithms{

using namespace std;
using namespace craam;

// *******************************************************
// Abstract nature specification
// *******************************************************


/**
 * Defines s,a-rectangular nature to be a function that is given a state index, action index,
 * the transition probability, and a z-function. It returns an the optimal transition probability
 * (usually the worst-case probabilities), and the value of the update.
 *
 * The z-function is rewards + discount * valuefunction. It is defined only for the same states that have non-zero
 * transition probabilites.
 */
using SANature = function<pair<numvec,prec_t>
                   (long stateid,long actionid,
                    const numvec& nominalprob,
                    const numvec& zvalues)>;

/**
 * Defines s-rectangular nature to be a function that is given a state index,
 * the transition probability, a z-function
 * for each state and action. It returns
 *  1) the optimal action distribution,
 *  2) the optimal transition probability (usually the worst-case probabilities) for each action.
 *      the vector could have length one when the action has 0 probability,
 *  3) the value of the update.
 *
 * The zvalues is rewards + discount * valuefunction. It is defined only for the same states that have non-zero
 * transition probabilites.
 */
using SNature = function<tuple<numvec,vector<numvec>,prec_t>
                   (long stateid,
                    const vector<numvec>& nominalprobs,
                    const vector<numvec>& zvalues)>;


}
