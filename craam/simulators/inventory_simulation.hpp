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

#include "../Samples.hpp"
#include "../definitions.hpp"

#include <utility>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>
#include <cmath>
#include <string>


namespace craam{
namespace msen {

using namespace std;
using namespace util::lang;

template<class Sim>
class InventoryPolicy{

public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;

    InventoryPolicy(const Sim& sim, long max_inventory,
                    random_device::result_type seed = random_device{}()) :
                sim(sim), max_inventory(max_inventory), gen(seed){
    }

    /** Returns an action accrding to the S,s policy, orders required amount to have
    the inventory to max level. */
    long operator() (long current_state){
        return max(0l, max_inventory-current_state);
    }

private:
    /// Internal reference to the originating simulator
    const Sim& sim;
    long max_inventory;
    /// Random number engine
    default_random_engine gen;
};

/**
A simulator that generates inventory data.

*/
class InventorySimulator{

public:
    /// Type of states
    typedef long State;
    /// Type of actions
    typedef long Action;

    /**
    Build a model simulator for the inventory problem

    The initial inventory level is 0
    */
    InventorySimulator(long initial, prec_t prior_mean, prec_t prior_std, prec_t demand_std, prec_t purchase_cost,
                       prec_t sale_price, prec_t delivery_cost, prec_t holding_cost, prec_t backlog_cost,
                       long max_inventory, long max_backlog, long max_order, random_device::result_type seed = random_device{}()) :
                initial(initial), prior_mean(prior_mean), prior_std(prior_std), demand_std(demand_std),
                purchase_cost(purchase_cost), sale_price(sale_price), delivery_cost(delivery_cost), holding_cost(holding_cost),
                backlog_cost(backlog_cost), max_inventory(max_inventory), max_backlog(max_backlog), max_order(max_order),
                gen(seed), inventory_status(initial) {

                init_demand_distribution();
    }

    /**
    Build a model simulator

    The initial inventory level is 0
    */
    InventorySimulator(long initial, prec_t prior_mean, prec_t prior_std, prec_t demand_std, prec_t purchase_cost,
                       prec_t sale_price, long max_inventory, random_device::result_type seed = random_device{}()) :
        initial(initial), prior_mean(prior_mean), prior_std(prior_std), demand_std(demand_std), purchase_cost(purchase_cost),
        sale_price(sale_price), max_inventory(max_inventory), gen(seed), inventory_status(initial) {

        init_demand_distribution();
    }

    long init_state() const{
        return initial;
    }

    void init_demand_distribution(){
        normal_distribution<prec_t> prior_distribution(prior_mean,prior_std);
        demand_mean = prior_distribution(gen);
        demand_distribution = normal_distribution<prec_t>(demand_mean, demand_std);
    }

    bool end_condition(State s) const
        {return inventory_status<0;}

    /**
    Returns a sample of the reward and a decision state following a state
    \param current_inventory Current inventory level
    \param action_order Action obtained from the policy
    \returns a pair of reward & next inventory level
    */
    pair<double,int> transition(long current_inventory, long action_order){

        assert(current_inventory >= 0 );
        assert(action_order >= 0);
		
		///Genrate demand from the normal demand distribution
        long demand = max(0l,(long)demand_distribution(gen));
        
        ///Compute the next inventory level
        long next_inventory = action_order + current_inventory - demand;
        
        ///Back calculate how many items were sold
        long sold_amount = current_inventory-next_inventory + action_order;
        
        ///Compute the obtained revenue
        prec_t revenue = sold_amount * sale_price;
        
        ///Compute the expense
        prec_t expense = action_order * purchase_cost;
        
        ///Reward is equivalent to the profit & obtained from revenue & total expense
        prec_t reward = revenue - expense;
        
        ///Keep track of the current inventory level
        inventory_status = next_inventory;

        return make_pair(reward, next_inventory);
    }

protected:
    long initial;
    ///Distribution for the demand
    normal_distribution<prec_t> demand_distribution;
    ///Distribution parameters
    prec_t prior_mean, prior_std, demand_std, demand_mean;
    prec_t purchase_cost, sale_price, delivery_cost, holding_cost, backlog_cost;
    long max_inventory, max_backlog, max_order;
    /// Random number engine
    default_random_engine gen;
    long inventory_status;
};

///Inventory policy to be used
using ModelInventoryPolicy = InventoryPolicy<InventorySimulator>;

} // end namespace msen
} // end namespace craam
