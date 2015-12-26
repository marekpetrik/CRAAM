// Simple development tests

#include <iostream>
#include <iterator>
#include <random>
#include <cmath>
#include <cassert>

#include "ImMDP.hpp"
#include "cpp11-range-master/range.hpp"

using namespace std;
using namespace craam;
using namespace craam::impl;
using namespace util::lang;

template<class T>
void print_vector(vector<T> vec){
    for(auto&& p : vec){
        cout << p << " ";
    }
}

int main_im(){
    const double discount = 0.9;

    cout << "Running ... " << endl;

    auto mdpi = MDPI_R::from_csv_file("mdp.csv","observ.csv","initial.csv");

    cout << "States: " << mdpi->state_count() << " Observations: " << mdpi->obs_count() << endl;
    cout << "Solving MDP ..." << endl;

    auto mdp = mdpi->get_mdp();
    auto&& sol = mdp->mpi_jac_ave(numvec(0), discount);
    auto&& initial = mdpi->get_initial();

    cout << "Return: " << sol.total_return(initial) << endl;
    cout << "Policy: ";
    print_vector(sol.policy);
    cout << endl;

    // check that the policy is correct
    auto res = mdp->assert_policy_correct(indvec(mdp->state_count(), 0), indvec(mdp->state_count(), 0));
    assert(res == -1);

    auto sol_base = mdp->vi_jac_fix(numvec(0),discount,indvec(mdp->state_count(), 0),
                                    indvec(mdp->state_count(), 0));
    cout << "Baseline policy return: " << sol_base.total_return(initial) << endl;

    cout << "Solving constrained MDP ... " << endl;

    for(auto i : range(0,5)){
        auto pol = mdpi->solve_reweighted(i,0.9);
        cout << "Iteration: " << i << "  :  ";
        print_vector(pol);
        cout << endl;
    }

    auto pol = mdpi->solve_reweighted(10,discount);

    auto sol_impl = mdp->vi_jac_fix(numvec(0),discount, mdpi->obspol2statepol(pol),
                    indvec(mdp->state_count(), 0));

    cout << "Return implementable: " << sol_impl.total_return(initial) << endl;

    cout << "Generating implementable policies (randomly) ..." << endl;

    auto max_return = 0.0;
    indvec max_pol(mdpi->obs_count(),-1);

    for(auto i : range(0,20000)){
        auto rand_pol = mdpi->random_policy();

        auto ret = mdp->vi_jac_fix(numvec(0),discount, mdpi->obspol2statepol(rand_pol),
                    indvec(mdp->state_count(), 0)).total_return(initial);

        if(ret > max_return){
            max_pol = rand_pol;
            max_return = ret;
        }

        //cout << "index " << i << " return " << ret << endl;
    }

    cout << "Maximal return " << max_return << endl;
    cout << "Best policy: ";
    print_vector(max_pol);
    cout << endl;

    return 0;

}

int main(void){

    vector<Action> a(3);
    cout << "Size is: " << a[2].get_threshold() << endl;

    RMDP rmdp(3);

    // nonrobust
    // action 1 is optimal, with transition matrix [[0,1,0],[0,0,1],[0,0,1]] and rewards [0,0,1.1]
    rmdp.add_transition_d(0,1,1,1,0);
    rmdp.add_transition_d(1,1,2,1,0);
    rmdp.add_transition_d(2,1,2,1,1.1);

    rmdp.add_transition_d(0,0,0,1,0);
    rmdp.add_transition_d(1,0,0,1,1);
    rmdp.add_transition_d(2,0,1,1,1);

    Transition init_d({0,1,2},{1.0/3.0,1.0/3.0,1.0/3.0},{0,0,0});

    numvec initial{0,0,0};

    // average
    auto re = rmdp.vi_gs_ave(initial,0.9, 10000,0);

    int x;

    cout << re.valuefunction[0] << "; " << x << endl;

    return 0;
}
