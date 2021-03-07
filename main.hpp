#include <iostream>
#include <vector>
#include <cmath>
#include "mkl_lapacke.h"
#include "mkl.h"
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <chrono>

#define rho 7800
#define E 2.0e11
#define I 7.85e-05
#define L 1
#define c 1
#define G 7.6923e10
#define J 0.000157
#define ea 0.4
void A_func(const std::vector<double>&, std::vector<double>& );
void RHS_func(const std::vector<double>&, std::vector<double>& );

typedef void (*ode_ptr)(const std::vector<double>&, std::vector<double>&, double); //state_vec, dydt, t
typedef void (*ode_print)(const std::vector<double>&, double); // state_vec, t

void RK45(ode_ptr ODE_func, const std::vector<double>& init_vec, double t_start, double t_end, double t_step, ode_print PTR_func);