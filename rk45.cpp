#include "main.hpp"

using namespace std;

typedef void (*ode_ptr)(const vector<double>&, vector<double>&, double); //state_vec, dydt, t
typedef void (*ode_print)(const vector<double>&, double); // state_vec, t

void RK45(ode_ptr ODE_func, const vector<double>& init_vec, double t_start, double t_end, double t_step, ode_print PTR_func)
{
	double tol = 1.0e-7;
	MKL_INT vec_size = init_vec.size();
	MKL_INT incx = 1;
	MKL_INT incy = 1;
	vector<double> k1(vec_size);
	vector<double> k12(vec_size);
	vector<double> k2(vec_size);
	vector<double> k23(vec_size);
	vector<double> k3(vec_size);
	vector<double> k34(vec_size);
	vector<double> k4(vec_size);

	double t = t_start;
	vector<double> state_vec = init_vec;
	PTR_func(state_vec, t);
	t += t_step;
	t_end += tol;
	while (t <= t_end)
	{
		cblas_dcopy(vec_size, &*state_vec.begin(), incx, &*k12.begin(), incy); // k12 = state_vec;
		cblas_dcopy(vec_size, &*state_vec.begin(), incx, &*k23.begin(), incy); // k23 = state_vec;
		cblas_dcopy(vec_size, &*state_vec.begin(), incx, &*k34.begin(), incy); // k34 = state_vec;

		ODE_func(state_vec, k1, t); // k1 = f(t, y)
		cblas_daxpy(vec_size, t_step / 2.0, &*k1.begin(), incx, &*k12.begin(), incy); // k12 = k12 + t_step/2*k1
		ODE_func(k12, k2, t + t_step / 2.0); // k2 = f(t + h/2, y + h*k1/2)
		cblas_daxpy(vec_size, t_step / 2.0, &*k2.begin(), incx, &*k23.begin(), incy); // k23 = k23 + t_step/2*k2
		ODE_func(k23, k3, t + t_step / 2.0); // k3 = f(t + h/2, y + h*k2/2)
		cblas_daxpy(vec_size, t_step, &*k3.begin(), incx, &*k34.begin(), incy); // k34 = k34 + t_step*k3
		ODE_func(k34, k4, t + t_step); // k4 = f(t + h, y + h*k3)

		cblas_daxpy(vec_size, t_step / 6.0, &*k1.begin(), incx, &*state_vec.begin(), incy);
		cblas_daxpy(vec_size, t_step / 3.0, &*k2.begin(), incx, &*state_vec.begin(), incy);
		cblas_daxpy(vec_size, t_step / 3.0, &*k3.begin(), incx, &*state_vec.begin(), incy);
		cblas_daxpy(vec_size, t_step / 6.0, &*k4.begin(), incx, &*state_vec.begin(), incy);

		PTR_func(state_vec, t);

		t += t_step;

	}
}