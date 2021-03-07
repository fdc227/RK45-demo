#include "main.hpp"
#define N 30
#define NRHS 1
#define LDA N
#define LDB NRHS
using namespace std;
using namespace boost::numeric::odeint;
vector<double> A(LDA * N);
vector<double> rhs(N);
vector<double> var_list(2*N);
void initial_condition(vector<double>& var)
{
    for (int i = 0; i < var.size(); i++)
    {
        if (i < var.size() / 2)
        {
            if (i % 3 == 0)
                var[i] = 0.0;
            else if (i % 3 == 1)
                var[i] = 0.0;
            else
                var[i] = 0.1 * (double)(i + 1) / 3;
        }
        else
            var[i] = 0.0;
    }
}
void ptr_array_to_console(const vector<double>& x, const double t) 
{
    cout << t<< ' ';
    for (int i = 0; i < N; i++)
    {
        cout << x[i] << ' ';
    }
    cout << endl;
}
void ODE_dydt(const vector<double>& var_list, vector<double>& dydx, double t)
{
    MKL_INT n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
    MKL_INT ipiv[N];
    A_func(var_list, A);
    RHS_func(var_list, rhs);
    info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, &*A.begin(), lda, ipiv, &*rhs.begin(), ldb);
    for(int i = 0; i < N; i ++)
    {
    dydx[i] = var_list[i + N];
    dydx[i + N] = rhs[i];
    }
}
int main(void)
{
    initial_condition(var_list);
    //integrate(ODE_dydt, var_list, 0.0, 10.0, 0.1, ptr_array_to_console);
    auto pt1 = chrono::high_resolution_clock::now();
    integrate_const(runge_kutta4<vector<double>>(), ODE_dydt, var_list, 0.0, 5.0, .001, ptr_array_to_console);
    auto pt2 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>(pt2 - pt1);
    cout << "time of odeint is " << duration1.count() << endl;

    initial_condition(var_list);
    auto pt3 = chrono::high_resolution_clock::now();
    RK45(ODE_dydt, var_list, 0.0, 5.0, .001, ptr_array_to_console);
    auto pt4 = chrono::high_resolution_clock::now();

    auto duration2 = chrono::duration_cast<chrono::microseconds>(pt4 - pt3);


    cout << "time of RK45 is " << duration2.count() << endl;

    return 0;
}
